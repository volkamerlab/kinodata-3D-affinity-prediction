from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.lightning_datamodule import LightningDataset
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.transforms import Compose

from kinodata.configuration import Config
from kinodata.data.data_split import Split
from kinodata.data.dataset import KinodataDocked
import kinodata.transform as T
from kinodata.types import NodeType

Kwargs = Dict[str, Any]


def assert_unique_value(key: str, *kwarg_dicts: Optional[Kwargs], msg: str = ""):
    values = []
    for kwarg_dict in kwarg_dicts:
        if not kwarg_dict:
            continue
        if key in kwarg_dict:
            values.append(kwarg_dict[key])
    assert len(set(values)) <= 1, msg


def make_data_module(
    split: Split,
    batch_size: int,
    num_workers: int,
    dataset_cls: type[InMemoryDataset],
    train_kwargs: Kwargs,
    val_kwargs: Optional[Kwargs] = None,
    test_kwargs: Optional[Kwargs] = None,
    **kwargs,
) -> LightningDataset:
    assert_unique_value("pre_transform", train_kwargs, val_kwargs, test_kwargs)

    if split.val_split is not None and val_kwargs is None:
        val_kwargs = deepcopy(train_kwargs)

    if split.test_split is not None and test_kwargs is None:
        test_kwargs = deepcopy(val_kwargs)

    train_dataset = dataset_cls(**train_kwargs)[split.train_split]
    val_dataset = (
        dataset_cls(**val_kwargs)[split.val_split]
        if split.val_split is not None
        else None
    )
    test_dataset = (
        dataset_cls(**test_kwargs)[split.test_split]
        if split.test_split is not None
        else None
    )

    return LightningDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs,
    )


class ComposedDatamodule(LightningDataset):
    def __init__(self, data_modules: List[LightningDataset]):
        self.data_modules = data_modules

    def train_dataloader(self) -> DataLoader:
        return [data_module.train_dataloader() for data_module in self.data_modules]

    def val_dataloader(self) -> DataLoader:
        return [data_module.val_dataloader() for data_module in self.data_modules]

    def test_dataloader(self) -> DataLoader:
        return [data_module.test_dataloader() for data_module in self.data_modules]


def compose(transforms: Optional[list]) -> Optional[Callable]:
    return None if transforms is None else Compose(transforms)


def make_kinodata_module(config: Config, transforms=None) -> LightningDataset:
    if transforms is None:
        transforms = []

    augmentations = []

    if config.perturb_ligand_positions and config.need_distances:
        augmentations.append(
            T.PerturbAtomPositions(NodeType.Ligand, std=config.perturb_ligand_positions)
        )
    if config.perturb_pocket_positions and config.need_distances:
        augmentations.append(
            T.PerturbAtomPositions(NodeType.Pocket, std=config.perturb_pocket_positions)
        )

    if config.need_distances:
        subset = list(product(config.node_types, config.node_types))
        subset = {
            nt_pair: config.residue_interaction_radius
            if NodeType.PocketResidue in nt_pair
            else config.interaction_radius
            for nt_pair in subset
        }
        if (NodeType.Pocket, NodeType.Pocket) in subset:
            del subset[(NodeType.Pocket, NodeType.Pocket)]
        transforms.append(
            T.AddDistancesAndInteractions(config.interaction_radius, edge_types=subset)
        )

    if config.add_docking_scores:
        assert config.need_distances
        raise NotImplementedError

    train_transform = compose(augmentations + transforms)
    val_transform = compose(transforms)

    if not Path(config.data_split).exists():
        raise FileNotFoundError(config.data_split)

    print(f"Loading split from {config.data_split}..")
    # split that assigns *idents* to train/val/test
    split = Split.from_data_frame(pd.read_csv(config.data_split))
    split.source_file = str(config.data_split)
    print(split)
    print("Remapping idents to dataset index..")
    index_mapping = KinodataDocked().ident_index_map()
    split = split.remap_index(index_mapping)

    print("Creating data module:")
    print(f"    split:{split}")
    print(f"    train_transform:{train_transform}")
    print(f"    val_transform:{val_transform}")
    data_module = make_data_module(
        split,
        config.batch_size,
        config.num_workers,
        dataset_cls=KinodataDocked,
        train_kwargs={"transform": train_transform},
        val_kwargs={"transform": val_transform},
        test_kwargs={"transform": val_transform},
    )

    return data_module
