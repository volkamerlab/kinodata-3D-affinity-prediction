from copy import deepcopy
import copy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

import pandas as pd
import numpy as np
from torch_geometric.data import InMemoryDataset

# from torch_geometric.data.lightning_datamodule import LightningDataset
from torch_geometric.data.lightning import LightningDataset
from torch_geometric.data.dataset import IndexType
from torch_geometric.transforms import Compose

from kinodata.configuration import Config
from kinodata.data.data_split import Split
from kinodata.data.grouped_split import KinodataKFoldSplit
from kinodata.data.dataset import (
    KinodataDocked,
    Filtered,
)
import kinodata.transform as T
from kinodata.transform.normalize import GroupNormalizeTarget
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


def create_dataset(
    dataset: InMemoryDataset,
    transform: Optional[Callable] = None,
    split: Union[int, np.integer, IndexType, None] = None,
    subset_data: int = 0,
) -> Optional[InMemoryDataset]:
    if split is None:
        return None
    dataset = copy.copy(dataset)
    dataset.transform = transform
    dataset = dataset[split]
    if subset_data > 0:
        dataset = dataset[:subset_data]
    return dataset


def make_data_module(
    split: Split,
    batch_size: int,
    num_workers: int,
    dataset: InMemoryDataset,
    train_transform: Callable = None,
    val_transform: Callable = None,
    test_transform: Callable = None,
    one_time_transform: Optional[Callable[[InMemoryDataset], InMemoryDataset]] = None,
    subset_data: int = 0,
    **kwargs,
) -> LightningDataset:
    dataset = one_time_transform(dataset) if one_time_transform else dataset
    train_dataset = create_dataset(
        dataset, train_transform, split.train_split, subset_data
    )
    assert train_dataset is not None
    val_dataset = create_dataset(dataset, val_transform, split.val_split, subset_data)
    test_dataset = create_dataset(
        dataset, test_transform, split.test_split, subset_data
    )

    return LightningDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs,
    )


def compose(transforms: Optional[list]) -> Optional[Callable]:
    return None if transforms is None else Compose(transforms)


def load_precomputed_split(config) -> Split:
    if not Path(config.data_split).exists():
        raise FileNotFoundError(config.data_split)
    print(f"Loading split from {config.data_split}..")
    # split that assigns *idents* to train/val/test
    split = Split.from_data_frame(pd.read_csv(config.data_split))
    split.source_file = str(config.data_split)
    return split


def fix_split_for_batch_norm(split: Split, batch_size: int) -> Split:
    def fix(a):
        if len(a) % batch_size == 1:
            a = a[:-1]
        return a

    if split.train_size > 0:
        split.train_split = fix(split.train_split)
    if split.val_size > 0:
        split.val_split = fix(split.val_split)
    if split.test_size > 0:
        split.test_split = fix(split.test_split)
    return split


def make_kinodata_module(
    config: Config, transforms=None, one_time_transform=None, subset_data=0
) -> LightningDataset:
    dataset_cls = partial(KinodataDocked, remove_hydrogen=config.remove_hydrogen)

    if config.filter_rmsd_max_value is not None:
        dataset_cls = Filtered(
            dataset_cls(), T.FilterDockingRMSD(config.filter_rmsd_max_value)
        )

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
    if "perturb_complex_positions" in config and config.perturb_complex_positions > 0.0:
        augmentations.append(
            T.PerturbAtomPositions(
                NodeType.Complex, std=config.perturb_complex_positions
            )
        )

    if config.need_distances:
        ...

    if config.add_docking_scores:
        assert config.need_distances
        raise NotImplementedError

    train_transform = compose(augmentations + transforms)
    val_transform = compose(transforms)

    dataset = dataset_cls()
    if config.data_split is not None:
        split = load_precomputed_split(config)
        print(split)
        print("Remapping idents to dataset index..")
        index_mapping = dataset.ident_index_map()
        split = split.remap_index(index_mapping)
    else:
        splitter = KinodataKFoldSplit(config.split_type, config.k_fold)
        splits = splitter.split(dataset)
        split = splits[config.split_index]

    # dirty batchnorm fix
    split = fix_split_for_batch_norm(split, config.batch_size)

    print("Creating data module:")
    print(f"    split:{split}")
    print(f"    train_transform:{train_transform}")
    print(f"    val_transform:{val_transform}")
    data_module = make_data_module(
        split,
        config.batch_size,
        config.num_workers,
        dataset=dataset,  # type: ignore
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        one_time_transform=one_time_transform,
        subset_data=subset_data,
    )
    if (group_key := config.get("target_normalization_group", None)) is not None:
        # TODO maybe we want to return the transform for further use?
        norm_transform, (train_dataset, val_dataset, test_dataset) = (
            GroupNormalizeTarget.apply(
                group_key,
                data_module.train_dataset,
                data_module.val_dataset,
                data_module.test_dataset,
            )
        )
        data_module.train_dataset = train_dataset
        data_module.val_dataset = val_dataset
        data_module.test_dataset = test_dataset

    return data_module
