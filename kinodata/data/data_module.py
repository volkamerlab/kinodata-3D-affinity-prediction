from copy import deepcopy
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.transforms import Compose
try:
    from torch_geometric.data.lightning_datamodule import LightningDataset
except ModuleNotFoundError:  # compatibility
    from torch_geometric.data.lightning import LightningDataset

from kinodata.configuration import Config
from kinodata.data.data_split import Split
from kinodata.data.grouped_split import KinodataKFoldSplit
from kinodata.data.dataset import (
    KinodataDocked,
    Filtered,
)
from kinodata.data.dataset_davids_data import (
    DavidsdataDocked,
    Filtered_david,
)
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
    one_time_transform: Optional[Callable[[InMemoryDataset], InMemoryDataset]] = None,
    **kwargs,
) -> LightningDataset:
    assert_unique_value("pre_transform", train_kwargs, val_kwargs, test_kwargs)

    if split.val_split is not None and val_kwargs is None:
        val_kwargs = deepcopy(train_kwargs)

    if split.test_split is not None and test_kwargs is None:
        test_kwargs = deepcopy(val_kwargs)

    def create_dataset(cls, kwargs, split, ott) -> Optional[InMemoryDataset]:
        if split is None:
            return None
        dataset = cls(**kwargs)
        if ott is not None:
            dataset = ott(dataset)
        return dataset[split]
    
  

    train_dataset = create_dataset(
        dataset_cls, train_kwargs, split.train_split, one_time_transform
    )
    assert train_dataset is not None
    val_dataset = create_dataset(
        dataset_cls, val_kwargs, split.val_split, one_time_transform
    )
    test_dataset = create_dataset(
        dataset_cls, test_kwargs, split.test_split, one_time_transform
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
    config: Config, transforms=None, one_time_transform=None
) -> LightningDataset:
    
    
    dataset_cls_2= partial(DavidsdataDocked, remove_hydrogen=config.remove_hydrogen)
    
    dataset_cls_1 = partial(KinodataDocked, remove_hydrogen=config.remove_hydrogen)
    
    

    #dataset_cls ################

    if config.filter_rmsd_max_value is not None: #not sure if we want this anymore, since now we may not want to filter by RMSD? only appliying it to the kinodata dataset 
        dataset_cls_2 = Filtered_david(
            dataset_cls_2(), T.FilterDockingRMSD(config.filter_rmsd_max_value)
        )
        dataset_cls_1 = Filtered(
            dataset_cls_1(), T.FilterDockingRMSD(config.filter_rmsd_max_value)
        )
    #double check that the output of both datasets is below the RMSD cuttoff!!

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

    
    dataset_1 = dataset_cls_1()
    dataset_2 = dataset_cls_2()
    if config.data_split is not None: #do dataset1 and dataset 2 need to be splitted in the samew way??? ASK
        split = load_precomputed_split(config)
        print("Remapping idents to dataset_1 index..")
        index_mapping_1 = dataset_1.ident_index_map()
        split_1 = split.remap_index(index_mapping_1)
        print(split_1)
        print(f"Split 1: Train size {split_1.train_size}, Val size {split_1.val_size}, Test size {split_1.test_size}")
    else:
        splitter = KinodataKFoldSplit(config.split_type, config.k_fold)
        splits_1 = splitter.split(dataset_1)
        split_1 = splits_1[config.split_index]
    del dataset_1

    if config.data_split is not None:
        split = load_precomputed_split(config)
        print("Remapping idents to dataset_2 index..")
        index_mapping_2 = dataset_2.ident_index_map()
        split_2 = split.remap_index(index_mapping_2)
        print(split_2)
        print(f"Split 2: Train size {split_2.train_size}, Val size {split_2.val_size}, Test size {split_2.test_size}")
    else:
        splitter = KinodataKFoldSplit(config.split_type, config.k_fold)
        splits_2 = splitter.split(dataset_2)
        split_2 = splits_2[config.split_index]
    del dataset_2

    
    # dirty batchnorm fix ---DO I NEED THIS?
    split_1 = fix_split_for_batch_norm(split_1, config.batch_size)
    split_2 = fix_split_for_batch_norm(split_2, config.batch_size)

    print("Creating data module for kinodataset:")
    print(f"    split:{split_1}")
    print(f"    train_transform:{train_transform}")
    print(f"    val_transform:{val_transform}")
    data_module_1 = make_data_module(
        split_1,
        config.batch_size,
        config.num_workers,
        dataset_cls=dataset_cls_1,  # type: ignore
        train_kwargs={"transform": train_transform},
        val_kwargs={"transform": val_transform},
        test_kwargs={"transform": val_transform},
        one_time_transform=one_time_transform,
    )

    print("Creating data module for davidsdockeddataset:")
    print(f"    split:{split_2}")
    print(f"    train_transform:{train_transform}")
    print(f"    val_transform:{val_transform}")
    print('the length of the data set is')
    print(dataset_cls_2)
    data_module_2 = make_data_module(
        split_2,
        config.batch_size,
        config.num_workers,
        dataset_cls=dataset_cls_2,  # type: ignore
        train_kwargs={"transform": train_transform},
        val_kwargs={"transform": val_transform},
        test_kwargs={"transform": val_transform},
        one_time_transform=one_time_transform,
    )

    return data_module_1, data_module_2
