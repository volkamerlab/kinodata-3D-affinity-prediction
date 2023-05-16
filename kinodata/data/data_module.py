from typing import Any, Dict, List, Optional

from torch_geometric.data import InMemoryDataset
from torch_geometric.data.lightning_datamodule import LightningDataset
from torch_geometric.loader.dataloader import DataLoader

from kinodata.data.data_split import Split
from copy import deepcopy

Kwargs = Dict[str, Any]

# def make_data_module(
#     dataset: Dataset,
#     splitting_method: SplittingMethod,
#     batch_size: int,
#     num_workers: int,
#     seed: int = 0,
#     **kwargs
# ) -> LightningDataset:
#     split = splitting_method(dataset, seed)
#     return LightningDataset(
#         train_dataset=split.train,
#         val_dataset=split.val,
#         test_dataset=split.test,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         **kwargs,
#     )


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
    **kwargs
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
        **kwargs
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
