from typing import Any, Dict, List
from torch_geometric.data.lightning_datamodule import LightningDataset
from torch_geometric.loader.dataloader import DataLoader
from torch.utils.data import Dataset

from kinodata.data.data_split import SplittingMethod


def make_data_module(
    dataset: Dataset,
    splitting_method: SplittingMethod,
    batch_size: int,
    num_workers: int,
    seed: int = 0,
    **kwargs
) -> LightningDataset:
    split = splitting_method(dataset, seed)
    return LightningDataset(
        train_dataset=split.train,
        val_dataset=split.val,
        test_dataset=split.test,
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
