import wandb

from typing import Any, Dict, List, Optional, Protocol, Sequence
import torch
from torch_geometric.data.lightning_datamodule import LightningDataset
from torch.utils.data import random_split
import wandb
from torch_geometric.loader.dataloader import DataLoader
from torch.utils.data import Dataset, Subset


class SplittingMethod(Protocol):
    def __call__(
        self, dataset: Dataset, lengths: Sequence[int], generator: torch.Generator
    ) -> List[Subset]:
        ...


def make_data_module(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    train_size: float = 1.0,
    val_size: Optional[float] = None,
    test_size: Optional[float] = None,
    seed: int = 0,
    splitting_method: SplittingMethod = random_split,
) -> LightningDataset:

    split = dict()
    split["train_dataset"] = int(train_size * len(dataset))
    if val_size is not None:
        split["val_dataset"] = int(val_size * len(dataset))
    if test_size is not None:
        split["test_dataset"] = int(test_size * len(dataset))
    split["train_dataset"] += len(dataset) - sum(split.values())

    kwargs: Dict[str, Any] = dict()
    for split_name, subset in zip(
            split.keys(),
            splitting_method(
                dataset,
                list(split.values()),
                generator=torch.Generator().manual_seed(seed),
            ),
    ):
        kwargs[split_name] = subset
    
    kwargs["batch_size"] = batch_size
    kwargs["num_workers"] = num_workers

    return LightningDataset(**kwargs)


class ComposedDatamodule(LightningDataset):
    def __init__(self, data_modules: List[LightningDataset]):
        self.data_modules = data_modules

    def train_dataloader(self) -> DataLoader:
        return [data_module.train_dataloader() for data_module in self.data_modules]

    def val_dataloader(self) -> DataLoader:
        return [data_module.val_dataloader() for data_module in self.data_modules]

    def test_dataloader(self) -> DataLoader:
        return [data_module.test_dataloader() for data_module in self.data_modules]
