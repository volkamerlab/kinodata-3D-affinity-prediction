from collections import defaultdict
from typing import Dict, Optional, Protocol, Union
from dataclasses import dataclass

import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset, Subset


@dataclass(repr=False, frozen=True, eq=False)
class DataSplit:
    train: Subset
    val: Optional[Subset] = None
    test: Optional[Subset] = None

    @property
    def dataset(self) -> Dataset:
        return self.train.dataset

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def val_size(self) -> int:
        return len(self.val) if self.val else 0

    @property
    def test_size(self) -> int:
        return len(self.test) if self.test else 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(train={self.train_size}, val={self.val_size}, test={self.test_size})"


class SplittingMethod(Protocol):
    def __call__(self, dataset: Dataset, seed: int = 0) -> DataSplit:
        ...


class RandomSplit:
    def __init__(
        self,
        train_size: float = 1.0,
        val_size: Optional[float] = None,
        test_size: Optional[float] = None,
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def __call__(self, dataset: Dataset, seed: int = 0) -> DataSplit:
        split_sizes = {"train": self.train_size}
        if self.val_size is not None:
            split_sizes["val"] = self.val_size
        if self.test_size is not None:
            split_sizes["test"] = self.test_size

        for key, value in split_sizes.items():
            split_sizes[key] = int(value * len(dataset))

        # make sure split is congruent
        split_sizes["train"] -= sum(split_sizes.values()) - len(dataset)

        data_subsets = random_split(
            dataset,
            list(split_sizes.values()),
            generator=torch.Generator().manual_seed(seed),
        )

        split: Dict[str, Subset] = dict()
        for split_name, split_subset in zip(split_sizes.keys(), data_subsets):
            split[split_name] = split_subset

        return DataSplit(
            split["train"],
            val=split["val"] if "val" in split else None,
            test=split["test"] if "test" in split else None,
        )


class ColdSplit:
    def __call__(self, dataset: Dataset, seed: int = 0) -> DataSplit:
        # TODO
        ...
