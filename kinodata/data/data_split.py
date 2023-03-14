from dataclasses import dataclass
from typing import List, Optional, Protocol, Union, Mapping, TypeVar, Dict, Any, Generic
import json
from pathlib import Path
import pandas as pd

import numpy as np
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset

IndexLike = Union[Tensor, ndarray, List[int]]
PathLike = Union[Path, str]

from dataclasses import dataclass, asdict, field


Kwargs = Dict[str, Any]
PathLike = Union[Path, str]
IndexType = TypeVar("IndexType")
OtherIndexType = TypeVar("OtherIndexType")


@dataclass(repr=False)
class Split(Generic[IndexType]):

    train_split: List[IndexType]
    val_split: Optional[List[IndexType]] = field(default_factory=list)
    test_split: Optional[List[IndexType]] = field(default_factory=list)

    def __post_init__(self):
        self.train_split = list(self.train_split)
        self.val_split = list(self.val_split)
        self.test_split = list(self.test_split)

    @classmethod
    def random_split(
        cls, num_train: int, num_val: int, num_test: int, seed: int = 0
    ) -> "Split[int]":
        rng = np.random.default_rng(seed)
        num = num_train + num_val + num_test
        index = np.arange(num)
        rng.shuffle(index)

        return cls(
            index[0:num_train].tolist(),
            index[num_train : (num_train + num_val)].tolist(),
            index[(num_train + num_val) :].tolist(),
        )

    def remap_index(
        self, mapping: Mapping[IndexType, OtherIndexType]
    ) -> "Split[OtherIndexType]":
        return Split(
            [mapping[t] for t in self.train_split],
            [mapping[t] for t in self.val_split],
            [mapping[t] for t in self.test_split],
        )

    def to_data_frame(self) -> pd.DataFrame:
        full_split = self.train_split + self.val_split + self.test_split
        split_assignment = (
            ["train"] * len(self.train_split)
            + ["val"] * len(self.val_split)
            + ["test"] * len(self.test_split)
        )
        return pd.DataFrame({"ident": full_split, "split": split_assignment})

    @classmethod
    def from_data_frame(cls, df: pd.DataFrame):
        return cls(
            df[df.split == "train"].ident.values.tolist(),
            df[df.split == "val"].ident.values.tolist(),
            df[df.split == "test"].ident.values.tolist(),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{type(self.train_split[0])}](train={len(self.train_split)}, val={len(self.val_split)}, test={len(self.test_split)})"


class SplittingMethod(Protocol):
    def __call__(self, dataset: Dataset, seed: int = 0) -> Split:
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

    def __call__(self, dataset: Dataset, seed: int = 0) -> Split:
        split_sizes = {"train": self.train_size}
        if self.val_size is not None:
            split_sizes["val"] = self.val_size
        if self.test_size is not None:
            split_sizes["test"] = self.test_size

        for key, value in split_sizes.items():
            split_sizes[key] = int(value * len(dataset))

        # make sure split is congruent
        split_sizes["train"] -= sum(split_sizes.values()) - len(dataset)

        split = Split.random_split(
            num_train=split_sizes["train"],
            num_val=split_sizes["val"],
            num_test=split_sizes["test"],
            seed=seed,
        )
        return split
