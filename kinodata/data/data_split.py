from dataclasses import dataclass
from typing import List, Optional, Union, Mapping, TypeVar, Dict, Any, Generic
from pathlib import Path
import pandas as pd

import numpy as np
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset

IndexLike = Union[Tensor, ndarray, List[int]]
PathLike = Union[Path, str]

from dataclasses import dataclass, field


Kwargs = Dict[str, Any]
PathLike = Union[Path, str]  # type: ignore
IndexType = TypeVar("IndexType")
OtherIndexType = TypeVar("OtherIndexType")


@dataclass(repr=False)
class Split(Generic[IndexType]):

    train_split: List[IndexType]
    val_split: List[IndexType] = field(default_factory=list)  # type: ignore
    test_split: List[IndexType] = field(default_factory=list)  # type: ignore
    source_file: Optional[str] = None

    def __post_init__(self):
        self.train_split = list(self.train_split)
        self.val_split = list(self.val_split)
        self.test_split = list(self.test_split)

    @property
    def train_size(self) -> int:
        return len(self.train_split)

    @property
    def val_size(self) -> int:
        return len(self.val_split)

    @property
    def test_size(self) -> int:
        return len(self.test_split)

    @classmethod
    def random_split(
        cls, num_train: int, num_val: int, num_test: int, seed: int = 0
    ) -> "Split[int]":
        rng = np.random.default_rng(seed)
        num = num_train + num_val + num_test
        index = np.arange(num)
        rng.shuffle(index)

        return cls(  # type: ignore
            index[0:num_train].tolist(),
            index[num_train : (num_train + num_val)].tolist(),
            index[(num_train + num_val) :].tolist(),
        )

    def remap_index(
        self, mapping: Mapping[IndexType, OtherIndexType], strict: bool = False
    ) -> "Split[OtherIndexType]":
        new_splits = [
            [mapping[t] for t in split if (t in mapping or strict)]
            for split in (self.train_split, self.val_split, self.test_split)
        ]
        remapped = Split(
            train_split=new_splits[0],  # type: ignore
            val_split=new_splits[1],
            test_split=new_splits[2],
        )
        remapped.source_file = self.source_file
        return remapped

    def to_data_frame(
        self, split_key: str = "split", index_key: str = "ident"
    ) -> pd.DataFrame:
        full_split = self.train_split + self.val_split + self.test_split
        split_assignment = (
            ["train"] * self.train_size
            + ["val"] * self.val_size
            + ["test"] * self.test_size
        )
        return pd.DataFrame({index_key: full_split, split_key: split_assignment})

    @classmethod
    def from_data_frame(
        cls,
        df: pd.DataFrame,
        split_key: str = "split",
        index_key: str = "ident",
        train_identifier: Any = "train",
        val_identifier: Any = "val",
        test_identifier: Any = "test",
    ):
        return cls(
            df[df[split_key] == train_identifier][index_key].values.tolist(),
            df[df[split_key] == val_identifier][index_key].values.tolist(),
            df[df[split_key] == test_identifier][index_key].values.tolist(),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[{type(self.train_split[0])}](train={len(self.train_split)}, val={len(self.val_split)}, test={len(self.test_split)}, source={self.source_file})"


class RandomSplit:
    def __init__(
        self,
        train_size: float = 1.0,
        val_size: float = 0.0,
        test_size: float = 0.0,
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

    def __call__(self, dataset: Dataset, seed: int = 0) -> Split:
        split_sizes = {"train": self.train_size}
        if self.val_size > 0:
            split_sizes["val"] = self.val_size
        if self.test_size > 0:
            split_sizes["test"] = self.test_size

        for key, value in split_sizes.items():
            split_sizes[key] = int(value * len(dataset))  # type: ignore

        # make sure split is congruent
        split_sizes["train"] -= sum(split_sizes.values()) - len(
            dataset  # type : ignore
        )

        split = Split.random_split(
            num_train=split_sizes["train"],
            num_val=split_sizes["val"],
            num_test=split_sizes["test"],
            seed=seed,
        )
        return split
