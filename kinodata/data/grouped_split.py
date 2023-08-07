from pathlib import Path
from typing import List, Protocol
import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import GroupKFold, KFold
import pandas as pd

from .data_split import Split
from .utils.cluster import AffinityPropagation
from .utils.similarity import BLOSUMSubstitutionSimilarity
from .dataset import KinodataDocked


def _split_random(a: np.ndarray, r: float, seed: int = 0):
    rng = default_rng(seed)
    k = int(a.shape[0] * r)
    permuted = rng.permutation(a)
    return permuted[:k], permuted[k:]


def _generator_to_list(generator):
    splits = [
        Split(train_index, *_split_random(test_index, 0.5))
        for train_index, test_index in generator
    ]
    return splits


def group_k_fold_split(
    group_index: np.ndarray,
    k: int,
) -> List[Split]:
    group_k_fold = GroupKFold(k)
    _X = np.zeros((group_index.shape[0], 1))
    generator = group_k_fold.split(_X, groups=group_index)
    return _generator_to_list(generator)


def random_k_fold_split(data_index: np.ndarray, k: int) -> List[Split]:
    k_fold = KFold(k, shuffle=True)
    generator = k_fold.split(data_index)
    return _generator_to_list(generator)


class KinodataKFoldSplit:
    """
    Wraps all supported splitting methods.
    and implements dataset-dependent caching.
    Splits are cached in a datasets processed dir.
    """

    pocket_clustering = AffinityPropagation()
    pocket_similarity_measure = BLOSUMSubstitutionSimilarity

    def __init__(self, split_type: str, k: int) -> None:
        assert split_type in (
            "scaffold-k-fold",
            "pocket-k-fold",
            "random-k-fold",
        ), f"Unknown split type {split_type}"
        self.k = k
        self.split_type = split_type

    def cache_dir(self, dataset: KinodataDocked) -> Path:
        return Path(dataset.processed_dir) / self.split_type

    def split_files(self, dataset: KinodataDocked) -> List[Path]:
        cache_dir = self.cache_dir(dataset)
        return [cache_dir / f"{i}:{self.k}.csv" for i in range(1, self.k + 1)]

    def _split(self, dataset: KinodataDocked):
        if self.split_type == "scaffold-k-fold":
            scaffolds, idents = zip(*[(data.scaffold, data.ident) for data in dataset])
            scaffolds = np.array(scaffolds)
            splits = group_k_fold_split(group_index=scaffolds, k=self.k)
            return splits
        if self.split_type == "pocket-k-fold":
            pocket_sequences, idents = zip(
                *[(data.pocket_sequence, data.ident) for data in dataset]
            )
            df_cluster_labels = self.pocket_clustering(
                pd.DataFrame({"pocket_sequence": pocket_sequences, "ident": idents}),
                "pocket_sequence",
                fn_similarity=self.pocket_similarity_measure(),
            )
            return group_k_fold_split(
                df_cluster_labels[self.pocket_clustering.cluster_key].values,
                k=self.k,
            )
        if self.split_type == "random-k-fold":
            idents = [data.ident for data in dataset]
            return random_k_fold_split(idents, self.k)

    def split(self, dataset: KinodataDocked) -> List[Split]:
        split_files = self.split_files(dataset)
        if all(f.exists() for f in split_files):
            return [Split.from_csv(f) for f in split_files]

        splits: List[Split] = self._split(dataset)

        for split, f in zip(splits, split_files):
            if not f.parents[0].exists():
                f.parents[0].mkdir()
            split.to_data_frame().to_csv(f, index=False)
            split.source_file = str(f)

        return splits
