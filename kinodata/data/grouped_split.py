from typing import List
import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import GroupKFold

from .data_split import Split


def _split_random(a: np.ndarray, r: float, seed: int = 0):
    rng = default_rng(seed)
    k = int(a.shape[0] * r)
    permuted = rng.permutation(a)
    return permuted[:k], permuted[k:]


def group_k_fold_split(group_index: np.ndarray, k: int) -> List[Split]:
    group_k_fold = GroupKFold(k)
    generator = group_k_fold.split(
        np.zeros((group_index.shape[0], 1)), groups=group_index
    )
    return [
        Split(train_index, *_split_random(test_index, 0.5))
        for train_index, test_index in generator
    ]
