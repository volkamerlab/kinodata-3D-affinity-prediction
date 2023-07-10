from functools import reduce
from itertools import takewhile
import numpy as np
from kinodata.data.grouped_split import group_k_fold_split


def test_simple():
    groups = np.random.randint(0, 10, (100,))
    splits = group_k_fold_split(groups, 3)
    assert len(splits) == 3
    for split in splits:
        assert split.train_size > 0, split
        assert split.val_size > 0, split
        assert split.test_size > 0, split


def test_approx_eval_set_size(
    n: int,
    target_eval_set_ratio: int,
    eps: float = 0.03,
    maximum_group_size: float = 0.1,
):
    _maximum_group_size = int(n * maximum_group_size)
    groups = []
    for j in takewhile(lambda _: len(groups) < n, range(n)):
        groups.extend([j] * np.random.randint(0, _maximum_group_size))
    groups = np.array(groups)
    groups = groups[:n]

    for split in group_k_fold_split(groups, target_eval_set_ratio):
        n_test = split.val_size + split.test_size
        frac_test = n_test / (n_test + split.train_size)
        assert np.isclose(frac_test, 1 / target_eval_set_ratio, atol=eps), split


if __name__ == "__main__":
    test_simple()
    test_approx_eval_set_size(
        n=1000,
        target_eval_set_ratio=5,
    )
