from torch_geometric.data import Data
from kinodata.transform.normalize import GroupNormalizeTarget
from kinodata.data.dataset import KinodataDocked
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import random


def test_two_groups():
    g1 = "group1"
    g2 = "group2"
    group_size = 1000
    group1 = [Data(y=torch.randn(1) * 0.1 + 3, group=g1) for _ in range(group_size)]
    group2 = [Data(y=torch.randn(1) * 0.1 - 2, group=g2) for _ in range(group_size)]
    dataset = group1 + group2
    random.shuffle(dataset)
    norm = GroupNormalizeTarget(group_key="group", scale=True)
    norm._fit(dataset)

    assert np.isclose(norm._group_means[g1], 3, atol=0.1)
    assert np.isclose(norm._group_means[g2], -2, atol=0.1)

    assert np.isclose(norm._group_stds[g1], 0.1, atol=0.1)
    assert np.isclose(norm._group_stds[g2], 0.1, atol=0.1)

    transformed_dataset = [norm(data.clone()) for data in dataset]
    batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False)))
    transformed_batch = next(
        iter(DataLoader(transformed_dataset, batch_size=32, shuffle=False))
    )
    set_group_means = getattr(transformed_batch, norm.group_mean_key)

    def delta_model(data):
        return torch.randn(data.y.size(0)) * 0.1

    model = norm.get_raw_predictor(delta_model, expect_transformed_data=False)
    delta_model_pred = delta_model(batch)
    model_pred = model(batch)

    model = norm.get_raw_predictor(delta_model, expect_transformed_data=True)
    model_pred = model(transformed_batch)


def test_apply():
    dataset = KinodataDocked()
    data = dataset[0]
    t, (dataset,) = GroupNormalizeTarget.apply("pocket_sequence", dataset)
    print(dataset.transform)
    data_ = dataset[0]

    assert hasattr(data_, t.target_delta_key)
    assert hasattr(data_, t.group_mean_key)


if __name__ == "__main__":
    test_two_groups()
    test_apply()
