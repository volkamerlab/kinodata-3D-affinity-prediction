import torch
from torch import Tensor
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from ..data.dataset import apply_transform_instance_permament


def get_target(data, storage_key, attr_name):
    if storage_key is not None:
        data = data[storage_key]
    return data[attr_name]


class NormalizeTarget(BaseTransform):

    def __init__(
        self,
        mean: float,
        std: float,
        attr_name: str = "y",
        storage_key: str | None = None,
    ):
        self.mean = mean
        self.std = std
        self.attr_name = attr_name
        self.storage_key = storage_key

    def forward(self, data: Data) -> Data:
        y = get_target(data, self.storage_key, self.attr_name)
        assert isinstance(y, Tensor)
        y = (y - self.mean) / self.std
        if self.storage_key is not None:
            data[self.storage_key][self.attr_name] = y
        else:
            data[self.attr_name] = y
        return data


def normalize_dataset(
    dataset,
    attr_name: str = "y",
    storage_key: str | None = None,
):
    data_list = [data for data in dataset]
    y = torch.cat(
        [get_target(data, storage_key, attr_name) for data in data_list], dim=0
    )
    mean = y.mean().item()
    std = y.std().item()
    transform = NormalizeTarget(mean, std, attr_name, storage_key)
    dataset = apply_transform_instance_permament(dataset, transform)
    return dataset, transform
