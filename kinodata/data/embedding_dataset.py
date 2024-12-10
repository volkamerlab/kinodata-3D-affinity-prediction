import torch
from torch import nn, Tensor
from pathlib import Path
from typing import Callable, Deque, Optional

from torch_geometric.data.on_disk_dataset import OnDiskDataset
from torch_geometric.loader import DataLoader

from kinodata.data.dataset import KinodataDocked
from kinodata.model.complex_transformer import ComplexTransformer
from kinodata.model.sparse_transformer import SPAB
import re

OptCallable = Optional[Callable]


class EmbeddingDataset(OnDiskDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(EmbeddingDataset, self).__init__(
            root, transform, pre_transform, pre_filter
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        data_list = []


class HookedModule(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self._stored_outputs = Deque()

    def forward(self, *args, **kwargs):
        outputs = self.module(*args, **kwargs)
        self._stored_outputs.append(outputs)
        return outputs
    
    def pop


def generate_embedding_dataset
    model: ComplexTransformer,
    dataset: KinodataDocked,
    root: str,
    pre_transform: OptCallable = None,
    pre_filter: OptCallable = None,
) -> tuple[EmbeddingDataset, str]:
    raw_dir = Path(root) / "raw"
    loader = DataLoader(dataset, batch_size=64)
    for data in loader:
        ...
