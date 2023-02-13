from kinodata.data.dataset import KinodataDocked, _DATA
from torch_geometric.data import Data, HeteroData
import torch

from typing import Callable
import copy

import numpy as np

def replace_ligand(original: HeteroData, replacement: HeteroData) -> HeteroData:
    center_original = original["ligand"].pos.mean(dim=0, keepdim=True)
    center_replacement = replacement["ligand"].pos.mean(dim=0, keepdim=True)
    modified = copy.copy(original)
    modified["ligand"].z = replacement["ligand"].z
    modified["ligand"].pos = replacement["ligand"].pos - center_replacement + center_original
    return modified

class ContrastiveKinodataDocked(KinodataDocked):
    def __init__(
        self,
        root: str = _DATA,
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
        percentage_valid: float = 0.5,
        seed: int = 0,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.percentage_valid = percentage_valid
        self.rng = np.random.default_rng(seed)

    def get(self, index: int) -> Data:
        original = super().get(index)
        p = self.rng.uniform()
        if p < self.percentage_valid:
            valid_docking = torch.ones(1)
        else:
            random_index = self.rng.integers(0, len(self))
            replacement = super().get(random_index)
            original = replace_ligand(original, replacement)
            valid_docking = torch.zeros(1)
        
        original.valid_docking = valid_docking
        return original


    