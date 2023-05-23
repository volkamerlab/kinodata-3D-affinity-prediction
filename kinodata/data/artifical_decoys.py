"""
DEPRECATED
"""
from typing import Callable
import copy

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData

from kinodata.transform.kabsch import pseudo_kabsch_alignment
from kinodata.data.dataset import KinodataDocked, _DATA


def replace_ligand(original: HeteroData, replacement: HeteroData) -> HeteroData:
    modified = copy.copy(original)

    # this ensures E(3)-invariance of the 'contrastive' / pretraining loss
    aligned_replacement_pos = pseudo_kabsch_alignment(
        replacement["ligand"].pos, original["ligand"].pos
    )

    modified["ligand"].z = replacement["ligand"].z
    modified["ligand"].pos = aligned_replacement_pos

    _, edge_types = original.metadata()
    if ("ligand", "bond", "ligand") in edge_types:
        modified["ligand", "bond", "ligand"].edge_index = replacement[
            "ligand", "bond", "ligand"
        ].edge_index
        modified["ligand", "bond", "ligand"].edge_attr = replacement[
            "ligand", "bond", "ligand"
        ].edge_attr

    return modified


class KinodataDockedWithDecoys(KinodataDocked):
    def __init__(
        self,
        root: str = str(_DATA),
        add_bond_info: bool = True,
        remove_hydrogen: bool = True,
        transform: Callable = None,
        pre_transform: Callable = None,
        pre_filter: Callable = None,
        percentage_valid: float = 0.5,
        seed: int = 0,
    ):
        super().__init__(
            root, add_bond_info, remove_hydrogen, transform, pre_transform, pre_filter
        )
        self.percentage_valid = percentage_valid
        self.rng = np.random.default_rng(seed)

    def get(self, index: int) -> Data:
        original = super().get(index)
        p = self.rng.uniform()
        if p > self.percentage_valid:
            random_index = self.rng.integers(0, len(self))
            replacement = super().get(random_index)
            original = replace_ligand(original, replacement)
            original.y = torch.zeros_like(original.y)

        return original
