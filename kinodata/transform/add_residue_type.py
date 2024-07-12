import torch
from torch.nn.functional import one_hot
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData

from ..types import NodeType, RelationType

NUM_RESIDUES = 23

class AddResidueType(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data: HeteroData) -> HeteroData:
        ligand_store = data[NodeType.Ligand]
        pocket_store = data[NodeType.Pocket]

        lig_residue = data[NodeType.Ligand].residue.to(torch.int64)
        assert torch.max(lig_residue) < NUM_RESIDUES, lig_residue
        lig_residue = one_hot(lig_residue, num_classes=NUM_RESIDUES)
        data[NodeType.Ligand].x = torch.cat(
            [data[NodeType.Ligand].x, lig_residue], dim=1
        )
        pocket_residue = data[NodeType.Pocket].residue
        assert torch.max(pocket_residue) < NUM_RESIDUES, pocket_residue
        pocket_residue = one_hot(data[NodeType.Pocket].residue, num_classes=NUM_RESIDUES)
        data[NodeType.Pocket].x = torch.cat(
            [data[NodeType.Pocket].x, pocket_residue], dim=1
        )

        return data
