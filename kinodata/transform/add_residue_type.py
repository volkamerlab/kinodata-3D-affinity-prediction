import torch
from torch.nn.functional import one_hot
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData

from ..types import NodeType, RelationType


class AddResidueType(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data: HeteroData) -> HeteroData:
        ligand_store = data[NodeType.Ligand]
        pocket_store = data[NodeType.Pocket]

        lig_residue = one_hot(data[NodeType.Ligand].residue, num_classes=21)
        data[NodeType.Ligand].x = torch.cat(
            [data[NodeType.Ligand].x, lig_residues], dim=1
        )
        pocket_residue = one_hot(data[NodeType.Pocket].residue, num_classes=21)
        data[NodeType.Pocket].x = torch.cat(
            [data[NodeType.Pocket].x, pocket_residue], dim=1
        )

        return data
