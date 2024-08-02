import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData

from ..types import NodeType, RelationType


class TransformToComplexGraph(BaseTransform):
    def __init__(
        self,
        remove_heterogeneous_representation: bool = False,
        ligand_ty: NodeType = NodeType.Ligand,
        pocket_ty: NodeType = NodeType.Pocket,
        complex_ty: NodeType = NodeType.Complex,
    ):
        self.remove_heterogeneous_representation = remove_heterogeneous_representation
        self.ligand_ty = ligand_ty
        self.pocket_ty = pocket_ty
        self.complex_ty = complex_ty

    def __call__(
        self,
        data: HeteroData,
    ) -> HeteroData:
        ligand_store = data[self.ligand_ty]
        pocket_store = data[self.pocket_ty]
        ligand_edge_store = data[self.ligand_ty, RelationType.Covalent, self.ligand_ty]
        pocket_edge_store = data[self.pocket_ty, RelationType.Covalent, self.pocket_ty]

        x = torch.cat((pocket_store.x, ligand_store.x), dim=0)
        z = torch.cat((pocket_store.z, ligand_store.z), dim=0)
        pos = torch.cat((pocket_store.pos, ligand_store.pos), dim=0)
        edge_index = torch.cat(
            (
                pocket_edge_store.edge_index,
                ligand_edge_store.edge_index + pocket_store.x.size(0),
            ),
            dim=1,
        )
        edge_attr = torch.cat(
            (
                pocket_edge_store.edge_attr,
                ligand_edge_store.edge_attr,
            ),
            dim=0,
        )

        data[self.complex_ty].x = x
        data[self.complex_ty].z = z
        data[self.complex_ty].pos = pos
        data[self.complex_ty, RelationType.Covalent, self.complex_ty].edge_index = (
            edge_index
        )
        data[self.complex_ty, RelationType.Covalent, self.complex_ty].edge_attr = (
            edge_attr
        )

        if self.remove_heterogeneous_representation:
            del data[self.ligand_ty]
            del data[self.pocket_ty]
            del data[self.ligand_ty, RelationType.Covalent, self.ligand_ty]
            del data[self.pocket_ty, RelationType.Covalent, self.pocket_ty]

        return data
