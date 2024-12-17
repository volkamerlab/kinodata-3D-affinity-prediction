import torch
from torch_geometric.utils import subgraph
from ..types import NodeType, RelationType
from torch_geometric.transforms import BaseTransform


class MaskComplexComponent(BaseTransform):

    def component(self, data) -> torch.Tensor: ...

    def forward(self, data):
        subset = self.component(data)
        bonds = data[
            NodeType.Complex, RelationType.Covalent, NodeType.Complex
        ].edge_index
        bond_features = data[
            NodeType.Complex, RelationType.Covalent, NodeType.Complex
        ].edge_attr
        bonds, bond_features = subgraph(
            subset,
            edge_index=bonds,
            edge_attr=bond_features,
            relabel_nodes=True,
        )
        data[NodeType.Complex].x = data[NodeType.Complex].x[subset]
        data[NodeType.Complex].pos = data[NodeType.Complex].pos[subset]
        data[NodeType.Complex].z = data[NodeType.Complex].z[subset]
        data[NodeType.Complex, RelationType.Covalent, NodeType.Complex].edge_index = (
            bonds
        )
        data[NodeType.Complex, RelationType.Covalent, NodeType.Complex].edge_attr = (
            bond_features
        )
        return data


class MaskLigand(MaskComplexComponent):

    def component(self, data) -> torch.Tensor:
        return ~data[NodeType.Complex].is_pocket_atom


class MaskPocket(MaskComplexComponent):

    def component(self, data) -> torch.Tensor:
        return data[NodeType.Complex].is_pocket_atom


class MaskLigandPosition(BaseTransform):
    offset: float = 1e4

    def forward(self, data):
        is_ligand = ~data[NodeType.Complex].is_pocket_atom
        translation = torch.zeros_like(data[NodeType.Complex].pos)
        translation[is_ligand] = self.offset
        data[NodeType.Complex].pos = data[NodeType.Complex].pos + translation
        return data
