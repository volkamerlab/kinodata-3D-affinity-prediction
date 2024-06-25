import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph

from ..types import NodeType, RelationType


class ToLigandOnlyComplex(BaseTransform):
    node_attrs = ("x", "z", "pos", "is_pocket_atom")
    
    def __call__(self, data: HeteroData) -> HeteroData:
        node_store = data[NodeType.Complex]
        is_ligand_mask = ~node_store.is_pocket_atom.squeeze()
        for attr in self.node_attrs:
            node_store[attr] = node_store[attr][is_ligand_mask]
        edge_store = data[NodeType.Complex, RelationType.Covalent, NodeType.Complex]
        edge_index, edge_attr = subgraph(
            is_ligand_mask, edge_store.edge_index, edge_store.edge_attr, relabel_nodes=True
        )
        edge_store["edge_index"] = edge_index
        edge_store["edge_attr"] = edge_attr
        return data