from typing import Dict, List, Optional, Set
import torch
from torch import full 
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import subgraph
from torch_geometric.data import HeteroData

from ..types import NodeType, RelationType


def mask_any_one_residue(
    data: HeteroData,
    open_list: Dict[int, Set[int]],
    residue_to_atom: Dict[int, Dict[int, List[int]]],
    edges_only: bool = True,
) -> Optional[HeteroData]:
    ident = int(data["ident"].item())
    open_residues = open_list[ident]
    if len(open_residues) == 0:
        data["masked_residue"] = torch.tensor([-1])
        return data
    residue_idx = open_residues.pop()
    residue_atoms = residue_to_atom[ident][residue_idx]
    
    node_store = data[NodeType.Complex]
    edge_store = data[NodeType.Complex, RelationType.Covalent, NodeType.Complex]
    x, z, pos = node_store.x, node_store.z, node_store.pos
    edge_index, edge_attr = edge_store.edge_index, edge_store.edge_attr
    
    num_nodes = x.size(0)
   
    if edges_only:
        residue_atoms = torch.tensor(residue_atoms)
        row, col = edge_index
        is_residue_pl_edge = torch.logical_or(torch.isin(row, residue_atoms), torch.isin(col, residue_atoms))
        edge_index = torch.stack((row[~is_residue_pl_edge], col[~is_residue_pl_edge]))
        edge_attr = edge_attr[~is_residue_pl_edge]
    else:
        mask = full((num_nodes,), 1, dtype=torch.bool)
        mask[residue_atoms] = False
        x = x[mask]
        z = z[mask]
        pos = pos[mask]
        edge_index, edge_attr = subgraph(mask, edge_index, edge_attr, relabel_nodes=True)
    
    data.masked_residue = torch.tensor([int(residue_idx)])
    data[NodeType.Complex].x = x
    data[NodeType.Complex].z = z
    data[NodeType.Complex].pos = pos
    data[NodeType.Complex, RelationType.Covalent, NodeType.Complex].edge_index = edge_index
    data[NodeType.Complex, RelationType.Covalent, NodeType.Complex].edge_attr = edge_attr
    return data

class MaskResidues(BaseTransform):
    open_list: Dict[int, Set[int]]
    residue_to_atom: Dict[int, Dict[int, List[int]]]
    
    def __init__(self, residue_to_atom: Dict[int, Dict[int, List[int]]], edges_only: bool) -> None:
        super().__init__()
        self.edges_only = edges_only
        self.residue_to_atom = residue_to_atom
        self.open_list = {
            ident: set(index.keys())
            for ident, index in residue_to_atom.items()
        }
    
    def __len__(self) -> int:
        return sum(len(val) for val in self.open_list.values())
    
    def filter(self, data: HeteroData) -> bool:
        ident = int(data["ident"].item())
        if ident not in self.open_list:
            return False
        return len(self.open_list[ident]) > 0
    
    def __call__(self, data: HeteroData) -> HeteroData:
        return mask_any_one_residue(
            data,
            self.open_list,
            self.residue_to_atom,
            edges_only=self.edges_only
        )
    