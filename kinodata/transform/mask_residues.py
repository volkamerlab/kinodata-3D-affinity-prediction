from typing import Dict, List, Literal, Optional, Set
import torch
from torch import full
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import subgraph
from torch_geometric.data import HeteroData

from ..types import NodeType, RelationType, MASK_RESIDUE_KEY

protein_letters_3to1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

protein_letters_1to3 = {v: k for k, v in protein_letters_3to1.items()}


def mask_any_one_residue(
    data: HeteroData,
    open_list: Dict[int, Set[int]],
    residue_to_atom: Dict[int, Dict[int, List[int]]],
    mask_bonds_only: bool = True,
    mask_pl_edges: bool = True,
    mask_pocket_atoms: bool = False,
) -> Optional[HeteroData]:
    if mask_pl_edges and mask_pocket_atoms:
        raise ValueError(
            "Only one of 'mask_pl_edges' and 'mask_pocket_atoms' can be True!"
        )

    ident = int(data["ident"].item())
    open_residues = open_list[ident]
    if len(open_residues) == 0:
        data["masked_residue"] = torch.tensor([-1])
        return data
    residue_idx = open_residues.pop()
    res_letter = data.pocket_sequence[int(residue_idx) - 1]
    res_name = (
        protein_letters_1to3[res_letter]
        if res_letter in protein_letters_1to3
        else "???"
    )
    residue_atoms = residue_to_atom[ident][residue_idx]

    node_store = data[NodeType.Complex]
    edge_store = data[NodeType.Complex, RelationType.Covalent, NodeType.Complex]
    x, z, pos = node_store.x, node_store.z, node_store.pos
    edge_index, edge_attr = edge_store.edge_index, edge_store.edge_attr

    num_nodes = x.size(0)

    if mask_bonds_only:
        residue_atoms = torch.tensor(residue_atoms)
        row, col = edge_index
        is_residue_pl_edge = torch.logical_or(
            torch.isin(row, residue_atoms), torch.isin(col, residue_atoms)
        )
        edge_index = torch.stack((row[~is_residue_pl_edge], col[~is_residue_pl_edge]))
        edge_attr = edge_attr[~is_residue_pl_edge]
    elif mask_pocket_atoms:
        mask = full((num_nodes,), 1, dtype=torch.bool)
        mask[residue_atoms] = False
        x = x[mask]
        z = z[mask]
        pos = pos[mask]
        edge_index, edge_attr = subgraph(
            mask, edge_index, edge_attr, relabel_nodes=True
        )
    elif mask_pl_edges:
        is_part_of_masked_residue = full((num_nodes,), 0, dtype=torch.bool)
        is_part_of_masked_residue[residue_atoms] = True
        data[NodeType.Complex][MASK_RESIDUE_KEY] = is_part_of_masked_residue

    data.masked_residue = torch.tensor([int(residue_idx)])
    data.masked_resname = res_name
    data.masked_res_letter = res_letter
    data[NodeType.Complex].x = x
    data[NodeType.Complex].z = z
    data[NodeType.Complex].pos = pos
    data[NodeType.Complex, RelationType.Covalent, NodeType.Complex].edge_index = (
        edge_index
    )
    data[NodeType.Complex, RelationType.Covalent, NodeType.Complex].edge_attr = (
        edge_attr
    )
    return data


class MaskResidues(BaseTransform):
    open_list: Dict[int, Set[int]]
    residue_to_atom: Dict[int, Dict[int, List[int]]]

    def __init__(
        self,
        residue_to_atom: Dict[int, Dict[int, List[int]]],
        mask_type: Literal["atom_objects", "pl_interactions", "bonds"] = "atom_objects",
    ) -> None:
        super().__init__()
        assert mask_type in ["atom_objects", "pl_interactions", "bonds"]
        self.mask_type = mask_type
        self.residue_to_atom = residue_to_atom
        self.open_list = {
            ident: set(index.keys()) for ident, index in residue_to_atom.items()
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
            mask_bonds_only=self.mask_type == "bonds",
            mask_pocket_atoms=self.mask_type == "atom_objects",
            mask_pl_edges=self.mask_type == "pl_interactions",
        )
