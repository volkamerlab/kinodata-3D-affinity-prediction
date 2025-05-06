from enum import StrEnum
from typing import Any, Dict, List, Tuple
from torch import Tensor
from torch.utils.data import Subset


class NodeType:
    Ligand = "ligand"
    Pocket = "pocket"
    Complex = "complex"
    PocketResidue = "pocket_residue"


class RelationType:
    Covalent = "bond"
    Interacts = "interacts"  # between different molecules
    Intraacts = "intraacts"  # within a molecule, superset of covalent interactions
    IsPartOf = "is_part_of"  # between a residue and a molecule


Kwargs = Dict[str, Any]
NodeEmbedding = Dict[NodeType, Tensor]
EdgeType = Tuple[NodeType, RelationType, NodeType]
DataSplit = Dict[str, Subset]

INTERMOL_STRUCTURAL_EDGE_TYPES: List[EdgeType] = [
    (NodeType.Ligand, RelationType.Interacts, NodeType.Pocket),
    (NodeType.Pocket, RelationType.Interacts, NodeType.Ligand),
]

INTRAMOL_STRUCTURAL_EDGE_TYPES: List[EdgeType] = [
    (NodeType.Ligand, RelationType.Intraacts, NodeType.Ligand),
    (NodeType.Pocket, RelationType.Intraacts, NodeType.Pocket),
]

COVALENT_EDGE_TYPES: List[EdgeType] = [
    (NodeType.Ligand, RelationType.Covalent, NodeType.Ligand),
    (NodeType.Pocket, RelationType.Covalent, NodeType.Pocket),
]


MASK_RESIDUE_KEY = "mask_residue_interactions"


class COLS(StrEnum):
    ACTIVITY_ID = "activities.activity_id"
    KLIFS_ID = "similar.klifs_structure_id"
    SEQUENCE = "structure.pocket_sequence"
    DUNBRACK = "abreviated_dunbrack_state"
    DFG = "dfg_state"
    DUNBRACK_CONF = "dunbrack_conf"
    DUNBRACK_ACTIVE = "dunbrack_active"
    DUNBRACK_SIMPLIFIED = "dunbrack_simplified"
    REFERENCE_PREDICTION = "reference_pred"
    MASKED_PREDICTION = "masked_pred"
    DELTA = "delta"
    RESIDUE_IMPORTANCE = "residue_importance"
    UNIPROT_ID = "UniprotID"
