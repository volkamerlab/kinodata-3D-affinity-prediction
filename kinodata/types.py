from typing import Any, Dict, List, Tuple
from torch import Tensor
from torch.utils.data import Subset


class NodeType:
    Ligand = "ligand"
    Pocket = "pocket"
    PocketResidue = "pocket_residue"


class RelationType:
    Bond = "bond"
    Interacts = "interacts"


Kwargs = Dict[str, Any]
NodeEmbedding = Dict[NodeType, Tensor]
EdgeType = Tuple[NodeType, RelationType, NodeType]
DataSplit = Dict[str, Subset]

STRUCTURAL_EDGE_TYPES: List[EdgeType] = [
    (NodeType.Ligand, RelationType.Interacts, NodeType.Pocket),
    (NodeType.Ligand, RelationType.Interacts, NodeType.Pocket),
]

COVALENT_EDGE_TYPES: List[EdgeType] = [
    (NodeType.Ligand, RelationType.Bond, NodeType.Ligand),
    (NodeType.Pocket, RelationType.Bond, NodeType.Pocket),
]
