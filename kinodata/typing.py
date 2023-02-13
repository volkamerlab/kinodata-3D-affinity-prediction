from typing import Dict, Tuple
from torch import Tensor

NodeType = str
NodeEmbedding = Dict[NodeType, Tensor]
EdgeType = Tuple[NodeType, str, NodeType]