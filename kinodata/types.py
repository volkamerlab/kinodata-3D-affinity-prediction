from typing import Any, Dict, Tuple
from torch import Tensor
from torch.utils.data import Subset

NodeType = str
NodeEmbedding = Dict[NodeType, Tensor]
EdgeType = Tuple[NodeType, str, NodeType]
DataSplit = Dict[str, Subset]
