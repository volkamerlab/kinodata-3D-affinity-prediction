from typing import List, Optional
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_scatter import scatter_add

from kinodata.types import NodeType, EdgeType


def compute_isolated_nodes(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
    how: int = 0,
) -> Tensor:
    assert how in (0, 1)
    if num_nodes is None:
        num_nodes = edge_index[how].max().item() + 1
    num_neighbors: Tensor = torch.zeros(num_nodes)
    index = edge_index[how]
    scatter_add(
        torch.ones_like(index, dtype=num_neighbors.dtype), index, out=num_neighbors
    )
    return num_neighbors == 0


class RemoveIsolatedNodes:
    node_type: NodeType
    edge_type: EdgeType
    node_level_keys: List[str] = ["x", "z"]

    @property
    def index_dim(self) -> int:
        return 1 - int(self.node_type == self.edge_type[0])

    def __init__(self, node_type: NodeType, edge_type: EdgeType) -> None:
        source_type, _, target_type = edge_type
        if node_type not in (source_type, edge_type):
            raise ValueError(
                f"Incompatible node and edge type {node_type}, {edge_type}."
            )
        self.node_type = node_type
        self.edge_type = edge_type

    def __call__(self, data: HeteroData) -> HeteroData:
        num_nodes = data[self.node_type].num_nodes
        edge_index: Tensor = data[
            self.edge_type[0], self.edge_type[1], self.edge_type[2]
        ].edge_index
        isolated_mask = compute_isolated_nodes(
            edge_index,
            num_nodes=num_nodes,
            how=self.index_dim,
        )
        num_isolated = isolated_mask.sum()
        if num_isolated == 0:
            return data

        for name in self.node_level_keys:
            storage = getattr(data[self.node_type], name)
            setattr(data[self.node_type], name, storage[~isolated_mask])
        relabeling = torch.empty(num_nodes, dtype=torch.long)
        relabeling[~isolated_mask] = torch.arange(num_nodes - num_isolated)
        edge_index[self.index_dim] = relabeling[edge_index[self.index_dim]]
        data[
            self.edge_type[0], self.edge_type[1], self.edge_type[2]
        ].edge_index = edge_index

        return data
