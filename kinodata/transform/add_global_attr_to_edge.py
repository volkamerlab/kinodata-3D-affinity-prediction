from typing import List
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData

from kinodata.types import EdgeType


class AddGlobalAttrToEdge(BaseTransform):
    def __init__(
        self,
        edge_type: EdgeType,
        global_attr_key: List[str],
    ) -> None:
        self.edge_type = edge_type
        self.global_attr_key = global_attr_key

    def __call__(self, data: HeteroData) -> HeteroData:
        attr = torch.cat(
            tuple(getattr(data, key).view(1, -1) for key in self.global_attr_key), dim=1
        )
        edge_store = data[self.edge_type[0], self.edge_type[1], self.edge_type[2]]
        num_edges = edge_store.edge_index.size(1)
        edge_attr = attr.expand(num_edges, -1)
        if not hasattr(edge_store, "edge_attr"):
            edge_store.edge_attr = edge_attr
        else:
            edge_store.edge_attr = torch.cat((edge_store.edge_attr, edge_attr), dim=1)
        return data
