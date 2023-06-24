from typing import Tuple
import torch
from torch import Tensor, cat
from torch.nn import ModuleList, Embedding, Sequential, Linear, Module, Parameter
from torch.nn.init import zeros_
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.aggr import (
    SumAggregation,
    MaxAggregation,
    MeanAggregation,
    MinAggregation,
    StdAggregation,
)
from torch_geometric.utils import coalesce
from torch_cluster import radius_graph

from ..types import NodeEmbedding, NodeType, RelationType
from .shared.dist_embedding import GaussianDistEmbedding
from .sparse_transformer import SPAB
from .regression import RegressionModel
from .resolve import resolve_act

aggr_cls = {
    "sum": SumAggregation,
    "min": MinAggregation,
    "max": MaxAggregation,
    "mean": MeanAggregation,
    "std": StdAggregation,
}


def make_aggrs(*aggrs: str):
    return ModuleList([aggr_cls[aggr]() for aggr in aggrs])


class MultiAggr(Module):
    def __init__(self, aggrs, hidden_channels, aggr_channels) -> None:
        super().__init__()
        self.aggrs = make_aggrs(*aggrs)
        self.lin_aggr = Linear(
            hidden_channels, aggr_channels * len(self.aggrs), bias=False
        )
        self.lin_out = Linear(aggr_channels * len(self.aggrs), hidden_channels)

    def forward(self, x, batch):
        x = self.lin_aggr(x).chunk(len(self.aggrs), dim=1)
        z = torch.cat([aggr(_x, batch) for aggr, _x in zip(self.aggrs, x)], dim=1)
        return self.lin_out(z)


class InteractionModule(Module):
    def __init__(self, interaction_radius: float) -> None:
        super().__init__()
        self.interaction_radius = interaction_radius

    def forward(self, data: HeteroData) -> EdgeStorage:
        pos = data[NodeType.Complex].pos
        batch = data[NodeType.Complex].batch
        edge_index = radius_graph(pos, self.interaction_radius, batch=batch)
        distances = (pos[edge_index[0]] - pos[edge_index[1]]).pow(2).sum(dim=1).sqrt()
        return EdgeStorage(edge_index=edge_index, edge_weight=distances, edge_attr=None)


class ComplexTransformer(RegressionModel):
    def __init__(
        self,
        config,
        hidden_channels: int,
        num_heads: int,
        num_attention_blocks: int,
        interaction_radius: float,
        act: str,
        max_atomic_number: int = 100,
        edge_attr_size: int = 4,
        ln1: bool = True,
        ln2: bool = False,
        ln3: bool = True,
        graph_norm: bool = True,
        precomputed_distance: bool = False,
    ) -> None:
        super().__init__(config)
        self.act = resolve_act(act)
        self.d_cut = interaction_radius
        self.interactions = InteractionModule(interaction_radius)
        self.atom_embedding = Embedding(max_atomic_number, hidden_channels)
        self.attention_blocks = ModuleList(
            [
                SPAB(hidden_channels, num_heads, self.act, ln1, ln2, ln3)
                for _ in range(num_attention_blocks)
            ]
        )
        if graph_norm:
            self.norm_layers = ModuleList(
                [GraphNorm(hidden_channels) for _ in range(num_attention_blocks)]
            )
        else:
            self.norm_layers = [lambda x, b: x] * num_attention_blocks
        self.dist_embedding = Sequential(
            GaussianDistEmbedding(hidden_channels, self.d_cut),
            Linear(hidden_channels, hidden_channels, bias=False),
        )
        self.covalent_embedding = Linear(edge_attr_size, hidden_channels, bias=False)
        self.edge_bias = Parameter(torch.empty(hidden_channels), requires_grad=True)
        zeros_(self.edge_bias)

        self.aggr = MultiAggr(
            ["min", "max", "mean"], hidden_channels, hidden_channels // 4
        )
        self.out = Sequential(
            Linear(hidden_channels, hidden_channels),
            self.act,
            Linear(hidden_channels, 1),
        )

    def forward(self, data: HeteroData) -> NodeEmbedding:
        node_store = data[NodeType.Complex]
        bond_store = data[NodeType.Complex, RelationType.Covalent, NodeType.Complex]
        intr_store = self.interactions(data)
        node_repr = self.atom_embedding(node_store.z)
        bond_repr = self.covalent_embedding(bond_store.edge_attr.float())
        dist_repr = self.dist_embedding(intr_store.edge_weight.view(-1, 1))

        edge_index, edge_repr = coalesce(
            cat((intr_store.edge_index, bond_store.edge_index), dim=1),
            cat((dist_repr, bond_repr), dim=0),
            reduce="add",
        )
        edge_repr = self.act(edge_repr + self.edge_bias)

        for sparse_attention_block, norm in zip(
            self.attention_blocks, self.norm_layers
        ):
            node_repr, edge_repr = sparse_attention_block(
                node_repr, edge_repr, edge_index
            )
            node_repr = norm(node_repr, node_store.batch)

        graph_repr = self.act(self.aggr(node_repr, node_store.batch))
        return self.out(graph_repr)
