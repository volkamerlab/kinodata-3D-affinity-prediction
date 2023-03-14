from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MeanAggregation
from torch_scatter import scatter

from torch_geometric.nn.norm import GraphNorm

from collections import defaultdict

from kinodata.types import NodeType, NodeEmbedding, EdgeType
from kinodata.model.resolve import resolve_act


class EGNNMessageLayer(nn.Module):
    def __init__(
        self,
        source_node_size: int,
        target_node_size: int,
        edge_attr_size: int,
        distance_size: int,
        hidden_channels: int,
        output_channels: int,
        act: str,
        final_act: str = "none",
        reduce: str = "sum",
        **kwargs,
    ) -> None:
        super().__init__()

        self.act = resolve_act(act)
        self.final_act = resolve_act(final_act)
        self.reduce = reduce

        message_repr_size = self.compute_message_repr_size(
            source_node_size,
            target_node_size,
            edge_attr_size,
            distance_size,
            hidden_channels,
        )
        self.fn_message = nn.Sequential(
            nn.Linear(message_repr_size, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, hidden_channels),
            self.act,
        )
        self.residual_proj = nn.Linear(target_node_size, output_channels, bias=False)
        self.fn_combine = nn.Linear(target_node_size + hidden_channels, output_channels)
        self.norm = GraphNorm(output_channels)

    def compute_message_repr_size(
        self,
        source_node_size: int,
        target_node_size: int,
        edge_attr_size: int,
        distance_size: int,
        hidden_channels: int,
    ) -> int:
        return source_node_size + target_node_size + edge_attr_size + distance_size

    def create_message_repr(
        self,
        source_node: Tensor,
        target_node: Tensor,
        edge_attr: Tensor,  # unused
        distance: Tensor,
    ) -> Tensor:
        return torch.cat((source_node, target_node, distance), dim=1)

    def forward(
        self,
        source_node: Tensor,
        target_node: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        distance: Tensor,
        target_batch: LongTensor,
    ):
        i_source, i_target = edge_index
        messages = self.fn_message(
            self.create_message_repr(
                source_node[i_source], target_node[i_target], edge_attr, distance
            )
        )
        aggr_messages = torch.zeros_like(target_node)
        scatter(messages, i_target, 0, reduce=self.reduce, out=aggr_messages)
        return self.norm(
            self.final_act(
                self.residual_proj(target_node)
                + self.fn_combine(torch.cat((target_node, aggr_messages), dim=1))
            ),
            target_batch,
        )


class ExpnormRBFEmbedding(nn.Module):
    def __init__(self, size: int, d_cut: float, trainable: bool = False) -> None:
        super().__init__()
        self.size = size
        self.d_cut = nn.parameter.Parameter(
            torch.tensor([d_cut], dtype=torch.float32), requires_grad=False
        )

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    # taken from torchmd-net
    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(-self.d_cut.squeeze())
        means = torch.linspace(start_value, 1, self.size)
        betas = torch.tensor([(2 / self.size * (1 - start_value)) ** -2] * self.size)
        return means, betas

    def cosine_cutoff(self, d: Tensor) -> Tensor:
        d = d.clamp_max(self.d_cut)
        return 0.5 * (torch.cos(d * torch.pi / self.d_cut) + 1)

    def forward(self, d: Tensor) -> Tensor:
        cutoff = self.cosine_cutoff(d)
        rbf = torch.exp(-self.betas * (torch.exp(-d) - self.means).pow(2))
        return cutoff * rbf


class RBFLayer(EGNNMessageLayer):
    def __init__(
        self,
        source_node_size: int,
        target_node_size: int,
        edge_attr_size: int,
        distance_size: int,
        hidden_channels: int,
        output_channels: int,
        act: str,
        final_act: str = "none",
        reduce: str = "sum",
        interaction_radius: float = 5.0,
        rbf_size: int = None,
        **kwargs,
    ) -> None:
        super().__init__(
            source_node_size,
            target_node_size,
            edge_attr_size,
            distance_size,
            hidden_channels,
            output_channels,
            act,
            final_act,
            reduce,
        )
        if rbf_size is None:
            rbf_size = hidden_channels
        self.rbf = ExpnormRBFEmbedding(rbf_size, interaction_radius)
        self.w_dist = nn.Linear(rbf_size, hidden_channels, bias=False)
        self.w_edge = nn.Linear(
            source_node_size + target_node_size + edge_attr_size,
            hidden_channels,
            bias=False,
        )

    def compute_message_repr_size(
        self,
        source_node_size: int,
        target_node_size: int,
        edge_attr_size: int,
        distance_size: int,
        hidden_channels: int,
    ) -> int:
        return hidden_channels

    def create_message_repr(
        self,
        source_node: Tensor,
        target_node: Tensor,
        edge_attr: Tensor,
        distance: Tensor,
    ) -> Tensor:
        dist_emb = self.w_dist(self.rbf(distance)).tanh()
        if edge_attr is not None:
            edge_repr = torch.cat([source_node, target_node, edge_attr], dim=1)
        else:
            edge_repr = torch.cat([source_node, target_node], dim=1)
        edge_emb = self.w_edge(edge_repr)
        return dist_emb * edge_emb


def resolve_mp_type(mp_type: str) -> type[EGNNMessageLayer]:
    if mp_type == "egnn":
        return EGNNMessageLayer
    if mp_type == "rbf":
        return RBFLayer
    raise ValueError(f"Unknown message passing type: {mp_type}", mp_type)


class EGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        final_embedding_size: int = None,
        edge_attr_size: Optional[Dict[Any, int]] = None,
        num_mp_layers: int = 2,
        mp_type: str = "rbf",
        node_types: List[NodeType] = [],
        edge_types: List[EdgeType] = [],
        act: str = "elu",
        message_layer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types

        mp_class = resolve_mp_type(mp_type)

        if edge_attr_size is None:
            edge_attr_size = dict()
        _edge_attr_size = defaultdict(int)
        for key, value in edge_attr_size.items():
            _edge_attr_size[key] = value
        edge_attr_size = _edge_attr_size

        if message_layer_kwargs is None:
            message_layer_kwargs = dict()

        if final_embedding_size is None:
            final_embedding_size = hidden_channels

        # equation (1) "psi_0"
        self.f_initial_embed = nn.ModuleDict(
            {nt: nn.Embedding(100, hidden_channels) for nt in node_types}
        )

        # create stacks of MP layers
        self.message_passing_layers = nn.ModuleList()
        channels = [hidden_channels] * (num_mp_layers) + [final_embedding_size]
        for d_in, d_out in zip(channels[:-1], channels[1:]):
            self.message_passing_layers.append(
                nn.ModuleDict(
                    {
                        "_".join(edge_type): mp_class(
                            source_node_size=d_in,
                            target_node_size=d_in,
                            edge_attr_size=edge_attr_size[edge_type],
                            distance_size=1,
                            hidden_channels=d_out,
                            output_channels=d_out,
                            act=act,
                            final_act=act,
                            **message_layer_kwargs,
                        )
                        for edge_type in edge_types
                        if edge_type[1] == "interacts"
                    }
                )
            )

    def encode(self, data: HeteroData) -> NodeEmbedding:
        node_embed = dict()
        for node_type in self.node_types:
            node_embed[node_type] = self.f_initial_embed[node_type](data[node_type].z)

        for mp_layer_dict in self.message_passing_layers:
            new_node_embed = {nt: x.clone() for nt, x in node_embed.items()}
            for edge_type, layer in mp_layer_dict.items():
                edge_type = tuple(edge_type.split("_"))
                source_nt, _, target_nt = edge_type
                edge_weight = data[edge_type].edge_weight
                edge_attr = (
                    data[edge_type].edge_attr
                    if "edge_attr" in data[edge_type]
                    else None
                )
                source_node_embed = node_embed[source_nt]
                target_node_embed = node_embed[target_nt]

                new_node_embed[target_nt] += layer(
                    source_node_embed,
                    target_node_embed,
                    data[edge_type].edge_index,
                    edge_attr,
                    edge_weight,
                    data[target_nt].batch,
                )
            node_embed = new_node_embed

        return node_embed

    def forward(self, data: HeteroData) -> NodeEmbedding:
        return self.encode(data)
