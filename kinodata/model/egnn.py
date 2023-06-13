from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)
import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_scatter import scatter

from torch_geometric.nn.norm import GraphNorm

from collections import defaultdict

from kinodata.types import NodeType, NodeEmbedding, Kwargs, RelationType
from kinodata.types import EdgeType
from kinodata.model.resolve import resolve_act

RawMessage = Tuple[Tensor, Tensor, Tensor, Tensor]


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
        self.hidden_channels = hidden_channels

        self.message_repr_size = self.compute_message_repr_size(
            source_node_size,
            target_node_size,
            edge_attr_size,
            distance_size,
            hidden_channels,
        )
        self.init_parameters()

        if target_node_size != output_channels:
            self.residual_proj: nn.Module = nn.Linear(
                target_node_size, output_channels, bias=False
            )
        else:
            self.residual_proj = nn.Identity()
        self.fn_combine = nn.Sequential(
            nn.Linear(target_node_size + hidden_channels, hidden_channels),
            self.act,
            nn.Linear(hidden_channels, output_channels),
        )
        self.norm = GraphNorm(output_channels)

    def init_parameters(self):
        self._fn_message = nn.Sequential(
            nn.Linear(self.message_repr_size, self.hidden_channels),
            self.act,
            nn.Linear(self.hidden_channels, self.hidden_channels),
            self.act,
        )

    def fn_message(self, message: Tensor) -> Tensor:
        return self._fn_message(message)

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
        return self.final_act(
            self.norm(
                self.residual_proj(target_node)
                + self.fn_combine(torch.cat((target_node, aggr_messages), dim=1)),
                target_batch,
            )
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
        self.w_edge = nn.Sequential(
            nn.Linear(
                source_node_size + target_node_size + edge_attr_size,
                hidden_channels,
            ),
            self.act,
            nn.Linear(hidden_channels, hidden_channels),
        )

    def init_parameters(self):
        self.bias = nn.Parameter(torch.zeros(self.hidden_channels), requires_grad=True)

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
    ) -> RawMessage:
        return source_node, target_node, edge_attr, distance

    def fn_message(self, message: RawMessage) -> Tensor:
        source_node, target_node, edge_attr, distance = message
        dist_emb = self.w_dist(self.rbf(distance))
        if edge_attr is not None:
            edge_repr = torch.cat([source_node, target_node, edge_attr], dim=1)
        else:
            edge_repr = torch.cat([source_node, target_node], dim=1)
        edge_emb = self.w_edge(edge_repr)
        return self.act(dist_emb * edge_emb + self.bias)


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
        output_channels: int = None,
        edge_attr_size: Optional[Dict[Any, int]] = None,
        num_mp_layers: int = 2,
        mp_type: str = "rbf",
        node_types: List[NodeType] = [],
        edge_types: List[EdgeType] = [],
        act: str = "elu",
        message_layer_kwargs: Optional[Dict[EdgeType, Kwargs]] = None,
    ) -> None:
        """
        E(3)-invariant graph neural network for heterogeneous point clouds.

        Parameters
        ----------
        hidden_channels : int
            Number of hidden channels of node embeddings.
        output_channels : int, optional
            _description_, if None, will be set to be the same as hidden_channels.
        edge_attr_size : Optional[Dict[Any, int]], optional
            number of edge features, by default None if no edge features are used.
        num_mp_layers : int, optional
            number of message passing steps to perform, by default 2
        mp_type : str, optional
            the type of message passing layer, by default "rbf"
        node_types : List[NodeType], optional
            the types of nodes, by default []
        edge_types : List[EdgeType], optional
            the, by default []
        act : str, optional
            _description_, by default "elu"
        message_layer_kwargs : Optional[Dict[EdgeType, Kwargs]], optional
            _description_, by default None
        """
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
            message_layer_kwargs = defaultdict(dict)

        if output_channels is None:
            output_channels = hidden_channels

        self.reset_parameters(
            hidden_channels,
            num_mp_layers,
            output_channels,
            mp_class,
            edge_attr_size,
            act,
            message_layer_kwargs,
            edge_types,
        )

    def reset_parameters(
        self,
        hidden_channels: int,
        num_mp_layers: int,
        output_channels: int,
        mp_class: Type[EGNNMessageLayer],
        edge_attr_size: Mapping[EdgeType, int],
        act: Union[str, nn.Module],
        message_layer_kwargs: Dict[str, Any],
        edge_types: Iterable[EdgeType],
    ):
        # create stacks of MP layers
        self.message_passing_layers = nn.ModuleList()
        channels = [hidden_channels] * (num_mp_layers) + [output_channels]
        for d_in, d_out in zip(channels[:-1], channels[1:]):
            self.message_passing_layers.append(
                nn.ModuleDict(
                    {
                        "__".join(edge_type): mp_class(
                            source_node_size=d_in,
                            target_node_size=d_in,
                            edge_attr_size=edge_attr_size[edge_type],
                            distance_size=1,
                            hidden_channels=d_out,
                            output_channels=d_out,
                            act=act,
                            final_act=act,
                            **message_layer_kwargs[edge_type],
                        )
                        for edge_type in edge_types
                    }
                )
            )

    def forward(self, data: HeteroData, node_embedding: NodeEmbedding) -> NodeEmbedding:
        for mp_layer_dict in self.message_passing_layers:
            new_node_embed = {nt: x.clone() for nt, x in node_embedding.items()}
            for edge_type, layer in mp_layer_dict.items():
                edge_type = tuple(edge_type.split("__"))
                source_nt, _, target_nt = edge_type
                edge_weight = data[edge_type].edge_weight
                edge_attr = (
                    data[edge_type].edge_attr
                    if "edge_attr" in data[edge_type]
                    else None
                )
                source_node_embed = node_embedding[source_nt]
                target_node_embed = node_embedding[target_nt]

                new_node_embed[target_nt] += layer(
                    source_node_embed,
                    target_node_embed,
                    data[edge_type].edge_index,
                    edge_attr,
                    edge_weight,
                    data[target_nt].batch,
                )
            node_embedding = new_node_embed

        return node_embedding


class REGNN(EGNN):
    def assert_edge_attr_matches(
        self, edge_attr_size: Mapping[EdgeType, int]
    ) -> Mapping[RelationType, int]:
        size_sets = defaultdict(set)
        for (_, relation, _), size in edge_attr_size.items():
            size_sets[relation].add(size)
        assert all(len(size_set) == 1 for size_set in size_sets.values())
        return {relation: size_set.pop() for relation, size_set in size_sets.items()}

    def reset_parameters(
        self,
        hidden_channels: int,
        num_mp_layers: int,
        output_channels: int,
        mp_class: Type[EGNNMessageLayer],
        edge_attr_size: Mapping[EdgeType, int],
        act: Union[str, nn.Module],
        message_layer_kwargs: Dict[str, Any],
        edge_types: Iterable[EdgeType],
    ):
        self.edge_types = edge_types
        relation_edge_attr_size = self.assert_edge_attr_matches(edge_attr_size)
        relation_message_layer_kwargs = {
            relation: kwargs
            for (_, relation, _), kwargs in message_layer_kwargs.items()
        }
        self.message_passing_layers = nn.ModuleList()
        for i_layer in range(num_mp_layers):
            self.message_passing_layers.append(
                nn.ModuleDict(
                    {
                        relation: mp_class(
                            hidden_channels,
                            hidden_channels,
                            _edge_attr_size,
                            1,
                            hidden_channels,
                            hidden_channels,
                            act=act,
                            **relation_message_layer_kwargs[relation],
                        )
                        for relation, _edge_attr_size in relation_edge_attr_size.items()
                    }
                )
            )
        pass

    def forward(self, data: HeteroData, node_embedding: NodeEmbedding) -> NodeEmbedding:
        for layer_dict in self.message_passing_layers:
            new_node_embed = {nt: x.clone() for nt, x in node_embedding.items()}
            for source_node_type, relation, target_node_type in self.edge_types:
                edge_store = data[source_node_type, relation, target_node_type]
                edge_weight = edge_store.edge_weight
                edge_attribute = (
                    edge_store.edge_attr if "edge_attr" in edge_store else None
                )
                source_node_embedding = node_embedding[source_node_type]
                target_node_embedding = node_embedding[target_node_type]

                message_passing_layer = layer_dict[relation]
                new_node_embed[target_node_type] += message_passing_layer(
                    source_node_embedding,
                    target_node_embedding,
                    edge_store.edge_index,
                    edge_attribute,
                    edge_weight,
                    data[target_node_type].batch,
                )
            node_embedding = new_node_embed
        return node_embedding
