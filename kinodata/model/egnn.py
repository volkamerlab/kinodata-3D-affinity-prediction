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

from collections import defaultdict

from kinodata.types import NodeType, NodeEmbedding, Kwargs, RelationType
from kinodata.types import EdgeType
from kinodata.model.resolve import resolve_act
from kinodata.model.shared.dist_embedding import GaussianDistEmbedding
import numpy as np

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
        self.lin_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            self.final_act,
            nn.LayerNorm(hidden_channels),
        )

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
        return self.lin_out(aggr_messages)

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

        self.rbf = GaussianDistEmbedding(rbf_size, interaction_radius)
        self.w_dist = nn.Linear(rbf_size, hidden_channels * 2, bias=False)
        self.w_edge = nn.Sequential(
            nn.Linear(
                source_node_size + target_node_size + edge_attr_size,
                hidden_channels,
            ),
            self.act,
            nn.Linear(hidden_channels, hidden_channels, bias=False),
        )

    def init_parameters(self):
        return

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
        dist_mul, dist_add = torch.chunk(self.w_dist(self.rbf(distance)), 2, -1)
        if edge_attr is not None:
            edge_repr = torch.cat([source_node, target_node, edge_attr], dim=1)
        else:
            edge_repr = torch.cat([source_node, target_node], dim=1)
        edge_emb = self.w_edge(edge_repr)
        return self.act((1 + dist_mul) * edge_emb + dist_add)


def resolve_mp_type(mp_type: str) -> type[EGNNMessageLayer]:
    if mp_type == "egnn":
        return EGNNMessageLayer
    if mp_type == "rbf":
        return RBFLayer
    raise ValueError(f"Unknown message passing type: {mp_type}", mp_type)


class NodeUpdate(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        node_types: List[NodeType],
        act: str,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.lins = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(hidden_channels * 2, hidden_channels),
                    resolve_act(act),
                    nn.Linear(hidden_channels, hidden_channels),
                    resolve_act(act),
                    nn.Dropout(dropout),
                )
                for node_type in self.node_types
            }
        )
        self.lns = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_channels) for node_type in self.node_types}
        )

    def forward(
        self, node_embedding: NodeEmbedding, aggregated_messages: Dict[NodeType, Tensor]
    ) -> NodeEmbedding:
        out = dict()
        for node_type in node_embedding:
            residual = node_embedding[node_type]
            incoming_message = aggregated_messages[node_type]
            update = self.lins[node_type](torch.cat((residual, incoming_message), -1))
            out[node_type] = self.lns[node_type](residual + update)
        return out


class HeteroEGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        edge_attr_size: Optional[Dict[Any, int]] = None,
        num_mp_layers: int = 2,
        mp_type: str = "rbf",
        node_types: List[NodeType] = [],
        edge_types: List[EdgeType] = [],
        act: str = "elu",
        message_layer_kwargs: Optional[Dict[EdgeType, Kwargs]] = None,
        dropout: float = 0.0,
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

        self.reset_message_passing_layers(
            hidden_channels,
            num_mp_layers,
            mp_class,
            edge_attr_size,
            act,
            message_layer_kwargs,
            edge_types,
        )
        self.reset_update_layers(
            hidden_channels,
            num_mp_layers,
            act,
            node_types,
            dropout=dropout,
        )

    def reset_update_layers(
        self,
        hidden_channels: int,
        num_mp_layers: int,
        act: str,
        node_types: List[NodeType],
        dropout: float = 0.0,
    ):
        self.update_layers = nn.ModuleList()
        for _ in range(num_mp_layers):
            self.update_layers.append(
                NodeUpdate(hidden_channels, node_types, act, dropout)
            )

    def reset_message_passing_layers(
        self,
        hidden_channels: int,
        num_mp_layers: int,
        mp_class: Type[EGNNMessageLayer],
        edge_attr_size: Mapping[EdgeType, int],
        act: Union[str, nn.Module],
        message_layer_kwargs: Dict[str, Any],
        edge_types: List[EdgeType],
    ):
        # create stacks of MP layers
        self.message_passing_layers = nn.ModuleList()
        channels = [hidden_channels] * (num_mp_layers + 1)
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

    def compute_aggregated_messages(
        self,
        data: HeteroData,
        node_embedding: NodeEmbedding,
        message_passing: Dict[Any, nn.Module],
    ) -> NodeEmbedding:
        aggregated_messages = {
            nt: x.new_zeros(x.size()) for nt, x in node_embedding.items()
        }
        for edge_type, layer in message_passing.items():
            edge_type = tuple(edge_type.split("__"))
            source_node_type, _, target_node_type = edge_type
            edge_weight = data[edge_type].edge_weight
            edge_attribute = (
                data[edge_type].edge_attr if "edge_attr" in data[edge_type] else None
            )
            source_node_embed = node_embedding[source_node_type]
            target_node_embed = node_embedding[target_node_type]

            aggregated_messages[target_node_type] += layer(
                source_node_embed,
                target_node_embed,
                data[edge_type].edge_index,
                edge_attribute,
                edge_weight,
                data[target_node_type].batch,
            )
        return aggregated_messages

    def forward(self, data: HeteroData, node_embedding: NodeEmbedding) -> NodeEmbedding:
        for mp_layer_dict, node_update_module in zip(
            self.message_passing_layers, self.update_layers
        ):
            aggregated_messages = self.compute_aggregated_messages(
                data, node_embedding, mp_layer_dict
            )
            node_embedding = node_update_module(node_embedding, aggregated_messages)

        return node_embedding


class HeteroEGNNRel(HeteroEGNN):
    def assert_edge_attr_matches(
        self, edge_attr_size: Mapping[EdgeType, int]
    ) -> Mapping[RelationType, int]:
        size_sets = defaultdict(set)
        for (_, relation, _), size in edge_attr_size.items():
            size_sets[relation].add(size)
        assert all(len(size_set) == 1 for size_set in size_sets.values())
        return {relation: size_set.pop() for relation, size_set in size_sets.items()}

    def reset_message_passing_layers(
        self,
        hidden_channels: int,
        num_mp_layers: int,
        mp_class: Type[EGNNMessageLayer],
        edge_attr_size: Mapping[EdgeType, int],
        act: Union[str, nn.Module],
        message_layer_kwargs: Dict[str, Any],
        edge_types: Iterable[EdgeType],
    ):
        self.edge_types = edge_types
        relations = set([relation for _, relation, _ in self.edge_types])
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
                            relation_edge_attr_size[relation],
                            1,
                            hidden_channels,
                            hidden_channels,
                            act=act,
                            **relation_message_layer_kwargs[relation],
                        )
                        for relation in relations
                    }
                )
            )
        pass

    def compute_aggregated_messages(
        self,
        data: HeteroData,
        node_embedding: NodeEmbedding,
        message_passing: Dict[Any, nn.Module],
    ) -> NodeEmbedding:
        aggregated_messages = {
            nt: x.new_zeros(x.size()) for nt, x in node_embedding.items()
        }
        for source_node_type, relation, target_node_type in self.edge_types:
            edge_store = data[source_node_type, relation, target_node_type]
            edge_weight = edge_store.edge_weight
            edge_attribute = edge_store.edge_attr if "edge_attr" in edge_store else None
            source_node_embedding = node_embedding[source_node_type]
            target_node_embedding = node_embedding[target_node_type]
            message_passing_layer = message_passing[relation]
            aggregated_messages[target_node_type] += message_passing_layer(
                source_node_embedding,
                target_node_embedding,
                edge_store.edge_index,
                edge_attribute,
                edge_weight,
                data[target_node_type].batch,
            )
        return aggregated_messages
