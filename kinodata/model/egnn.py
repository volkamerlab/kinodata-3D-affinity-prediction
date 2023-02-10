from typing import Any, Callable, Dict, List, Tuple
import torch
from torch import Tensor, LongTensor
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MeanAggregation
from torch_geometric.loader import DataLoader
from torch_scatter import scatter

from torch_geometric.nn.norm import GraphNorm

from collections import defaultdict

NodeType = str
EdgeType = Tuple[NodeType, str, NodeType]


class ScaledSigmoid(nn.Module):
    def __init__(self, alpha: float, beta: float, requires_grad=False) -> None:
        super().__init__()
        self.alpha = nn.parameter.Parameter(
            torch.ones(1) * alpha, requires_grad=requires_grad
        )
        self.beta = nn.parameter.Parameter(
            torch.ones(1) * beta, requires_grad=requires_grad
        )

    def forward(self, x):
        return torch.sigmoid(x) * (self.beta - self.alpha) + self.alpha


_act = {
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "silu": nn.SiLU(),
    "none": nn.Identity(),
}


def resolve_act(act: str) -> nn.Module:
    try:
        return _act[act]
    except KeyError:
        raise ValueError(act)


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
    def __init__(self, size: int, d_cut: float, trainable: bool = True) -> None:
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


def resolve_mp_type(mp_type: str) -> EGNNMessageLayer:
    if mp_type == "egnn":
        return EGNNMessageLayer
    if mp_type == "rbf":
        return RBFLayer


class EGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        final_embedding_size: int = None,
        edge_attr_size: Dict[Any, int] = None,
        target_size: int = 1,
        num_mp_layers: int = 2,
        mp_type: str = "rbf",
        node_types: List[NodeType] = [],
        edge_types: List[EdgeType] = [],
        act: str = "elu",
        message_layer_kwargs: Dict[str, Any] = None,
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types

        mp_class = resolve_mp_type(mp_type)

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

        # modules required for readout of a graph-level
        # representation and graph-level property prediction
        self.aggregation = MeanAggregation()
        self.f_predict = nn.Sequential(
            nn.LayerNorm(final_embedding_size),
            nn.Linear(final_embedding_size, final_embedding_size),
            resolve_act(act),
            nn.Linear(final_embedding_size, target_size),
            nn.Softplus(),
        )

    def encode(self, data: HeteroData) -> Dict[NodeType, Tensor]:
        node_embed = dict()
        for node_type in self.node_types:
            node_embed[node_type] = self.f_initial_embed[node_type](data[node_type].z)

        for mp_layer_dict in self.message_passing_layers:
            new_node_embed = {nt: x.clone() for nt, x in node_embed.items()}
            for edge_type, layer in mp_layer_dict.items():
                edge_type = tuple(edge_type.split("_"))
                source_nt, _, target_nt = edge_type
                dist = data[edge_type].dist
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
                    dist,
                    data[target_nt].batch,
                )
            node_embed = new_node_embed

        return node_embed

    def _predict(self, node_embed: Dict[NodeType, Tensor], data: HeteroData) -> Tensor:
        aggr = {nt: self.aggregation(x, data[nt].batch) for nt, x in node_embed.items()}
        aggr = sum(aggr.values())
        return self.f_predict(aggr)

    def forward(self, data: HeteroData) -> Tensor:
        node_embed = self.encode(data)
        pred = self._predict(node_embed, data)
        return pred


if __name__ == "__main__":
    from kinodata.data.dataset import KinodataDocked
    from kinodata.transform import AddDistancesAndInteractions

    dataset = KinodataDocked(transform=AddDistancesAndInteractions(radius=2.0))[:128]
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    node_types, edge_types = dataset[0].metadata()
    node_types, edge_types
    model = EGNN(0, 64, 64, 1, node_types=node_types, edge_types=edge_types)
    print("Num parameters", sum(p.numel() for p in model.parameters()))

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf")
    for epoch in range(100):
        total_loss = 0
        for data in loader:
            y = model(data).flatten()
            loss = (y - data.y).abs().mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * data.y.size(0)

        best = min([best, total_loss / 128])
        print(
            f"{epoch+1}/{100} | MAE {total_loss / 128}, BEST {best}",
            end="\r",
            flush=False,
        )
    print(best)
