from typing import Callable, Dict, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MeanAggregation
from torch_geometric.loader import DataLoader
from torch_scatter import scatter

NodeType = str
EdgeType = Tuple[NodeType, str, NodeType]

_act = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
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
        reduce: str = "mean",
    ) -> None:
        super().__init__()

        act = resolve_act(act)
        self.final_act = resolve_act(final_act)
        self.reduce = reduce

        self.input_size = self.combine_message_inputs(
            torch.empty(1, source_node_size),
            torch.empty(1, target_node_size),
            torch.empty(1, edge_attr_size),
            torch.empty(1, distance_size),
        ).size(1)

        self.fn_message = nn.Sequential(
            nn.Linear(self.input_size, hidden_channels), act
        )
        self.residual_proj = nn.Linear(target_node_size, output_channels, bias=False)
        self.fn_combine = nn.Linear(target_node_size + hidden_channels, output_channels)
        self.ln = nn.LayerNorm(output_channels)

    def combine_message_inputs(
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
    ):
        i_source, i_target = edge_index
        messages = self.fn_message(
            self.combine_message_inputs(
                source_node[i_source], target_node[i_target], edge_attr, distance
            )
        )
        aggr_messages = torch.zeros_like(target_node)
        scatter(messages, i_target, 0, reduce=self.reduce, out=aggr_messages)
        return self.ln(
            self.final_act(
                self.residual_proj(target_node)
                + self.fn_combine(torch.cat((target_node, aggr_messages), dim=1))
            )
        )


class RBFLayer(EGNNMessageLayer):
    ...


class EGNN(nn.Module):
    def __init__(
        self,
        edge_attr_size: int,
        hidden_channels: int,
        final_embedding_size: int = None,
        target_size: int = 1,
        num_mp_layers: int = 2,
        # mp_type: Callable[..., EGNNMessageLayer] = EGNNMessageLayer,
        node_types: List[NodeType] = [],
        edge_types: List[EdgeType] = [],
        act: str = "elu",
    ) -> None:
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types

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
                        "_".join(edge_type): EGNNMessageLayer(
                            source_node_size=d_in,
                            target_node_size=d_in,
                            edge_attr_size=edge_attr_size,
                            distance_size=1,
                            hidden_channels=d_out,
                            output_channels=d_out,
                            act=act,
                            final_act=act,
                        )
                        for edge_type in edge_types
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
            nn.Sigmoid(),
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
                )
            node_embed = new_node_embed

        return node_embed

    def _predict(self, node_embed: Dict[NodeType, Tensor], data: HeteroData) -> Tensor:
        aggr = {nt: self.aggregation(x, data[nt].batch) for nt, x in node_embed.items()}
        aggr = aggr["ligand"]
        return self.f_predict(aggr) * 14

    def forward(self, data: HeteroData) -> Tensor:
        node_embed = self.encode(data)
        pred = self._predict(node_embed, data)
        return pred


if __name__ == "__main__":
    from kinodata.dataset import KinodataDocked
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
