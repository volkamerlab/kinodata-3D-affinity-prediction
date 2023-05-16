from torch import Tensor
from torch.nn import Module, ModuleList, Sequential, Linear
from torch_geometric.nn import GINEConv, GraphNorm
from kinodata.model.resolve import resolve_act


class GINE(Module):
    def __init__(
        self, channels: int, num_layers: int, edge_channels: int, act: str = "silu"
    ) -> None:
        super().__init__()
        self.act = resolve_act(act)
        self.conv_layers = ModuleList()
        self.norm_layers = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels), self.act, Linear(channels, channels)
            )
            self.conv_layers.append(GINEConv(nn, edge_dim=channels))
            self.norm_layers.append(GraphNorm(channels))
        self.lin_edge = Sequential(Linear(edge_channels, channels), self.act)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor
    ) -> Tensor:
        edge_attr = self.lin_edge(edge_attr.float())
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = norm(conv(x=x, edge_index=edge_index, edge_attr=edge_attr), batch)
        return x
