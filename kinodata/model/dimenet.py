from torch import nn
import torch
from torch.nn.functional import silu
from torch_geometric.nn import DimeNetPlusPlus

from kinodata.model.regression import RegressionModel
from kinodata.model.resolve import resolve_aggregation
from kinodata.types import NodeType


class ReadoutMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, x):
        x = self.norm(x)
        x = silu(self.lin1(x))
        x = self.lin2(x)
        return x


class DimeNetWrapper(RegressionModel):
    def __init__(self, config):
        super().__init__(config)
        self.dime_net = config.init(DimeNetPlusPlus)
        self.agg = resolve_aggregation(config.get("agg", "sum"))
        dimenet_out_features = self.dime_net.output_blocks[0].lin.out_features
        hidden_channels = config.get("hidden_channels")
        self.readout = ReadoutMLP(dimenet_out_features, hidden_channels)

    def forward(self, batch):
        repr = self.dime_net(
            batch[NodeType.Complex].z,
            batch[NodeType.Complex].pos,
            batch[NodeType.Complex].batch,
        )
        repr = self.agg(repr, batch[NodeType.Complex].batch)
        out = self.readout(repr)
        return out

    @torch.no_grad()
    def compute_representation(self, batch):
        repr = self.dime_net(
            batch[NodeType.Complex].z,
            batch[NodeType.Complex].pos,
            batch[NodeType.Complex].batch,
        )
        repr = self.agg(repr, batch[NodeType.Complex].batch)
        readout_repr = silu(self.readout.lin1(self.readout.norm(repr)))
        return {
            "after_aggr": repr,
            "during_readout": readout_repr,
        }
