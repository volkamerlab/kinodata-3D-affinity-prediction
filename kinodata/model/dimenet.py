from torch import nn
from torch_geometric.nn import DimeNetPlusPlus

from kinodata.model.regression import RegressionModel
from kinodata.model.resolve import resolve_aggregation
from kinodata.types import NodeType


class LinActSkipNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.lin(x)) + x)


class DimeNetWrapper(RegressionModel):
    def __init__(self, config):
        super().__init__(config)
        self.dime_net = config.init(DimeNetPlusPlus)
        self.agg = resolve_aggregation(config.get("agg", "sum"))
        dimenet_out_features = self.dime_net.output_blocks[0].lin.out_features
        hidden_channels = config.get("hidden_channels")
        self.readout = nn.Sequential(
            nn.Linear(dimenet_out_features, hidden_channels),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus(),
        )

    def forward(self, batch):
        repr = self.dime_net(
            batch[NodeType.Complex].z,
            batch[NodeType.Complex].pos,
            batch[NodeType.Complex].batch,
        )
        repr = self.agg(repr, batch[NodeType.Complex].batch)
        out = self.readout(repr)
        return out
