from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Embedding

from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP

from kinodata.model.resolve import resolve_loss
from kinodata.model.regression import RegressionModel
from kinodata.configuration import Config
from kinodata.model.shared.node_embedding import (
    FeatureEmbedding,
    AtomTypeEmbedding,
    CombineSum,
)
from kinodata.types import NodeType, RelationType


class LigandGNNBaseline(RegressionModel):
    def __init__(
        self,
        config: Config,
        encoder: BasicGNN,
        aggr: Aggregation,
    ) -> None:
        super().__init__(config)
        self.initial_embedding = CombineSum(
            [
                AtomTypeEmbedding(
                    NodeType.Ligand,
                    hidden_chanels=config.hidden_channels,
                    act=config.act,
                ),
                FeatureEmbedding(
                    NodeType.Ligand,
                    hidden_channels=config.hidden_channels,
                    act=config.act,
                ),
            ]
        )
        self.encoder = encoder
        self.aggr = aggr
        self.prediction_head = MLP(
            channel_list=[
                self.hparams.hidden_channels,
                self.hparams.hidden_channels,
                1,
            ],
            plain_last=True,
        )

        self.criterion = resolve_loss(self.hparams.loss_type)
        self.define_metrics()

    def forward(self, batch: HeteroData) -> Tensor:
        x = self.initial_embedding(batch)
        h = self.encoder.forward(
            x,
            batch[NodeType.Ligand, RelationType.Bond, NodeType.Ligand].edge_index,
            edge_attr=batch[
                NodeType.Ligand, RelationType.Bond, NodeType.Ligand
            ].edge_attr,
        )
        aggr = self.aggr.forward(h, index=batch[NodeType.Ligand].batch)
        pred = self.prediction_head(aggr)
        return pred
