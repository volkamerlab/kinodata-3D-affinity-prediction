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


class LigandGNNBaseline(RegressionModel):
    def __init__(
        self,
        config: Config,
        encoder: BasicGNN,
        aggr: Aggregation,
    ) -> None:
        super().__init__(config)
        self.initial_embedding = Embedding(100, self.hparams.hidden_channels)
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
        x = self.initial_embedding(batch["ligand"].z)
        h = self.encoder.forward(
            x,
            batch["ligand", "bond", "ligand"].edge_index,
            edge_attr=batch["ligand", "bond", "ligand"].edge_attr,
        )
        aggr = self.aggr.forward(h, index=batch["ligand"].batch)
        pred = self.prediction_head(aggr)
        return pred
