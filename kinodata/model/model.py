from typing import Any, Dict, List, Optional, Protocol

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torch_geometric.data import HeteroData

from kinodata.model.egnn import EGNN
from kinodata.typing import NodeEmbedding
from kinodata.configuration import Config


class Encoder(Protocol):
    def encode(self, data: HeteroData) -> NodeEmbedding:
        ...


class Model(pl.LightningModule):
    def __init__(self, encoder: Encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def configure_optimizers(self):
        optim = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optim,
            mode="min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.min_lr,
        )

        return [optim], [
            {
                "scheduler": scheduler,
                "monitor": "val/mae",
                "interval": "epoch",
                "frequency": 1,
            }
        ]
