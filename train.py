import sys
import os

sys.path.append(".")

from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

from kinodata.dataset import KinodataDocked
from kinodata.transform import AddDistancesAndInteractions
from kinodata.model.egnn import EGNN
import wandb
import torch
from torch.utils.data import random_split
from pytorch_lightning.loggers.wandb import WandbLogger

from torch.nn.functional import smooth_l1_loss


class Model(pl.LightningModule):
    def __init__(
        self, node_types, edge_types, hidden_channels: int, num_mp_layers: int, act: str
    ) -> None:
        super().__init__()
        self.save_hyperparameters("hidden_channels", "num_mp_layers", "act")
        self.egnn = EGNN(
            edge_attr_size=0,
            hidden_channels=hidden_channels,
            final_embedding_size=hidden_channels,
            target_size=1,
            num_mp_layers=num_mp_layers,
            act=act,
            node_types=node_types,
            edge_types=edge_types,
        )

    def training_step(self, batch, *args):
        pred = self.egnn(batch).flatten()
        loss = smooth_l1_loss(pred, batch.y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, *args):
        pred = self.egnn(batch).flatten()
        val_mae = (pred - batch.y).abs().mean()
        self.log("val_mae", val_mae)
        return {"val_mae": val_mae}

    def configure_optimizers(self):
        return torch.optim.Adam(self.egnn.parameters(), lr=1e-3)


if __name__ == "__main__":
    dataset = KinodataDocked(transform=AddDistancesAndInteractions(radius=2.0))
    node_types, edge_types = dataset[0].metadata()
    model = Model(node_types, edge_types, 64, 3, "elu")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader, val_loader = DataLoader(train_data, 64, True), DataLoader(
        val_data, 64, False
    )

    key = os.environ["WANDB_API_KEY"]
    wandb.login(key=key)
    logger = WandbLogger(project="Kinodata", log_model=True)

    trainer = pl.Trainer(
        logger=logger,
        auto_select_gpus=True,
        max_epochs=100,
        accelerator="gpu",
    )

    trainer.fit(model, train_loader, val_loader)
