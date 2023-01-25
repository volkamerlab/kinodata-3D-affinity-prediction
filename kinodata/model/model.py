from typing import Any, Dict
import torch.nn as nn
import pytorch_lightning as pl
from kinodata.model.egnn import EGNN
import torch
from torch.nn.functional import smooth_l1_loss
import matplotlib.pyplot as plt
import numpy as np
import wandb


def resolve_loss(loss_type: str) -> nn.Module:
    if loss_type == "mse":
        return nn.MSELoss()
    raise ValueError(loss_type)


class Model(pl.LightningModule):
    def __init__(
        self,
        node_types,
        edge_types,
        hidden_channels: int,
        num_mp_layers: int,
        act: str,
        lr: float,
        batch_size: int,
        weight_decay: float,
        mp_type: str,
        loss_type: str,
        mp_kwargs: Dict[str, Any] = None,
    ) -> None:
        super().__init__()
        hparams = [
            "hidden_channels",
            "num_mp_layers",
            "act",
            "lr",
            "batch_size",
            "weight_decay",
            "mp_type",
            "loss_type",
        ]
        self.save_hyperparameters(*hparams)
        self.criterion = resolve_loss(loss_type)
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
        pred = self.egnn(batch).view(-1, 1)
        loss = self.criterion(pred, batch.y.view(-1, 1))
        self.log("train_loss", loss, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, *args):
        pred = self.egnn(batch).flatten()
        val_mae = (pred - batch.y).abs().mean()
        self.log("val_mae", val_mae, batch_size=self.hparams.batch_size)
        return {"val_mae": val_mae, "pred": pred, "target": batch.y}

    def validation_epoch_end(self, outputs, *args, **kwargs) -> None:
        super().validation_epoch_end(outputs)
        pred = torch.cat([output["pred"] for output in outputs], 0)
        target = torch.cat([output["target"] for output in outputs], 0)
        corr = ((pred - pred.mean()) * (target - target.mean())).mean() / (
            pred.std() * target.std()
        ).cpu().item()
        y_min = min(pred.min().cpu().item(), target.min().cpu().item())
        y_max = max(pred.max().cpu().item(), target.max().cpu().item())
        fig, ax = plt.subplots()
        ax.scatter(target.cpu().numpy(), pred.cpu().numpy(), s=0.7)
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Pred")
        ax.set_xlabel("Target")
        ax.set_title(f"corr={corr}")
        wandb.log({"scatter_test": wandb.Image(fig)})
        plt.close(fig)
        self.log("test_corr", corr)

    def predict_step(self, batch, *args):
        return {"pred": self.egnn(batch).flatten(), "target": batch.y}

    def test_step(self, batch, *args, **kwargs):
        ...

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.egnn.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
