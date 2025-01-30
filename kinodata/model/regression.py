from typing import Dict, List, Optional

import torch
from torch import nn
from torch import Tensor
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torchmetrics.regression import PearsonCorrCoef
from torchmetrics import MetricCollection

from kinodata.configuration import Config
from kinodata.model.resolve import resolve_loss
from kinodata.model.resolve import resolve_optim


def cat_many(
    data: List[Dict[str, Tensor]], subset: Optional[List[str]] = None, dim: int = 0
) -> Dict[str, Tensor]:
    if subset is None:
        subset = list(data[0].keys())
    assert set(subset).issubset(data[0].keys())

    def ensure_tensor(sub_data, key):
        if isinstance(sub_data[key], torch.Tensor):
            return sub_data[key]
        if isinstance(sub_data[key], list):
            return torch.tensor([int(x) for x in sub_data[key]])
        raise ValueError(sub_data, key, "cannot convert to tensor")

    return {
        key: torch.cat([ensure_tensor(sub_data, key) for sub_data in data], dim=dim)
        for key in subset
    }


class RegressionModel(pl.LightningModule):
    log_scatter_plot: bool = False
    log_test_predictions: bool = False

    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)  # triggers wandb hook
        self.define_metrics()
        self.set_criterion()

    def set_criterion(self):
        self.criterion = resolve_loss(self.config.loss_type)

    def forward(self, batch) -> Tensor:
        raise NotImplementedError

    def configure_optimizers(self):
        Opt = resolve_optim(self.hparams.optim)
        optim = Opt(
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

    def define_metrics(self):
        wandb.init()
        wandb.define_metric("val/mae", summary="min")
        wandb.define_metric("val/corr", summary="max")
        self.correlation_metrics = {
            "train": PearsonCorrCoef(),
            "val": PearsonCorrCoef(),
            "test": PearsonCorrCoef(),
        }
        for metric in self.correlation_metrics.values():
            metric.reset()

    def training_step(self, batch, *args) -> Tensor:
        pred = self.forward(batch).view(-1, 1)
        loss = self.criterion(pred, batch.y.view(-1, 1))
        self.log("train/loss", loss, batch_size=pred.size(0), on_epoch=True)
        self.correlation_metrics["train"](
            pred.detach().cpu().flatten(), batch.y.cpu().flatten()
        )
        return loss

    def validation_step(self, batch, *args, key: str = "val"):
        pred = self.forward(batch).flatten()
        val_mae = (pred - batch.y).abs().mean()
        self.log(f"{key}/mae", val_mae, batch_size=pred.size(0), on_epoch=True)
        self.correlation_metrics[key](
            pred.detach().cpu().flatten(), batch.y.cpu().flatten()
        )
        return {
            f"{key}/mae": val_mae,
            "pred": pred,
            "target": batch.y,
            "ident": batch.ident,
        }

    def _shared_on_epoch_end(self, *args, key: str = None):
        self.log(f"{key}/corr", self.correlation_metrics[key].compute())
        self.correlation_metrics[key].reset()

    def on_train_epoch_end(self):
        return self._shared_on_epoch_end(key="train")

    def on_validation_epoch_end(self):
        return self._shared_on_epoch_end(key="val")

    def on_test_epoch_end(self):
        return self._shared_on_epoch_end(key="test")

    def process_eval_outputs(self, outputs) -> float:
        pred = torch.cat([output["pred"] for output in outputs], 0)
        target = torch.cat([output["target"] for output in outputs], 0)
        corr = ((pred - pred.mean()) * (target - target.mean())).mean() / (
            pred.std() * target.std()
        ).cpu().item()
        mae = (pred - target).abs().mean()
        return pred, target, corr, mae

    def predict_step(self, batch, *args):
        pred = self.forward(batch).flatten()
        return {
            "pred": pred,
            "target": batch.y.flatten(),
            "chembl_activity_id": batch.chembl_activity_id,
        }

    def test_step(self, batch, *args, **kwargs):
        info = self.validation_step(batch, key="test")
        return info
