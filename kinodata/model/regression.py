from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

from kinodata.configuration import Config
from kinodata.model.resolve import resolve_loss
from kinodata.model.resolve import resolve_optim
from kinodata.transform.baseline_masking import MaskLigandPosition


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
            # what have i done
            return torch.tensor([int(x) for x in sub_data[key]])
        raise ValueError(sub_data, key, "cannot convert to tensor")

    return {
        key: torch.cat([ensure_tensor(sub_data, key) for sub_data in data], dim=dim)
        for key in subset
    }


class RegressionModel(pl.LightningModule):
    log_scatter_plot: bool = False
    log_test_predictions: bool = False

    def __init__(self, config: Config, prediction_model_cls: type[nn.Module]) -> None:
        super().__init__()
        self.config = config
        self.model = config.init(prediction_model_cls)
        self.save_hyperparameters(config)  # triggers wandb hook
        self.define_metrics()
        self.set_criterion()

    def set_criterion(self):
        self.criterion = resolve_loss(self.config.loss_type)

    def forward(self, batch) -> Tensor:
        return self.model(batch)

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

    def training_step(self, batch, *args) -> Tensor:
        pred = self.forward(batch).view(-1, 1)
        loss = self.criterion(pred, batch.y.view(-1, 1))
        self.log("train/loss", loss, batch_size=pred.size(0), on_epoch=True)
        return loss

    def validation_step(self, batch, *args, key: str = "val"):
        pred = self.forward(batch).flatten()
        val_mae = (pred - batch.y).abs().mean()
        self.log(f"{key}/mae", val_mae, batch_size=pred.size(0), on_epoch=True)
        return {
            f"{key}/mae": val_mae,
            "pred": pred,
            "target": batch.y,
            "ident": batch.ident,
        }

    def process_eval_outputs(self, outputs) -> float:
        pred = torch.cat([output["pred"] for output in outputs], 0)
        target = torch.cat([output["target"] for output in outputs], 0)
        corr = ((pred - pred.mean()) * (target - target.mean())).mean() / (
            pred.std() * target.std()
        ).cpu().item()
        mae = (pred - target).abs().mean()
        return pred, target, corr, mae

    def validation_epoch_end(self, outputs, *args, **kwargs) -> None:
        super().validation_epoch_end(outputs)
        pred, target, corr, mae = self.process_eval_outputs(outputs)
        self.log("val/corr", corr)

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

    def test_epoch_end(self, outputs, *args, **kwargs) -> None:
        pred, target, corr, mae = self.process_eval_outputs(outputs)
        self.log("test/mae", mae)
        self.log("test/corr", corr)


class DebiasingBaselineRegression(RegressionModel):

    def __init__(self, config, prediction_model_cls):
        super().__init__(config, prediction_model_cls)
        self.set_biasing_mask()

    def set_biasing_mask(self):
        if self.config.get("biasing_mask", None) is None:
            raise ValueError("Biasing mask must be provided for debiasing baseline")
        if self.config.biasing_mask == "remove_pl_interactions":
            self.biasing_mask = self.config.get("pl_interactions")
        else:
            raise ValueError("Unknown biasing mask")

    def debiasing_criterion(
        self,
        data,
        biased_pred: Tensor,
    ):
        baseline = data["y_group_mean"].flatten()
        return F.mse_loss(biased_pred.flatten(), baseline)

    def forward(self, batch) -> Tensor:
        clean_pred = self.model(batch)
        biased_pred = self.model(self.biasing_mask(batch))
        return clean_pred, biased_pred

    def training_step(self, batch, *args) -> Tensor:
        clean_pred, biased_pred = self.forward(batch)
        loss = self.criterion(clean_pred.flatten(), batch["y_delta"].flatten())
        debiasing_loss = self.debiasing_criterion(batch, biased_pred)
        loss = loss + debiasing_loss
        self.log("train/loss", loss, batch_size=clean_pred.size(0), on_epoch=True)
        return loss

    def validation_step(self, batch, *args, key: str = "val"):
        clean_pred, biased_pred = self.forward(batch)

        pred = clean_pred.flatten() + biased_pred.flatten()
        pred_ = clean_pred.flatten() + batch["y_group_mean"].flatten()

        mae_baseline = (biased_pred - batch["y_group_mean"]).abs().mean()
        val_mae = (pred - batch.y).abs().mean()
        val_mae_ = (pred_ - batch.y).abs().mean()
        self.log(f"{key}/mae", val_mae, batch_size=pred.size(0), on_epoch=True)
        self.log(f"{key}/mae_", val_mae_, batch_size=pred.size(0), on_epoch=True)
        self.log(
            f"{key}/mae_baseline", mae_baseline, batch_size=pred.size(0), on_epoch=True
        )
        return {
            "pred": pred,
            "target": batch.y,
        }

    def process_eval_outputs(self, outputs) -> float:
        pred = torch.cat([output["pred"] for output in outputs], 0)
        target = torch.cat([output["target"] for output in outputs], 0)
        corr = ((pred - pred.mean()) * (target - target.mean())).mean() / (
            pred.std() * target.std()
        ).cpu().item()
        mae = (pred - target).abs().mean()
        return pred, target, corr, mae

    def validation_epoch_end(self, outputs, *args, **kwargs) -> None:
        super().validation_epoch_end(outputs)
        pred, target, corr, mae = self.process_eval_outputs(outputs)
        self.log("val/corr", corr)

    def predict_step(self, batch, *args):
        clean_pred, baseline_pred = self.forward(batch).flatten()
        pred = clean_pred + baseline_pred
        return {
            "clean_pred": clean_pred.flatten(),
            "baseline_pred": baseline_pred.flatten(),
            "pred": pred.flatten(),
            "target": batch.y.flatten(),
            "chembl_activity_id": batch.chembl_activity_id,
        }

    def test_step(self, batch, *args, **kwargs):
        info = self.validation_step(batch, key="test")
        return info

    def test_epoch_end(self, outputs, *args, **kwargs) -> None:
        pred, target, corr, mae = self.process_eval_outputs(outputs)
        self.log("test/mae", mae)
        self.log("test/corr", corr)
