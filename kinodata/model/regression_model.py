from typing import Any, Callable, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import torch
from torch import Tensor
import wandb

from kinodata.model.model import Model
from kinodata.model.readout import HeteroReadout
from kinodata.model.resolve import resolve_loss, resolve_aggregation
from kinodata.configuration import Config


def cat_many(
    data: List[Dict[str, Tensor]], subset: Optional[List[str]] = None, dim: int = 0
) -> Dict[str, Tensor]:
    if subset is None:
        subset = list(data[0].keys())
    assert set(subset).issubset(data[0].keys())
    return {key: torch.cat([sub_data[key] for sub_data in data]) for key in subset}


class RegressionModel(Model):
    def __init__(
        self, config: Config, encoder_cls: Callable[..., torch.nn.Module]
    ) -> None:
        self.save_hyperparameters(config)
        super().__init__(config.init(encoder_cls))

        # default: use all nodes for readout
        readout_node_types = config.node_types

        self.readout = HeteroReadout(
            readout_node_types,
            resolve_aggregation(config.readout_aggregation_type),
            config.hidden_channels,
            config.hidden_channels,
            1,
            act=config.act,
            final_act=config.final_act,
        )
        self.criterion = resolve_loss(config.loss_type)
        self.define_metrics()

    def define_metrics(self):
        wandb.define_metric("val_mae", summary="min")

    def forward(self, batch) -> Tensor:
        node_embed = self.encoder.encode(batch)
        return self.readout(node_embed, batch)

    def training_step(self, batch, *args) -> Tensor:
        pred = self.forward(batch).view(-1, 1)
        loss = self.criterion(pred, batch.y.view(-1, 1))
        self.log("train_loss", loss, batch_size=pred.size(0), on_epoch=True)
        return loss

    def validation_step(self, batch, *args):
        pred = self.forward(batch).flatten()
        val_mae = (pred - batch.y).abs().mean()
        self.log("val_mae", val_mae, batch_size=pred.size(0), on_epoch=True)
        return {
            "val_mae": val_mae,
            "pred": pred,
            "target": batch.y,
            "ident": batch.ident,
        }

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
        wandb.log({"scatter_val": wandb.Image(fig)})
        plt.close(fig)
        self.log("val_corr", corr)

    def predict_step(self, batch, *args):
        pred = self.forward(batch).flatten()
        return {"pred": pred, "target": batch.y.flatten()}

    def test_step(self, batch, *args, **kwargs):
        info = self.validation_step(batch)
        info["test_mae"] = info["val_mae"]
        del info["val_mae"]
        return info

    def test_epoch_end(self, outputs, *args, **kwargs) -> None:
        test_predictions = wandb.Artifact("test_predictions", type="predictions")
        data = cat_many(outputs, subset=["pred", "ident"])
        values = torch.stack(tuple(data.values()), dim=1)
        table = wandb.Table(columns=list(data.keys()), data=values.tolist())
        test_predictions.add(table, "predictions")
        wandb.log_artifact(test_predictions)
        pass
