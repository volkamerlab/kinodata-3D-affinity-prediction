from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from torch import Tensor
import wandb

from kinodata.model.model import Model
from kinodata.model.egnn import EGNN
from kinodata.model.readout import HeteroReadout
from kinodata.model.resolve import resolve_loss, resolve_aggregation
from kinodata.typing import NodeType, EdgeType


class RegressionModel(Model):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        hidden_channels: int,
        num_mp_layers: int,
        act: str,
        lr: float,
        batch_size: int,
        weight_decay: float,
        mp_type: str,
        loss_type: str,
        mp_kwargs: Optional[Dict[str, Any]] = None,
        readout_node_types: Optional[List[NodeType]] = None,
        readout_aggregation_type: str = "sum",
        use_bonds: bool = False,
        final_act: str = "softplus",
    ) -> None:
        self.save_hyperparameters()
        super().__init__(
            node_types,
            edge_types,
            hidden_channels,
            num_mp_layers,
            act,
            lr,
            batch_size,
            weight_decay,
            mp_type,
            mp_kwargs,
            use_bonds,
        )

        # default: use all nodes for readout
        if readout_node_types is None:
            readout_node_types = node_types

        self.readout = HeteroReadout(
            readout_node_types,
            resolve_aggregation(readout_aggregation_type),
            hidden_channels,
            hidden_channels,
            1,
            act=act,
            final_act=final_act,
        )
        self.criterion = resolve_loss(loss_type)
        self.define_metrics()

    def define_metrics(self):
        wandb.define_metric("val_mae", summary="min")

    def forward(self, batch) -> Tensor:
        node_embed = self.egnn(batch)
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
        wandb.log({"scatter_val": wandb.Image(fig)})
        plt.close(fig)
        self.log("val_corr", corr)

    def predict_step(self, batch, *args):
        pred = self.forward(batch).flatten()
        return {"pred": pred, "target": batch.y.flatten()}

    def test_step(self, batch, *args, **kwargs):
        ...
