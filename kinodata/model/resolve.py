from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as opt
from torch_geometric.nn import aggr


_act = {
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "silu": nn.SiLU(),
    "none": nn.Identity(),
}


def resolve_optim(optim: str) -> opt.Optimizer:
    optim = optim.lower().strip()
    if optim == "adam":
        return opt.Adam
    if optim == "adamw":
        return opt.AdamW
    if optim == "radam":
        return opt.RAdam
    if optim == "rmsprop":
        return opt.RMSprop
    if optim == "sgd":
        return opt.SGD
    raise ValueError(optim)


def resolve_act(act: str) -> nn.Module:
    try:
        return _act[act]
    except KeyError:
        raise ValueError(act)


class CriterionWithBaseline(nn.Module):
    def __init__(self, criterion) -> None:
        super().__init__()
        self.criterion = criterion

    def forward(self, pred: Tensor, baseline, target: Tensor):
        return self.criterion(pred, target) + self.criterion(
            baseline, torch.zeros_like(baseline)
        )


def resolve_loss(
    loss_type: str,
    with_baseline: bool = False,
) -> nn.Module:
    loss_type = loss_type.lower().lstrip()
    if loss_type == "mse":
        loss = nn.MSELoss()
    elif loss_type in ("mae", "l1"):
        loss = nn.L1Loss()
    elif loss_type == "smooth_l1":
        loss = nn.SmoothL1Loss()
    else:
        raise ValueError(loss_type)
    if with_baseline:
        return CriterionWithBaseline(loss)
    return loss


def resolve_aggregation(aggr_type: str) -> aggr.Aggregation:
    if aggr_type == "sum":
        return aggr.SumAggregation()
    raise ValueError(aggr_type)
