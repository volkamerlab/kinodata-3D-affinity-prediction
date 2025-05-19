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


def resolve_loss(loss_type: str) -> nn.Module:
    loss_type = loss_type.lower().lstrip()
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type in ("mae", "l1"):
        return nn.L1Loss()
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError(loss_type)


class _NoAggregation(aggr.Aggregation):
    def forward(self, x, *args, **kwargs):
        return x


def resolve_aggregation(aggr_type: str | None) -> aggr.Aggregation:
    if aggr_type == "sum":
        return aggr.SumAggregation()
    if aggr_type == "mean":
        return aggr.MeanAggregation()
    if aggr_type == "max":
        return aggr.MaxAggregation()
    if aggr_type == "attention":
        return aggr.SoftmaxAggregation(learn=True)
    if aggr_type == "none":
        return _NoAggregation()
    if aggr_type is None:
        return _NoAggregation()
    raise ValueError(aggr_type)
