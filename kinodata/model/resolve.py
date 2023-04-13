import torch.nn as nn
from torch_geometric.nn import aggr


_act = {
    "sigmoid": nn.Sigmoid(),
    "softplus": nn.Softplus(),
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "silu": nn.SiLU(),
    "none": nn.Identity(),
}


def resolve_act(act: str) -> nn.Module:
    try:
        return _act[act]
    except KeyError:
        raise ValueError(act)


def resolve_loss(loss_type: str) -> nn.Module:
    loss_type = loss_type.lower()
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "mae":
        return nn.L1Loss()
    raise ValueError(loss_type)


def resolve_aggregation(aggr_type: str) -> aggr.Aggregation:
    if aggr_type == "sum":
        return aggr.SumAggregation()
    raise ValueError(aggr_type)
