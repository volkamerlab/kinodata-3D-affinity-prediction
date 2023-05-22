from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.models import MLP
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
from kinodata.model.shared.node_embedding import HeteroEmbedding

from kinodata.types import EdgeType, NodeEmbedding, NodeType


class DistanceScaleShift(Protocol):
    def __call__(self, distance: Tensor) -> Tuple[Tensor, Tensor]:
        ...


class ExpnormRBFEmbedding(nn.Module):
    def __init__(self, size: int, d_cut: float, trainable: bool = False) -> None:
        super().__init__()
        self.size = size
        self.d_cut = nn.parameter.Parameter(
            torch.tensor([d_cut], dtype=torch.float32), requires_grad=False
        )

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    # taken from torchmd-net
    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(-self.d_cut.squeeze())
        means = torch.linspace(start_value, 1, self.size)
        betas = torch.tensor([(2 / self.size * (1 - start_value)) ** -2] * self.size)
        return means, betas

    def cosine_cutoff(self, d: Tensor) -> Tensor:
        d = d.clamp_max(self.d_cut)
        return 0.5 * (torch.cos(d * torch.pi / self.d_cut) + 1)

    def forward(self, d: Tensor) -> Tensor:
        cutoff = self.cosine_cutoff(d)
        rbf = torch.exp(-self.betas * (torch.exp(-d) - self.means).pow(2))
        return cutoff * rbf


class RBFDistanceScaleShift(nn.Module):
    def __init__(self, size: int, d_cut: float, act: str) -> None:
        super().__init__()
        self.rbf = ExpnormRBFEmbedding(size, d_cut)
        self.lin = Linear(size, size * 2)
        if act == "none":
            self.act = nn.Identity()
        else:
            self.act = activation_resolver(act)

    def forward(self, distance: Tensor) -> Tuple[Tensor, Tensor]:
        if distance.dim() == 1:
            distance = distance.view(-1, 1)
        out = self.act(self.lin(self.rbf(distance)))
        scale, shift = torch.chunk(out, 2, dim=1)
        return scale, shift


class EGINConv(MessagePassing):
    def __init__(
        self,
        nn_node: torch.nn.Module,
        nn_dist: DistanceScaleShift,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.nn_node = nn_node
        self.nn_dist = nn_dist
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn_node, torch.nn.Sequential):
                nn_node = self.nn_node[0]
            if hasattr(nn_node, "in_features"):
                in_channels = nn_node.in_features
            elif hasattr(nn_node, "in_channels"):
                in_channels = nn_node.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn_node)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: Tensor,  # distance
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        scale, shift = self.nn_dist(edge_weight)
        out = self.propagate(
            edge_index, x=x, scale=scale, shift=shift, edge_attr=edge_attr, size=size
        )

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn_node(out)

    def message(
        self, x_j: Tensor, scale: Tensor, shift: Tensor, edge_attr: OptTensor
    ) -> Tensor:
        if (
            self.lin is None
            and edge_attr is not None
            and x_j.size(-1) != edge_attr.size(-1)
        ):
            raise ValueError(
                "Node and edge feature dimensionalities do not "
                "match. Consider setting the 'edge_dim' "
                "attribute of 'EGINConv'"
            )

        if edge_attr is not None and self.lin is not None:
            edge_attr = self.lin(edge_attr)

        if edge_attr is not None:
            x_j = x_j + edge_attr

        return ((x_j) * scale + shift).relu()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(nn_node={self.nn_node}, nn_dist={self.nn_dist})"
        )


class EGIN(BasicGNN):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        d_cut: float,
        out_channels: Optional[int] = None,
        dropout: float = 0,
        act: Union[str, Callable, None] = "relu",
        d_act: str = "none",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        self.d_cut = d_cut
        self.d_act = d_act
        self.edge_dim = edge_dim
        super().__init__(
            in_channels,
            hidden_channels,
            num_layers,
            out_channels,
            dropout,
            act,
            act_first,
            act_kwargs,
            norm,
            norm_kwargs,
            jk,
            **kwargs,
        )

    supports_edge_weight = True
    supports_edge_attr = True

    def init_conv(
        self, in_channels: int, out_channels: int, **kwargs
    ) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        nn_dist = RBFDistanceScaleShift(in_channels, self.d_cut, act=self.d_act)
        return EGINConv(mlp, nn_dist, edge_dim=self.edge_dim)


class HeteroEGIN(nn.Module):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        hidden_channels: int,
        num_layers: int,
        d_cut: float,
        out_channels: Optional[int] = None,
        dropout: float = 0,
        act: Union[str, Callable, None] = "relu",
        d_act: str = "none",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        jk: Optional[str] = None,
        edge_attr_size: Optional[Dict[EdgeType, int]] = None,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        if edge_attr_size is not None:
            edge_dim = edge_dim if edge_dim else max(edge_attr_size.values())
        egin = EGIN(
            hidden_channels,
            hidden_channels,
            num_layers,
            d_cut,
            out_channels,
            dropout,
            act,
            d_act,
            act_first,
            act_kwargs,
            norm,
            norm_kwargs,
            jk,
            edge_dim,
            **kwargs,
        )
        self.initial_edge_lins = nn.ModuleDict()
        self.edge_key_mapping = dict()
        if edge_attr_size is not None:
            for key, size in edge_attr_size.items():
                self.initial_edge_lins[str(key)] = nn.Linear(size, edge_dim, bias=False)
                self.edge_key_mapping[str(key)] = key

        self.hetero_egin = to_hetero(egin, (node_types, edge_types))

    def forward(self, data: HeteroData, node_embedding: NodeEmbedding) -> NodeEmbedding:
        edge_attr_dict = data.edge_attr_dict
        for key, lin in self.initial_edge_lins.items():
            _key = self.edge_key_mapping[key]
            edge_attr_dict[_key] = lin(edge_attr_dict[_key])

        return self.hetero_egin(
            node_embedding,
            data.edge_index_dict,
            edge_attr=edge_attr_dict,
            edge_weight=data.edge_weight_dict,
        )

    def encode(self, data: HeteroData) -> NodeEmbedding:
        return self(data)


if __name__ == "__main__":

    x = torch.rand(10, 32)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    edge_attr = torch.rand(5, 4)
    distance = torch.rand(5, 1)

    egin = EGIN(32, 64, 3, 1.0, edge_dim=4)

    out = egin(x, edge_index, edge_weight=distance, edge_attr=edge_attr)

    print(out)
