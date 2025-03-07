import torch
import numpy as np
from torch import Tensor, jit
from torch.nn import Parameter, Module


def gaussian(x, mean, std):
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / ((2 * np.pi**0.5) * std)


class GaussianDistEmbedding(Module):
    def __init__(self, size: int, max_dist: float, learnable: bool = True) -> None:
        super().__init__()
        self.size = size
        self.learnable = learnable
        self.d_cut = Parameter(
            torch.tensor([max_dist], dtype=torch.float32), requires_grad=False
        )

        means, stds = self._initial_params()
        self.means = Parameter(means, requires_grad=learnable)
        self.stds = Parameter(stds, requires_grad=learnable)

    def _initial_params(self):
        means = torch.linspace(0, self.d_cut.item(), self.size)
        stds = torch.ones(self.size)
        return means, stds

    def forward(self, d: Tensor) -> Tensor:
        return gaussian(d.view(-1, 1), self.means, self.stds)
