import torch
from torch import Tensor, jit
from torch.nn import Parameter, Module, Identity, Linear, Sequential, SiLU
from torch.nn.init import normal_


def gaussian_rbf(x, loc, shape):
    return torch.exp(-shape * (x - loc) ** 2)


class GaussianDistEmbedding(Module):
    def __init__(
        self,
        size: int,
        max_dist: float,
        intial_blur: float = 2.0,
        learnable_location: bool = False,
        learnable_shape: bool = True,
        dense_proj: bool = True,
    ) -> None:
        super().__init__()
        self.size = size
        self.d_cut = Parameter(
            torch.tensor([max_dist], dtype=torch.float32), requires_grad=False
        )

        locations, shapes = self._initial_params(intial_blur)
        self.location = Parameter(locations, requires_grad=learnable_location)
        self.shape = Parameter(shapes, requires_grad=learnable_shape)
        self.proj = Identity()
        if dense_proj:
            self.proj = Sequential(
                Linear(self.size, self.size),
                SiLU(),
                Linear(self.size, self.size, bias=False),
            )

    @property
    def bin_size(self):
        return self.d_cut / (self.size - 1)

    def _initial_params(self, blur: float = 1):
        locations = torch.linspace(0, self.d_cut.item(), self.size)
        shapes = torch.ones(self.size) * (1 / (2 * (blur**2) * self.bin_size.pow(2)))
        return locations, shapes

    def forward(self, d: Tensor) -> Tensor:
        return self.proj(gaussian_rbf(d.view(-1, 1), self.location, self.shape))
