import torch
from torch_geometric.transforms import BaseTransform


class LinearScalarTransform(BaseTransform):
    def __init__(
        self,
        target_attr: str,
        add: torch.Tensor | float,
        mul: torch.Tensor | float = 1.0,
    ):
        super().__init__()
        self.target_attr = target_attr
        self.add = add
        self.mul = mul

    def forward(self, data):
        x = getattr(data, self.target_attr)
        setattr(data, self.target_attr, x * self.mul + self.add)
        return data
