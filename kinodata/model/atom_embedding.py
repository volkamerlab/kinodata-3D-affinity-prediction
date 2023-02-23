from typing import Dict
from torch.nn import Embedding
from torch import Tensor


class HeteroEmbedding(Embedding):
    def forward(self, **inputs: Tensor) -> Dict[str, Tensor]:
        return {
            key: super(HeteroEmbedding, self).forward(tensor_value)
            for key, tensor_value in inputs.items()
        }
