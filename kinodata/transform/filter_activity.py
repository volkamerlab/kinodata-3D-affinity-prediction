from typing import List, Sequence
from enum import Enum

from torch_geometric.data import Data


class ActivityTypes(Enum):
    ic50 = "pIC50"
    kd = "pKd"
    ki = "pKi"


class FilterActivityType:
    def __init__(self, allowed: Sequence[str]):
        self.allowed = set(allowed)
        assert self.allowed.issubset(set(iter(ActivityTypes)))

    def __call__(self, data: Data) -> bool:
        return data.activity_type in self.allowed
