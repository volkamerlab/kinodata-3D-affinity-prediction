from typing import List, Sequence, Callable
from enum import Enum

from torch_geometric.data import Data


class ActivityTypes(Enum):
    ic50 = "pIC50"
    kd = "pKd"
    ki = "pKi"


class FilterActivityType:
    def __init__(self, allowed: Sequence[str]):
        self.allowed = set(allowed)
        assert self.allowed.issubset(set([at.value for at in ActivityTypes]))

    def __call__(self, data: Data) -> bool:
        return data.activity_type in self.allowed


class FilterActivityScore:
    """
    Filter an activity by its `docking_score` attribute using `threshold` as an upper threshold.
    """
    def __init__(self, threshold: float = 0):
        self.threshold = threshold

    def __call__(self, data: Data) -> bool:
        return data.docking_score < self.threshold


class Compose:
    """
    Compose multiple activity filters using 'and'.
    """
    def __init__(self, filters: List[Callable]):
        self.filters = filters

    def __call__(self, data: Data) -> bool:
        return all(f(data) for f in self.filters)
