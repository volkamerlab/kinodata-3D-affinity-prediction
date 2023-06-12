from typing import List, Optional
import torch
from torch_geometric.data import Data
from pathlib import Path
import pandas as pd

_DATA = Path(__file__).parents[3] / "data"


# we precomputed kissim fps
def load_kissim(structure_id: int) -> pd.DataFrame:
    path = _DATA / "processed" / "kissim" / f"{structure_id}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


PHYSICOCHEMICAL = ["size", "hbd", "hba", "charge", "aromatic", "aliphatic"]
STRUCTURAL = ["sco", "exposure", "hinge_region", "dfg_region", "front_pocket", "center"]


def add_kissim_fp(
    data: Data,
    fp: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> Data:
    if subset is not None:
        fp = fp[list(set(subset))]
    fp = torch.from_numpy(fp.values)
    data.kissim_fp = fp.unsqueeze(0)
    return data
