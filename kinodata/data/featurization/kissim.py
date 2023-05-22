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


def add_kissim_fp(data: Data, fp: pd.DataFrame) -> Data:
    fp = torch.from_numpy(fp.values)
    data.kissim_fp = fp.unsqueeze(0)
    return data
