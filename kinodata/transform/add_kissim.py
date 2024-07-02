import pickle
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph

from ..types import NodeType, RelationType
from ..data.featurization.residue import add_kissim_fp, load_kissim, PHYSICOCHEMICAL, STRUCTURAL
from ..data.dataset import  _DATA


def _apprx_act(act):
    return int(1000 * round(act, 3))

class AddKissimFingerprint(BaseTransform):
    subset = PHYSICOCHEMICAL + STRUCTURAL
    
    def __init__(self) -> None:
        super().__init__()
        with open(_DATA / "processed" / "hacky_klifs_map.pkl", "rb") as fp:
            self.hacky_klifs_map = pickle.load(fp)
            
    def get_structure_id(self, data: HeteroData) -> int:
        hacky_index = (data["pocket_sequence"], _apprx_act(data.y.item()), data["smiles"])
        structure_id = self.hacky_klifs_map[hacky_index]
        return structure_id
    
    def __call__(self, data: HeteroData) -> HeteroData:
        structure_id = self.get_structure_id(data)
        df_kissim = load_kissim(structure_id)
        return add_kissim_fp(data, df_kissim, subset=self.subset)