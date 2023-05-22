from pathlib import Path
from typing import Union
import pandas as pd
from biopandas.mol2 import PandasMol2

pmol = PandasMol2()

klifs_mol2_columns = {
    0: ("atom.id", "int32"),
    1: ("atom.name", "string"),
    2: ("atom.x", "float32"),
    3: ("atom.y", "float32"),
    4: ("atom.z", "float32"),
    5: ("atom.type", "string"),
    6: ("residue.subst_id", "int32"),
    7: ("residue.subst_name", "string"),
    8: ("atom.charge", "float32"),
    9: ("atom.status_bit", "string"),
}


def read_klifs_mol2(path: Union[str, Path]) -> pd.DataFrame:
    return pmol.read_mol2(str(path), columns=klifs_mol2_columns).df
