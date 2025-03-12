from .dataset import KinodataDocked, KinodataDockedAgnostic, Filtered
from .dataset import _DATA as DATA
from .voxel.klifs_parser import klifs_mol2_columns as KLIFS_MOL2_COLUMNS

RAW_DATA_DIR = DATA / "raw"
PROCESSED_DATA_DIR = DATA / "processed"
