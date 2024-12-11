from collections import defaultdict
from rdkit.Chem.rdchem import BondType as BT

BOND_TYPE_TO_IDX = defaultdict(int)  # other bonds will map to 0
BOND_TYPE_TO_IDX[BT.SINGLE] = 1
BOND_TYPE_TO_IDX[BT.DOUBLE] = 2
BOND_TYPE_TO_IDX[BT.TRIPLE] = 3
BOND_TYPE_TO_IDX[BT.AROMATIC] = 4
NUM_BOND_TYPES = len(BOND_TYPE_TO_IDX) + 1
