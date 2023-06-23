import numpy as np
from torch_geometric.data import HeteroData

from kinodata.data.featurization.atoms import (
    AtomFeatures,
    FormalCharge,
    NumHydrogens,
)
from kinodata.data.featurization.residue import add_onehot_residues
import rdkit.Chem as Chem


def setup():
    # thiamine
    return Chem.MolFromSmiles("OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N")


def common_asserts(result):
    assert isinstance(result, np.ndarray)


def test_charge():
    mol = setup()
    result = FormalCharge.compute(mol)
    common_asserts(result)
    # thiamine has 1 charged atom (single pos. charge)
    pos1_index = FormalCharge.mapping[1]
    assert result[:, pos1_index].sum() == 1.0


def test_hydrogen():
    mol = setup()
    result = NumHydrogens.compute(mol)
    common_asserts(result)


def test_combined():
    mol = setup()
    result = AtomFeatures.compute(mol)
    common_asserts(result)


def test_residue():
    data = HeteroData()
    data = add_onehot_residues(data, "AAABBBCDD")
    x = data["pocket_residue"].x
    assert x is not None
    assert (x.sum(dim=1) == 1).all()


if __name__ == "__main__":
    test_charge()
    test_hydrogen()
    test_combined()
    test_residue()
