from pathlib import Path

from torch_geometric.data import HeteroData

from kinodata.data.featurization.biopandas import (
    add_pocket_information,
    prepare_pocket_information,
    remove_hydrogens,
)

data = Path(__file__).parents[1] / "data"


def test_prepare_data():
    df_atom, df_bond, df_residue = prepare_pocket_information(
        data / "raw/mol2/pocket/86_pocket.mol2"
    )
    df_atom, df_bond = remove_hydrogens(df_atom, df_bond)
    pass


def test_add_data():
    pyg_data = HeteroData()
    pyg_data = add_pocket_information(pyg_data, data / "raw/mol2/pocket/86_pocket.mol2")
    assert pyg_data is not None


if __name__ == "__main__":
    test_prepare_data()
    test_add_data()
