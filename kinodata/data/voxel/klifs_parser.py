from kinodata.data.io.read_klifs_mol2 import klifs_mol2_columns
from docktgrid.molecule import MolecularData, MolecularParser
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2


klifs_mol2_columns = {
    0: ("atom_id", "int32"),
    1: ("atom_name", "string"),
    2: ("x", "float32"),
    3: ("y", "float32"),
    4: ("z", "float32"),
    5: ("atom_type", "string"),
    6: ("subst_id", "int32"),
    7: ("subst_name", "string"),
    8: ("charge", "float32"),
    9: ("atom.status_bit", "string"),
}


class KlifsSymbolParser(MolecularParser):

    def _remap_klifs_atom_symbol(self, symbol: str) -> str:
        # CA (carbon alpha), CB (carbon beta), ...
        if symbol.startswith("C") and symbol.upper() == symbol:
            return "C"
        if symbol.startswith("H") and symbol.upper() == symbol:
            return "H"
        # occurs 4 times, idek
        if symbol == "A":
            return "C"
        # https://www.ebi.ac.uk/pdbe-srv/pdbechem/atom/show?cid=CRO&name=CG1
        if symbol == "Cg1":
            return "C"
        # https://www.ebi.ac.uk/pdbe-srv/pdbechem/atom/show?cid=CRO&name=CG1
        if symbol == "Cg":
            return "C"
        # Leucine oxygen https://www.ebi.ac.uk/pdbe-srv/pdbechem/atom/show?cid=LEU&name=OXT
        if symbol == "Oxt":
            return "O"
        # Threononine oxygen https://www.ebi.ac.uk/pdbe-srv/pdbechem/atom/show?cid=THR&name=OG1
        if symbol == "Og1":
            return "O"
        if symbol == "N1":
            return "N"
        if symbol == "C3*":
            return "C"
        if symbol == "O5*":
            return "O"
        if symbol == "Nz":
            return "N"
        return symbol

    def get_element_symbols_mol2(self):
        symbols = super().get_element_symbols_mol2()
        return [self._remap_klifs_atom_symbol(symbol) for symbol in symbols]


class KlifsPocketParser(KlifsSymbolParser):

    def parse_file(self, mol_file: str, ext: str) -> MolecularData:
        """Parse molecular file and return a MolecularData object."""
        self.ppdb = PandasPdb()
        self.pmol2 = PandasMol2()

        if ext.lower() in ("pdb", ".pdb"):  # PDB file format
            mol = self.ppdb.read_pdb(mol_file)
            self.df_atom = mol.df["ATOM"]
            self.df_hetatm = mol.df["HETATM"]
            return MolecularData(
                mol, self.get_coords_pdb(), self.get_element_symbols_pdb()
            )
        elif ext.lower() in ("mol2", ".mol2"):  # MOL2 file format
            mol = self.pmol2.read_mol2(mol_file, columns=klifs_mol2_columns)
            self.df_atom = mol.df
            return MolecularData(
                mol, self.get_coords_mol2(), self.get_element_symbols_mol2()
            )
        else:
            raise NotImplementedError(f"File format {ext} not implemented.")

    def get_element_symbols_mol2(self) -> list[str]:
        symbols = self.df_atom["atom_type"].values
        symbols = [symbol.split(".")[0] for symbol in symbols]
        return symbols
