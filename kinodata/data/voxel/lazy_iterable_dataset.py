from pathlib import Path
from typing import Generator
import torch
from torch.utils.data import IterableDataset
from docktgrid.molecule import MolecularComplex
from docktgrid.voxel import VoxelGrid
from kinodata.data.voxel.klifs_parser import KlifsSymbolParser, KlifsPocketParser


class IterableVoxelDataset(IterableDataset):

    def __init__(
        self,
        data_stream: Generator,
        voxel: VoxelGrid,
    ):
        self.data_stream = data_stream
        self.protein_parser = KlifsPocketParser(columns=None)
        self.ligand_parser = KlifsSymbolParser()
        self.voxel = voxel

    def _parse_ligand_file(self, f: Path):
        ext = f.suffix
        return self.ligand_parser.parse_file(f, ext)

    def _parse_protein_file(self, f):
        ext = f.name.split(".")[-1]
        return self.protein_parser.parse_file(f.name, ext)

    def __iter__(self):
        for protein_file, ligand_file, metadata in self.data_stream:
            protein_data = self._parse_protein_file(protein_file)
            ligand_data = self._parse_ligand_file(ligand_file)
            complex = MolecularComplex(protein_data, ligand_data)
            vox = self.voxel.voxelize(complex)
            label = torch.zeros(1)
            metadata["ligand_file"] = str(ligand_file)
            yield (
                vox,
                label,
                metadata,
            )
