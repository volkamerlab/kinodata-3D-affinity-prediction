from docktgrid import RandomRotation as RR
from docktgrid import MolecularComplex
from .base import BaseVoxelTransform


class RandomRotation(BaseVoxelTransform):
    _random_rotation = RR(inplace=True)

    def __call__(self, molecule: MolecularComplex):
        self._random_rotation(molecule.coords, molecule.ligand_center)
