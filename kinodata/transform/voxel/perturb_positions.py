from functools import partial
import torch

from docktgrid import MolecularComplex
from docktgrid.config import DTYPE
from .base import BaseVoxelTransform


def perturb_position(positions: torch.Tensor, std: float):
    noise = torch.randn_like(positions) * std
    torch.add(positions, noise, out=positions)


class PerturbPosition(BaseVoxelTransform):

    def __init__(self, std: float):
        super().__init__()
        self._perturb = partial(perturb_position, std=std)

    def __call__(self, molecule: MolecularComplex):
        self._perturb(molecule.coords)
        ligand_coords = molecule.coords[:, molecule.n_atoms_protein :]
        molecule.ligand_center.copy_(torch.mean(ligand_coords, 1).to(dtype=DTYPE))
