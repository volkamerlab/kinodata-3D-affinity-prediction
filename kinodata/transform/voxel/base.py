from abc import ABC, abstractmethod
from docktgrid import MolecularComplex


class BaseVoxelTransform(ABC):

    @abstractmethod
    def __call__(self, molecule: MolecularComplex) -> None: ...
