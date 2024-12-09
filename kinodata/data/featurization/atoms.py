from typing import Callable, Dict, Generic, List, Optional, Sequence, TypeVar, Any
import numpy as np
from rdkit.Chem import rdchem
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os

CT = TypeVar("CT")


class AtomFeaturizer:
    @property
    def size(self) -> int:
        return 0

    def compute(self, mol) -> np.ndarray: ...

    def position_meaning(self, k: int) -> tuple[str, Any]:
        return None

    def get_position(self, key: str, value: Any):
        for j in range(self.size):
            if self.position_meaning(j) == (key, value):
                return j
        return None


class OneHotFeaturizer(AtomFeaturizer, Generic[CT]):
    def __init__(
        self,
        categories: Sequence[CT],
        extractor: Callable[[rdchem.Atom], CT],
        name: Optional[str] = None,
    ):
        self.__mapping = {category: index for index, category in enumerate(categories)}
        self.__rev_mapping = {
            index: category for category, index in self.__mapping.items()
        }
        self.__extractor = extractor
        self.__name = self.__class__.__name__ if name is None else name

    @property
    def mapping(self) -> Dict[CT, int]:
        return self.__mapping

    @property
    def size(self) -> int:
        return len(self.__mapping) + 1

    def position_meaning(self, k: int) -> tuple[str, Any]:
        if k == self.size - 1:
            return self.__name, "default"
        if k in self.__rev_mapping:
            return self.__name, self.__rev_mapping[k]

    def get_category_index(self, atom) -> int:
        try:
            cat = self.__extractor(atom)
            return self.__mapping[cat]
        except (KeyError, ValueError):
            # default category
            return self.size - 1

    def compute(self, mol) -> np.ndarray:
        result = np.zeros((mol.GetNumAtoms(), self.size))
        for i, atom in enumerate(mol.GetAtoms()):
            k = self.get_category_index(atom)
            result[i, k] = 1.0
        return result

    def __repr__(self) -> str:
        return f"{self.__name}({self.__mapping})"


class ComposeOneHot(AtomFeaturizer):
    def __init__(self, oneh_hot_featurizers: List[OneHotFeaturizer]) -> None:
        super().__init__()
        self.__one_hot_featurizers = oneh_hot_featurizers
        self.offsets = np.cumsum(
            np.array([0] + [f.size for f in self.__one_hot_featurizers])
        )[:-1]

    @property
    def size(self) -> int:
        return sum(f.size for f in self.__one_hot_featurizers)

    def compute(self, mol) -> np.ndarray:
        result = np.zeros((mol.GetNumAtoms(), self.size))
        for i, atom in enumerate(mol.GetAtoms()):
            k = np.array(
                [f.get_category_index(atom) for f in self.__one_hot_featurizers]
            )
            result[i, k + self.offsets] = 1.0
        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([repr(f) for f in self.__one_hot_featurizers])})"

    def position_meaning(self, k: int) -> tuple[str, Any]:
        agg = 0
        for f in self.__one_hot_featurizers:
            if agg <= k < agg + f.size:
                return f.position_meaning(k - agg)
            agg += f.size
        return None


class ConcatenatedFeaturizer(AtomFeaturizer):

    def __init__(self, featurizers):
        self.featurizers = featurizers

    @property
    def size(self) -> int:
        return sum(f.size for f in self.featurizers)

    def compute(self, mol):
        feats = [f.compute(mol) for f in self.featurizers]
        feats = np.concatenate(feats, axis=1)
        return feats

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([repr(f) for f in self.featurizers])})"

    def position_meaning(self, k: int) -> tuple[str, Any]:
        agg = 0
        for f in self.featurizers:
            if agg <= k < agg + f.size:
                return f.position_meaning(k - agg)
            agg += f.size
        return None


class RDKitFeatures(AtomFeaturizer):

    def __init__(self):
        fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        self.factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    @property
    def size(self):
        return 3

    def compute(self, mol):
        # Identify hydrogen bond acceptors, donors, and hydrophobic atoms
        features = self.factory.GetFeaturesForMol(mol)
        acceptor_atoms = [
            feat.GetAtomIds() for feat in features if feat.GetFamily() == "Acceptor"
        ]
        donor_atoms = [
            feat.GetAtomIds() for feat in features if feat.GetFamily() == "Donor"
        ]
        hydrophobic_atoms = [
            feat.GetAtomIds() for feat in features if feat.GetFamily() == "Hydrophobe"
        ]

        # Flatten the lists for easier access
        acceptor_atom_ids = [atom_id for ids in acceptor_atoms for atom_id in ids]
        donor_atom_ids = [atom_id for ids in donor_atoms for atom_id in ids]
        hydrophobic_atom_ids = [atom_id for ids in hydrophobic_atoms for atom_id in ids]

        feature_tensor = np.zeros((mol.GetNumAtoms(), self.size))
        for atom_id in acceptor_atom_ids:
            feature_tensor[atom_id, 0] = 1
        for atom_id in donor_atom_ids:
            feature_tensor[atom_id, 1] = 1
        for atom_id in hydrophobic_atom_ids:
            feature_tensor[atom_id, 2] = 1

        return feature_tensor

    def position_meaning(self, k: int) -> tuple[str, Any]:
        if k == 0:
            return "RDKitFeatures", "Acceptor"
        if k == 1:
            return "RDKitFeatures", "Donor"
        if k == 2:
            return "RDKitFeatures", "Hydrophobe"


FormalCharge = OneHotFeaturizer(
    [-2, -1, 0, 1, 2], rdchem.Atom.GetFormalCharge, name="FormalCharge"
)
NumHydrogens = OneHotFeaturizer(
    [0, 1, 2, 3, 4], rdchem.Atom.GetTotalNumHs, name="NumHydrogens"
)
IsAromatic = OneHotFeaturizer(
    [False, True], rdchem.Atom.GetIsAromatic, name="IsAromatic"
)

AtomFeatures = ComposeOneHot([NumHydrogens, IsAromatic])
AtomFeatures = ConcatenatedFeaturizer([AtomFeatures, RDKitFeatures()])
