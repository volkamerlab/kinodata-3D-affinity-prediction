from dataclasses import dataclass
import requests as req
from pathlib import Path
from rdkit.Chem.rdmolfiles import MolFromMol2File

from .dataset import ComplexInformation, _DATA


@dataclass
class KlifsMOL2Artifact:
    klifs_id: int
    endpoint: str
    data_dir: Path
    cached_file_format: str = "{endpoint}_{klifs_id}.mol2"
    download: bool = False
    force: bool = False

    def __post_init__(self):
        if self.download:
            self.download()

    def query_klifs(self) -> bytes:
        resp = req.get(
            f"https://klifs.net/api/structure_get_{self.endpoint}?structure_ID={self.klifs_id}"
        )
        assert resp.ok
        return resp.content

    def download(self):
        if self.artifact.exists() and not self.force:
            return
        if not self.artifact.parent.exists():
            self.artifact.parent.mkdir(parents=True)
        with open(self.artifact, "wb") as f:
            f.write(self.query_klifs())

    @property
    def artifact(self) -> Path:
        return self.data_dir / self.cached_file_format.format(
            endpoint=self.endpoint, klifs_id=self.klifs_id
        )


@dataclass
class Kinodata3DPocketArtifact(KlifsMOL2Artifact):
    endpoint: str = "pocket"
    data_dir: Path = _DATA / "raw" / "mol2" / "pocket"
    cached_file_format: str = "{klifs_id}_{endpoint}.mol2"


@dataclass
class KlifsLigandArtifact(KlifsMOL2Artifact):
    endpoint: str = "ligand"
    data_dir: Path = _DATA / "raw" / "mol2" / "ligand"
    cached_file_format: str = "{klifs_id}_{endpoint}.mol2"


@dataclass
class KlifsComplexInformation(ComplexInformation):

    @classmethod
    def from_klifs_structure_id(
        cls: type["KlifsComplexInformation"], klifs_structure_id: int
    ):
        ligand = KlifsLigandArtifact(klifs_id=klifs_structure_id, download=True)
        pocket_mol2_file = Kinodata3DPocketArtifact(
            klifs_id=klifs_structure_id, download=True
        ).artifact

        # create rdkit ligand mol from mol2
        ligand_mol = MolFromMol2File(str(ligand.artifact))
