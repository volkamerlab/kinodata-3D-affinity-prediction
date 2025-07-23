from typing import Union, Iterable, Generator
import logging
import os
from rdkit import Chem

logger = logging.getLogger(__name__)


def read_sdf_molecules(
    sdf_input: Union[str, Iterable[str]],
) -> Generator[Chem.Mol, None, None]:
    """
    Reads RDKit molecule objects from a single multi-SDF file or an iterable of SDF file paths.

    Args:
        sdf_input (Union[str, Iterable[str]]): A single SDF file path or an iterable of SDF file paths.

    Yields:
        Chem.Mol: Molecule objects read from the SDF file(s).
    """
    # Convert single string input into an iterable for uniform processing
    if isinstance(sdf_input, str):
        sdf_files = [sdf_input]
    else:
        sdf_files = sdf_input

    for file_path in sdf_files:
        number_unread = 0
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"SDF file not found: {file_path}")
        supplier = Chem.SDMolSupplier(file_path)
        for mol in supplier:
            if mol is None:
                number_unread += 1
                continue
            yield mol
        fn_log = logger.info if number_unread == 0 else logger.warning
        fn_log(f"Finished reading {file_path} with {number_unread} unread molecules.")
