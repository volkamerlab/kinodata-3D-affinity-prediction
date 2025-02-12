from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from train_voxel_model import get_data_split_activity_ids, DATA_DIR, VoxelModel
from kinodata.data.voxel.dataset import get_kinodata3d_df, default_voxel
from kinodata.data.voxel.lazy_iterable_dataset import IterableVoxelDataset

from torch.utils.data import DataLoader
from resmo.generate_modified_data import generate_modified_data
from resmo.protein_model import Mol2ProteinModel
from resmo.modification import MaskResidue
import wandb
import typer


def main(
    fold: int = 0,
    model_name: str = "",
    reference_only: bool = False,
):
    run = wandb.init(project="kinodata-voxel")
    artifact = run.use_artifact(
        f"nextaids/kinodata-voxel/model-{model_name}:best", type="model"
    )
    artifact_dir = artifact.download()
    artifact_dir = Path(artifact_dir)
    _, _, test_split = get_data_split_activity_ids(
        "scaffold-k-fold",
        0,
        fold,
    )
    df = get_kinodata3d_df(DATA_DIR)
    df = df.loc[test_split]
    clean_protein_files = df["pocket_file"].values
    ligand_files = df["ligand_file"].values
    modification = MaskResidue()

    def generate_data():
        for ligand_file, clean_protein_file in zip(ligand_files, clean_protein_files):
            pocket_model = Mol2ProteinModel.from_file(clean_protein_file)
            for modified_pocket in modification.apply_all(pocket_model):
                signature = modified_pocket.modification_signature
                modified_protein_file = modified_pocket.temp_file_with_protein_data(
                    ".mol2"
                )
                yield modified_protein_file, ligand_file, signature.model_dump()

    def generate_reference_data():
        for ligand_file, clean_protein_file in zip(ligand_files, clean_protein_files):
            metadata = {
                "ligand_file": ligand_file,
                "source_file": clean_protein_file,
            }
            yield clean_protein_file, ligand_file, metadata

    dataset = IterableVoxelDataset(generate_data(), default_voxel)
    reference_dataset = IterableVoxelDataset(generate_reference_data(), default_voxel)

    loader = DataLoader(dataset, batch_size=64)
    reference_loader = DataLoader(reference_dataset, batch_size=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device", device)
    model = VoxelModel.load_from_checkpoint(artifact_dir / "model.ckpt")
    model = model.to(device)

    print("Generating masked predictions")
    df_reference = defaultdict(list)
    with torch.inference_mode():
        for batch in tqdm(reference_loader):
            metadata = batch[-1]
            inp = batch[0].to(device)
            pred = model(inp)

            df_reference["reference_pred"].extend(list(pred.cpu().numpy()))
            df_reference["ligand_file"].extend(metadata["ligand_file"])
            df_reference["protein_file"].extend(metadata["source_file"])

    df_reference = pd.DataFrame(df_reference)
    print(df_reference.head())
    df_reference.to_csv(
        f"voxel_crocodoc_reference_{fold}_{model_name}.csv", index=False
    )

    if not reference_only:
        print("Generating masked predictions")
        df = defaultdict(list)
        with torch.inference_mode():
            for batch in tqdm(loader):
                metadata = batch[-1]
                inp = batch[0].to(device)
                pred = model(inp)

                df["pred"].extend(list(pred.cpu().numpy()))
                df["ligand_file"].extend(metadata["ligand_file"])
                df["protein_file"].extend(metadata["source_file"])
                df["masked_residue_index"].extend(
                    list(metadata["residue_index"].cpu().numpy())
                )
                df["masked_residue_name"].extend(metadata["residue_name"])

        df = pd.DataFrame(df)
        print(df.head())
        df.to_csv(f"voxel_crocodoc_{fold}_{model_name}.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
