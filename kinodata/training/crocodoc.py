import numpy as np
from ..model.regression import RegressionModel, cat_many
from kinodata.data import KinodataDocked
from kinodata.transform.mask_residues import MaskResidues
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
import pandas as pd
from torch_geometric.transforms import Compose
import torch
import logging
import copy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
forbidden_seq = set(
    [
        "KALGKGLFSMVIRITLKVVGLRILNLPHLILEYCKAKDIIRFLQQKNFLLLINWGIR",
        "LIGKGDSARLDYLVVGRLLQLVREP",
        "LIIGKGDFGKVELSALKVVDIIRLILDYLVVGRLLQLVRE",
        "NKMGEGGFGVVYKVAVKKLQFDQEIKVMAKCQENLVELLGFCLVYVYMPNGSLLDRLSCFLHENHHIHRDIKSANILLISDFGLA",
        "_ALNVLDMSQKLYLLSSLDPYLLEMYSYLILEAPEGEIFNLLRQYLHSAMIIYRDLKPHNVLFIAA",
    ]
)


def get_required_data(dataset):
    data_list = [data for data in dataset]
    data_list = [
        data for data in data_list if data.pocket_sequence not in forbidden_seq
    ]
    required_idents = [int(data["chembl_activity_id"]) for data in data_list]
    residue_to_atom_index = MaskResidues.load_residue_index(required_idents)
    del_list = []
    for k, v in residue_to_atom_index.items():
        if v is None:
            print(f"Removing ident {k} due to missing index")
            del_list.append(k)

    for k in del_list:
        del residue_to_atom_index[k]

    return data_list, residue_to_atom_index


def crocodoc_cgnn(
    model: RegressionModel,
    dataset: KinodataDocked,
    trainer: Trainer | None = None,
    ckpt_path: str | None = "best",
    mask_type: str | None = None,
) -> pd.DataFrame:
    data_list, residue_to_atom_index = get_required_data(dataset)

    print("Preparing residue masking transform...")
    masking = MaskResidues(residue_to_atom_index, mask_type=mask_type)

    dfs = []
    while True:
        logger.info(f"{len(masking)} masked complexes remaining")
        pre_filter = [data for data in data_list if masking.filter(data)]
        transformed_data_list = [masking(copy.copy(data)) for data in pre_filter]
        transformed_data_list = transformed_data_list
        predictions = trainer.predict(
            model,
            DataLoader(
                transformed_data_list,
                batch_size=32,
                shuffle=False,
            ),
            ckpt_path=ckpt_path,
        )
        predictions = cat_many(predictions)
        meta = cat_many(
            [
                {
                    "ident": data["ident"],
                    "chembl_activity_id": data["chembl_activity_id"],
                    "klifs_structure_id": data["klifs_structure_id"],
                    "masked_residue": data.masked_residue,
                }
                for data in transformed_data_list
            ]
        )
        masked_resname = [data.masked_resname for data in transformed_data_list]
        masked_res_letter = [data.masked_res_letter for data in transformed_data_list]
        df = pd.DataFrame(
            {
                "ident": meta["ident"].cpu().numpy(),
                "chembl_activity_id": meta["chembl_activity_id"].cpu().numpy(),
                "klifs_structure_id": meta["klifs_structure_id"].cpu().numpy(),
                "masked_residue": meta["masked_residue"].cpu().numpy(),
                "masked_pred": predictions["pred"].cpu().numpy(),
                "masked_resname": masked_resname,
                "masked_res_letter": masked_res_letter,
            }
        )
        dfs.append(df)
        if len(masking) == 0:
            break

    masked_pred_df = pd.concat(dfs)
    reference_predictions = cat_many(
        trainer.predict(
            model,
            DataLoader(data_list, batch_size=32, shuffle=False),
            ckpt_path=ckpt_path,
        )
    )
    meta = cat_many(
        [
            {
                "ident": data["ident"],
                "chembl_activity_id": data["chembl_activity_id"],
                "klifs_structure_id": data["klifs_structure_id"],
            }
            for data in data_list
        ]
    )
    reference_df = pd.DataFrame(
        {
            "ident": meta["ident"].cpu().numpy(),
            "chembl_activity_id": meta["chembl_activity_id"].cpu().numpy(),
            "klifs_structure_id": meta["klifs_structure_id"].cpu().numpy(),
            "reference_pred": reference_predictions["pred"].cpu().numpy(),
        }
    )

    return masked_pred_df, reference_df
