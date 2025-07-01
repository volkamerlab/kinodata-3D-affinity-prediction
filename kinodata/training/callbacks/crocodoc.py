import copy
import gzip
import json
import logging
import shutil
import tarfile
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm

Callback = pl.Callback
Trainer = pl.Trainer
from scipy.spatial.distance import cosine
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

import kinodata.transform as T
from kinodata.data import KinodataDocked
from kinodata.transform.mask_residues import MaskResidues
from kinodata.types import COLS

from ...model.regression import RegressionModel, cat_many

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


def load_pli_reference(pli_reference_path: str | None) -> pd.DataFrame | None:
    if pli_reference_path is None:
        return None
    if not Path(pli_reference_path).exists():
        raise ValueError(f"PLI reference file {pli_reference_path} does not exist.")
    logger.info("Loading PLI reference file %s", pli_reference_path)
    pli_reference = pd.read_csv(pli_reference_path)
    if COLS.RESIDUE_IMPORTANCE not in pli_reference.columns:
        raise ValueError(
            f"PLI reference file must contain '{COLS.RESIDUE_IMPORTANCE}' column."
        )
    if COLS.ACTIVITY_ID not in pli_reference.columns:
        raise ValueError(
            f"PLI reference file must contain '{COLS.ACTIVITY_ID}' column."
        )
    return pli_reference


def cosine_similarity(
    group_data,
    x=None,
    y=None,
):
    assert None not in (x, y)
    assert x != y
    x = group_data[x]
    y = group_data[y]
    return 1 - cosine(x, y)


def compute_pli_alignment(
    model_delta: pd.DataFrame,
    pli_reference: pd.DataFrame,
    col_index: str = COLS.ACTIVITY_ID,
    col_delta: str = "delta",
    col_reference_attribution: str = "residue_importance",
    col_similarity: str = "cosine_similarity",
) -> pd.DataFrame:
    """
    Compute the cosine similarity between the model delta and the PLI reference.
    This function merges the model delta and PLI reference dataframes on the specified index column,
    and then computes the cosine similarity between the specified delta and reference attribution columns.
    The result is a dataframe with the index column and the cosine similarity values.

    Args:
        model_delta (pd.DataFrame): DataFrame containing the model's delta values.
        pli_reference (pd.DataFrame): DataFrame containing the PLI reference values.
        col_index (str, optional): Column name used to merge the dataframes. Defaults to "activities.activity_id".
        col_delta (str, optional): Column name in `model_delta` representing the delta values. Defaults to "delta".
        col_reference_attribution (str, optional): Column name in `pli_reference` representing the reference attribution values. Defaults to "residue_importance".

    Returns:
        pd.DataFrame: DataFrame containing the index column and the computed cosine similarity values.
    """
    merged_delta = model_delta.merge(pli_reference)
    return (
        merged_delta.groupby(col_index)
        .apply(partial(cosine_similarity, x=col_delta, y=col_reference_attribution))
        .reset_index(name=col_similarity, drop=False)
    )


def load_model_from_trainer_checkpoint(
    trainer,
    model_class: type,
    ckpt_path: str = "best",  # or "last" or a full path
    **model_kwargs,
) -> pl.LightningModule:
    if ckpt_path == "best":
        ckpt_file = trainer.checkpoint_callback.best_model_path
    elif ckpt_path == "last":
        ckpt_file = trainer._checkpoint_connector.last_checkpoint_path
    else:
        ckpt_file = ckpt_path  # assume it's a real path

    if not ckpt_file:
        raise ValueError(f"No checkpoint found for ckpt_path='{ckpt_path}'.")

    # Now load the model
    return model_class.load_from_checkpoint(ckpt_file, **model_kwargs)


def safe_predict(
    model: pl.LightningModule,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    ckpt_path: Optional[str] = None,  # "best" or "last" or a full path
) -> list[Any]:
    """Custom prediction function mimicking `trainer.predict` but safe during `trainer.fit()`."""
    _model = model
    if ckpt_path is not None:
        if not hasattr(model, "trainer"):
            raise ValueError(
                "Model must be a PyTorch Lightning model with a trainer to load from checkpoint."
            )
        # Load the model from the checkpoint
        _model = load_model_from_trainer_checkpoint(
            model.trainer, model.__class__, ckpt_path=ckpt_path
        )
    model_was_training = _model.training
    _model.eval()
    model_device = next(_model.parameters()).device
    if device is None:
        device = model_device
    if device != model_device:
        logger.warning(
            "Model device (%s) does not match specified device (%s). "
            "Moving model to specified device.",
            model_device,
            device,
        )
        _model.to(device)
    predictions = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (tuple, list)):
                batch = [b.to(device) for b in batch]
            else:
                batch = batch.to(device)
            preds = _model.predict_step(batch, batch_idx)
            predictions.append(preds)

    if model_was_training:
        _model.train()

    return predictions


def remove_augementation_transforms_from_dataset(dataset, return_orig_transform=False):
    orig_transform = dataset.transform
    if isinstance(dataset.transform, Compose):
        dataset.transform = Compose(
            [
                t
                for t in dataset.transform.transforms
                if not isinstance(t, T.PerturbAtomPositions)
            ]
        )
    elif isinstance(dataset.transform, T.PerturbAtomPositions):
        dataset.transform = None
    if return_orig_transform:
        return (dataset, orig_transform)
    return dataset


def remove_augmentation_transforms_from_data_module(data_module):
    data_module.train_dataset = remove_augementation_transforms_from_dataset(
        data_module.train_dataset
    )
    data_module.val_dataset = remove_augementation_transforms_from_dataset(
        data_module.val_dataset
    )
    data_module.test_dataset = remove_augementation_transforms_from_dataset(
        data_module.test_dataset
    )
    return data_module


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


def compress_directory_to_tar_gz(
    directory: Union[str, Path], output_filename: Union[str, Path] = None
):
    """
    Compresses the given directory into a .tar.gz file.

    Parameters:
        directory (str or Path): The path to the directory to compress.
        output_filename (str or Path, optional): The path of the output .tar.gz file.
            If not provided, uses the directory name with a .tar.gz extension.

    Returns:
        Path: The path to the created .tar.gz file.
    """
    directory = Path(directory).resolve()

    if not directory.is_dir():
        raise ValueError(f"{directory} is not a valid directory.")

    if output_filename is None:
        output_filename = directory.with_suffix(".tar.gz")
    else:
        output_filename = Path(output_filename).resolve()

    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(directory, arcname=directory.name)

    return output_filename


def run_crocodoc(
    model: RegressionModel,
    dataset: KinodataDocked,
    ckpt_path: str | None = None,
    mask_type: str | None = None,
    batch_size: int = 32,
    device: torch.device | str = "cuda:0",
) -> pd.DataFrame:
    data_list, residue_to_atom_index = get_required_data(dataset)

    print("Preparing residue masking transform...")
    masking = MaskResidues(residue_to_atom_index, mask_type=mask_type)

    dfs = []
    progress_bar = tqdm(total=len(masking), desc="Masked prediction")
    while True:
        increment = len(masking)
        pre_filter = [data for data in data_list if masking.filter(data)]
        transformed_data_list = [masking(copy.copy(data)) for data in pre_filter]
        transformed_data_list = transformed_data_list
        predictions = safe_predict(
            model,
            DataLoader(
                transformed_data_list,
                batch_size=batch_size,
                shuffle=False,
            ),
            ckpt_path=ckpt_path,
            device=device,
        )
        predictions = cat_many(predictions)
        meta = cat_many(
            [
                {
                    "ident": data["ident"],
                    COLS.ACTIVITY_ID: data["chembl_activity_id"],
                    COLS.KLIFS_ID: data["klifs_structure_id"],
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
                COLS.ACTIVITY_ID: meta[COLS.ACTIVITY_ID].cpu().numpy(),
                COLS.KLIFS_ID: meta[COLS.KLIFS_ID].cpu().numpy(),
                "masked_residue": meta["masked_residue"].cpu().numpy(),
                COLS.MASKED_PREDICTION: predictions["pred"].cpu().numpy(),
                "masked_resname": masked_resname,
                "masked_res_letter": masked_res_letter,
            }
        )
        dfs.append(df)
        increment = increment - len(masking)
        progress_bar.update(increment)
        if len(masking) == 0:
            break

    masked_pred_df = pd.concat(dfs)
    reference_predictions = cat_many(
        safe_predict(
            model,
            DataLoader(data_list, batch_size=32, shuffle=False),
            ckpt_path=ckpt_path,
            device=device,
        )
    )
    meta = cat_many(
        [
            {
                "ident": data["ident"],
                COLS.ACTIVITY_ID: data["chembl_activity_id"],
                COLS.KLIFS_ID: data["klifs_structure_id"],
            }
            for data in data_list
        ]
    )
    reference_df = pd.DataFrame(
        {
            "ident": meta["ident"].cpu().numpy(),
            COLS.ACTIVITY_ID: meta[COLS.ACTIVITY_ID].cpu().numpy(),
            COLS.KLIFS_ID: meta[COLS.KLIFS_ID].cpu().numpy(),
            COLS.REFERENCE_PREDICTION: reference_predictions["pred"].cpu().numpy(),
        }
    )

    return masked_pred_df, reference_df


def _get_run_id(trainer: Trainer) -> str:
    try:
        return str(trainer.logger.experiment.id)
    except AttributeError:
        return str(trainer.logger.run_id)


class CrocodocCallback(Callback):
    def __init__(
        self,
        datasets: dict[str, KinodataDocked],
        mask_type: str,
        frequency: int,
        start_epoch: int = 0,
        outdir: str | Path | None = None,
        pli_reference: pd.DataFrame | None = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.mask_type = mask_type
        self.frequency = frequency
        self.start_epoch = start_epoch
        self._outdir = outdir
        self._pli_reference = pli_reference
        if self.frequency == 0:
            logger.warning(
                "CrocodocCallback frequency is set to 0, Crocodoc will only run once at the end of training."
            )

    def log(self, *args, **kwargs):
        self._pl_module.log(*args, **kwargs)

    @cached_property
    def outdir(self) -> Path:
        assert hasattr(self, "_trainer"), "CrocodocCallback must be setup before use"
        if self._outdir is None:
            return Path(self._trainer.log_dir) / "crocodoc" / _get_run_id(self._trainer)
        return Path(self._outdir)

    @property
    def reference_pred_file(self) -> Path:
        return self.outdir / "crocodoc_reference.csv"

    @property
    def masked_pred_file(self) -> Path:
        return self.outdir / "crocodoc_masked.csv"

    @property
    def pli_alignment_file(self) -> Path:
        return self.outdir / "pli_alignment.csv"

    def _store_data_frame(self, data_frame: pd.DataFrame, file_path: str | Path):
        logger.info("Storing dataframe to %s", str(file_path))
        file_path = Path(file_path)
        if not file_path.exists():
            logger.info("Creating new dataframe csv file")
            data_frame.to_csv(file_path, index=False, mode="w", header=True)
            return
        logger.info("Appending to exisiting csv file")
        data_frame.to_csv(file_path, index=False, mode="a", header=False)

    def _compress_file(self, file_path: str | Path, remove_uncompressed: bool = True):
        if not file_path.exists():
            logger.warning(
                "File %s does not exist, skipping compression", str(file_path)
            )
            return
        logger.info("Compressing file %s", str(file_path))
        with open(file_path, "rb") as f:
            with gzip.open(str(file_path) + ".gz", "wb") as g:
                shutil.copyfileobj(f, g)
        if remove_uncompressed:
            logger.info("Removing uncompressed file %s", str(file_path))
            file_path.unlink()

    def _handle_crocodoc_result(
        self,
        masked_prediction: pd.DataFrame,
        reference_prediction: pd.DataFrame,
        dataset_key: str,
        epoch: int,
        log: bool = True,
    ):
        assert COLS.REFERENCE_PREDICTION in reference_prediction.columns
        assert COLS.MASKED_PREDICTION in masked_prediction.columns
        masked_prediction["dataset"] = dataset_key
        reference_prediction["dataset"] = dataset_key
        masked_prediction["epoch"] = epoch
        reference_prediction["epoch"] = epoch
        self._store_data_frame(masked_prediction, self.masked_pred_file)
        self._store_data_frame(reference_prediction, self.reference_pred_file)
        if self._pli_reference is None:
            return
        reference_activity_ids = self._pli_reference[COLS.ACTIVITY_ID].unique()
        model_delta = masked_prediction[
            masked_prediction[COLS.ACTIVITY_ID].isin(reference_activity_ids)
        ].merge(reference_prediction, how="left")
        model_delta[COLS.DELTA] = (
            model_delta[COLS.REFERENCE_PREDICTION] - model_delta[COLS.MASKED_PREDICTION]
        )
        pli_alignment = compute_pli_alignment(
            model_delta,
            self._pli_reference,
            col_index=COLS.ACTIVITY_ID,
            col_delta=COLS.DELTA,
            col_reference_attribution=COLS.RESIDUE_IMPORTANCE,
            col_similarity=COLS.ALIGNMENT,
        )
        pli_alignment["dataset"] = dataset_key
        pli_alignment["epoch"] = epoch
        self._store_data_frame(
            pli_alignment,
            self.pli_alignment_file,
        )
        if log:
            self.log(
                f"{dataset_key}/pli_alignment_median",
                pli_alignment[COLS.ALIGNMENT].median(),
                on_epoch=True,
            )
            self.log(
                f"{dataset_key}/pli_alignment_q25",
                pli_alignment[COLS.ALIGNMENT].quantile(0.25),
                on_epoch=True,
            )
            self.log(
                f"{dataset_key}/pli_alignment_q75",
                pli_alignment[COLS.ALIGNMENT].quantile(0.75),
                on_epoch=True,
            )

    def _write_config(self):
        config = {
            "mask_type": self.mask_type,
            "frequency": self.frequency,
            "start_epoch": self.start_epoch,
        }
        with open(self.outdir / "crocodoc_config.json", "w") as f:
            json.dump(config, f, indent=4)

    def _crocodoc(self, epoch, pl_module, **kwargs):
        for dataset_key, dataset in self.datasets.items():
            logger.info(f"Running Crocodoc on {dataset_key} dataset")
            dataset, orig_transform = remove_augementation_transforms_from_dataset(
                dataset, return_orig_transform=True
            )
            masked_prediction, reference_prediction = run_crocodoc(
                pl_module,
                dataset,
                trainer=None,
                ckpt_path=None,  # use the current model
                mask_type=self.mask_type,
            )
            dataset.transform = orig_transform
            self._handle_crocodoc_result(
                masked_prediction, reference_prediction, dataset_key, epoch, **kwargs
            )

    def _compress_results(self):
        logger.info("Compressing crocodoc results")
        compress_directory_to_tar_gz(self.outdir)

    def on_train_start(self, trainer, pl_module):
        self._trainer = trainer
        self._pl_module = pl_module
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
        logger.info("CrocodocCallback outdir: %s", self.outdir)
        self._write_config()
        return

    def on_train_epoch_end(self, trainer, pl_module):
        if self.frequency == 0:
            return
        if trainer.current_epoch < self.start_epoch:
            return
        if trainer.current_epoch % self.frequency != 0:
            return
        self._crocodoc(trainer.current_epoch, pl_module)
        return

    def on_fit_end(self, trainer, pl_module):
        self._crocodoc(trainer.current_epoch, pl_module, log=False)
        self._compress_results()
        return
