import copy
import gzip
import json
import logging
from functools import cached_property
from pathlib import Path
import shutil
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from tqdm import tqdm

Callback = pl.Callback
Trainer = pl.Trainer
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from kinodata.data import KinodataDocked
from kinodata.transform.mask_residues import MaskResidues
import kinodata.transform as T

from ..model.regression import RegressionModel, cat_many

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
    """Custom prediction function mimicking `trainer.predict` but safe during `fit()`."""
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
    if device is None:
        device = next(_model.parameters()).device
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
    if trainer is None:
        trainer = model.trainer
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


class CrocodocCallback(Callback):
    def __init__(
        self,
        datasets: dict[str, KinodataDocked],
        mask_type: str,
        frequency: int,
        start_epoch: int = 0,
        outdir: str | Path | None = None,
        accelarator: str = "cpu",
    ):
        super().__init__()
        self.datasets = datasets
        self.mask_type = mask_type
        self.frequency = frequency
        self.start_epoch = start_epoch
        self._outdir = outdir

    @cached_property
    def outdir(self) -> Path:
        assert hasattr(self, "_trainer"), "CrocodocCallback must be setup before use"
        if self._outdir is None:
            return Path(self._trainer.log_dir) / "crocodoc"
        return Path(self._outdir)

    @property
    def reference_pred_file(self) -> Path:
        return self.outdir / "crocodoc_reference.csv"

    @property
    def masked_pred_file(self) -> Path:
        return self.outdir / "crocodoc_masked.csv"

    def _write_config(self):
        config = {
            "mask_type": self.mask_type,
            "frequency": self.frequency,
            "start_epoch": self.start_epoch,
        }
        with open(self.outdir / "crocodoc_config.json", "w") as f:
            json.dump(config, f, indent=4)

    def on_train_start(self, trainer, pl_module):
        self._trainer = trainer
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
        logger.info("CrocodocCallback outdir: %s", self.outdir)
        self._write_config()
        return

    def _store_predictions(self, masked_prediction, reference_prediction):
        logger.info("Storing predictions")
        if not self.masked_pred_file.exists():
            logger.info("Creating new prediction files")
            masked_prediction.to_csv(
                self.masked_pred_file, index=False, mode="w", header=True
            )
            reference_prediction.to_csv(
                self.reference_pred_file, index=False, mode="w", header=True
            )
            return
        logger.info("Appending to exisiting prediction files")
        masked_prediction.to_csv(
            self.masked_pred_file, index=False, mode="a", header=False
        )
        reference_prediction.to_csv(
            self.reference_pred_file, index=False, mode="a", header=False
        )

    def _crocodoc(self, epoch, pl_module):
        for dataset_key, dataset in self.datasets.items():
            logger.info(f"Running Crocodoc on {dataset_key} dataset")
            dataset, orig_transform = remove_augementation_transforms_from_dataset(
                dataset, return_orig_transform=True
            )
            masked_prediction, reference_prediction = crocodoc_cgnn(
                pl_module,
                dataset,
                trainer=None,
                ckpt_path=None,  # use the current model
                mask_type=self.mask_type,
            )
            masked_prediction["dataset"] = dataset_key
            reference_prediction["dataset"] = dataset_key
            masked_prediction["epoch"] = epoch
            reference_prediction["epoch"] = epoch
            self._store_predictions(
                masked_prediction,
                reference_prediction,
            )
            dataset.transform = orig_transform

    def _compress_results(self):
        logger.info("Compressing crocodoc results")
        with open(self.reference_pred_file, "rb") as f:
            with gzip.open(str(self.reference_pred_file) + ".gz", "wb") as g:
                shutil.copyfileobj(f, g)
        with open(self.masked_pred_file, "rb") as f:
            with gzip.open(str(self.masked_pred_file) + ".gz", "wb") as g:
                shutil.copyfileobj(f, g)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            return
        if trainer.current_epoch % self.frequency != 0:
            return
        self._crocodoc(trainer.current_epoch, pl_module)
        return
