from collections import defaultdict
import logging
from functools import cached_property
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import Callback
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from kinodata.data import KinodataDocked
from kinodata.model.regression import RegressionModel

from .crocodoc import (
    _get_run_id,
    compress_directory_to_tar_gz,
    remove_augementation_transforms_from_dataset,
)

logger = logging.getLogger(__name__)


def reps_to_dataframe(representation: torch.Tensor, **metadata):
    """
    Converts a batch of representations and metadata to a long-format DataFrame.

    Args:
        reps (torch.Tensor): Shape (batch_size, hidden_dim)
        metadata (kwargs): Each should be a list/array/1D tensor of shape (batch_size,)

    Returns:
        pd.DataFrame: DataFrame with metadata columns + rep_0, rep_1, ..., rep_{hidden_dim-1}
    """
    reps_np = representation.cpu().numpy()
    batch_size, dim = reps_np.shape

    # Flatten to DataFrame with columns rep_0, rep_1, ...
    reps_df = pd.DataFrame(reps_np, columns=[f"rep_{i}" for i in range(dim)])

    # Add metadata
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        reps_df[key] = value

    return reps_df


class StoreModelRepresentation(Callback):
    def __init__(
        self,
        datasets: dict[str, KinodataDocked],
        alias: str | None = None,
        outdir: str | Path | None = None,
        batch_size: int = 32,
    ):
        super().__init__()
        self._datasets = datasets
        self._repr_name = self.__class__.__name__
        self._alias = alias
        self._outdir = outdir
        self._batch_size = batch_size

    @cached_property
    def outdir(self) -> Path:
        assert hasattr(self, "_trainer"), f"{self._repr_name} must be setup before use"
        if self._outdir is None:
            return (
                Path(self._trainer.log_dir)
                / "model_representations"
                / _get_run_id(self._trainer)
            )

    def representation_matrix_file(self, representation_type: str) -> Path:
        return self.outdir / f"{representation_type.lower().replace(' ', '_')}.csv"

    def setup(self, trainer, pl_module, stage):
        if stage in ("validate", "test"):
            return
        self._trainer = trainer
        if self._alias is not None:
            assert self._trainer.checkpoint_callback is not None, (
                "Checkpoint callback must be set up to use alias"
            )
        if not self.outdir.exists():
            logger.info("Creating output directory for model representations")
            self.outdir.mkdir(parents=True, exist_ok=True)

    def _store_data_frame(self, data_frame: pd.DataFrame, file_path: str | Path):
        logger.info("Storing dataframe to %s", str(file_path))
        file_path = Path(file_path)
        if not file_path.exists():
            logger.info("Creating new dataframe csv file")
            data_frame.to_csv(file_path, index=False, mode="w", header=True)
            return
        logger.info("Appending to exisiting csv file")
        data_frame.to_csv(file_path, index=False, mode="a", header=False)

    def _get_model(self, pl_module):
        ckpt_path = None
        model = pl_module
        model_cls = pl_module.__class__
        if self._alias == "best":
            ckpt_path = getattr(
                self._trainer.checkpoint_callback, "best_model_path", None
            )
            assert ckpt_path is not None, (
                "Best model path not found in checkpoint callback"
            )
            logger.info("Loading model from best checkpoint: %s", ckpt_path)
            raise NotImplementedError
        if self._alias == "last":
            ckpt_path = getattr(
                self._trainer.checkpoint_callback, "last_model_path", None
            )
            assert ckpt_path is not None, (
                "Last model path not found in checkpoint callback"
            )
            logger.info("Loading model from last checkpoint: %s", ckpt_path)
            raise NotImplementedError

        assert isinstance(model, RegressionModel), (
            f"Expected model to be a RegressionModel, got {model_cls.__name__}"
        )

        return model

    @torch.no_grad()
    def _compute_representation(self, pl_module, **kwargs):
        model = self._get_model(pl_module)
        model = model.eval()
        for data_key, dataset in self._datasets.items():
            loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                shuffle=False,
            )
            data_frames = defaultdict(list)
            for batch in tqdm(loader, desc=f"Computing representations for {data_key}"):
                batch = batch.to(model.device)
                representation = model.compute_representation(batch)
                activity_id = batch.chembl_activity_id
                for repr_key, reps in representation.items():
                    _df = reps_to_dataframe(
                        reps,
                        activity_id=activity_id,
                    )
                    _df["representation_type"] = repr_key
                    data_frames[repr_key].append(_df)

            for repr_key, df_list in data_frames.items():
                if not df_list:
                    logger.warning(
                        f"No representations computed for {repr_key} in {data_key}"
                    )
                    continue
                df = pd.concat(df_list, axis=0, ignore_index=True)
                df["split"] = data_key
                self._store_data_frame(df, self.representation_matrix_file(repr_key))

    def _compress_results(self):
        logger.info(f"Compressing {self._repr_name} results")
        compress_directory_to_tar_gz(self.outdir)

    def on_fit_end(self, trainer, pl_module):
        orig_transforms = {}
        for data_key, dataset in self._datasets.items():
            _, transform = remove_augementation_transforms_from_dataset(
                dataset, return_orig_transform=True
            )
            orig_transforms[data_key] = transform
        try:
            self._compute_representation(pl_module)
        except Exception as e:
            # Restore original transforms
            for data_key, dataset in self._datasets.items():
                dataset.transform = orig_transforms[data_key]
            raise e
        self._compress_results()
        return
