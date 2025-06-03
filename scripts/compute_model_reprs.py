from collections import defaultdict
from kinodata.data.featurization.atoms import AtomFeatures
from kinodata.data.featurization.bonds import NUM_BOND_TYPES
from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
from typing import Callable

import torch
from kinodata.configuration import Config
import json
import wandb
from tqdm import tqdm
from kinodata.wandb_utils import (
    retrieve_best_model_artifact,
    run_by_id,
    load_state_dict,
    RunInfo,
)
from kinodata.model.complex_transformer import make_model as make_sparse_transformer
from kinodata.model.dimenet import DimeNetWrapper
from kinodata.model.regression import RegressionModel
from kinodata.training.representation import reps_to_dataframe
from kinodata.data.data_module import make_kinodata_module
from kinodata.data.dataset import apply_transform_instance_permament
from kinodata.transform import TransformToComplexGraph
from kinodata.transform.feature_mask import FeatureMask
from torch_geometric.transforms import Compose
from wandb.apis.public import Run
from torch_geometric.data.lightning import LightningDataset
import typer


@dataclass
class TrainingResult:
    training_config: Config
    run: Run

    @classmethod
    def from_finished_run(cls, run: Run):
        training_config = RunInfo(run).config
        return cls(training_config, run)

    def data_module(self, **kwargs) -> LightningDataset:
        one_time_transform = TransformToComplexGraph(
            remove_heterogeneous_representation=True
        )
        if self.training_config.get("simplified_features", False):
            print("Using simplified features")
            mask = torch.zeros(AtomFeatures.size, dtype=torch.bool)
            is_hydrogen = list(range(5))
            mask[is_hydrogen] = True  # num hydrogens 0-4
            mask[-1] = True  # gasteiger charge
            bond_mask = torch.zeros(NUM_BOND_TYPES, dtype=torch.bool)
            bond_mask[[0, 1, 2, -1]] = True  # single, double, triple, other
            select_transform = FeatureMask(mask, bond_mask)
            one_time_transform = Compose([one_time_transform, select_transform])
            self.training_config["atom_attr_size"] = mask.sum().item()
            self.training_config["edge_size"] = bond_mask.sum().item()
        dm = make_kinodata_module(
            self.training_config,
            one_time_transform=partial(
                apply_transform_instance_permament,
                transform=one_time_transform,
            ),
        )
        dm.kwargs.update(kwargs)
        return dm

    @property
    def model_cls(self) -> Callable[[Config], RegressionModel]:
        return make_sparse_transformer

    @property
    def trained_model(self) -> RegressionModel:
        best_model_artifact = retrieve_best_model_artifact(self.run)
        state_dict = load_state_dict(best_model_artifact)["state_dict"]
        model = self.model_cls(self.training_config)
        model.load_state_dict(state_dict)
        model.eval()
        return model


def main(
    run_id: str,
    output_path: Path,
    log_level: str = "INFO",
    dry_run: bool = False,
):
    """
    Main function to compute model representations and save them to the specified output path.

    Args:
        run_id (str): The wandb ID of the run to load the model from.
        output_path (Path): The path where the computed representations will be saved.
        log_level (str, optional): The logging level. Defaults to "INFO".

    Returns:
        None
    """
    tags = ["compute_model_reprs"]
    wandb.init(
        config={
            "source_run": run_id,
        },
        project="kinodata-docked-rescore",
        tags=tags,
    )
    logging.basicConfig(level=log_level)
    logging.info(f"Looking for run with id {run_id}..")
    run = run_by_id(run_id)
    logging.info(
        f"Found run '{run.name}' ({run.state}) created at {run.created_at}! Loading model..."
    )
    train_result = TrainingResult.from_finished_run(run)
    logging.info(f"Loaded model:\n{repr(train_result.trained_model)}")
    logging.info(f"Loaded training config:\n{json.dumps(train_result.training_config)}")
    data_module = train_result.data_module(num_workers=0, persistent_workers=False)
    if dry_run:
        logging.info("Dry run enabled. Exiting without computing representations.")
        return
    output_path = output_path / run_id
    if not output_path.exists():
        logging.info(f"Creating output directory {output_path}...")
        output_path.mkdir(parents=True)
    logging.info(f"Computing reprs and saving results to {output_path}...")
    model = train_result.trained_model
    if torch.cuda.is_available():
        logging.info("Moving model to GPU...")
        model = model.cuda()
    reps = defaultdict(list)
    metadata = defaultdict(list)
    with torch.inference_mode():
        for key, loader in {
            "train": data_module.train_dataloader(),
            "val": data_module.val_dataloader(),
            "test": data_module.test_dataloader(),
        }.items():
            logging.info(f"Computing representations for {key} set...")
            if len(loader.dataset) == 0:
                logging.warning(
                    f"No data in {key} set. Skipping representation computation."
                )
            for index, batch in tqdm(enumerate(loader), total=len(loader.dataset)):
                batch_size = batch.num_graphs
                metadata["activities.activity_id"].extend(
                    [int(aid) for aid in batch.chembl_activity_id]
                )
                metadata["split"].extend([key] * batch_size)
                batch = batch.to(model.device)
                _reps = model.compute_representation(batch)
                _reps = {k: v.cpu() for k, v in _reps.items()}
                for k, v in _reps.items():
                    reps[k].append(v)
    reps = {k: torch.cat(v, dim=0) for k, v in reps.items()}
    for k, _reps in reps.items():
        reps_to_dataframe(_reps, **metadata).to_csv(
            output_path / f"{k}.csv",
            index=False,
        )


if __name__ == "__main__":
    typer.run(main)
