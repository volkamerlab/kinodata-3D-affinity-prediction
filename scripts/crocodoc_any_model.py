import pandas as pd
import torch
import typer
import logging
from typer import Option
from pathlib import Path
from typing import Literal, Optional
import json
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
import wandb

from kinodata.training.callbacks.crocodoc import run_crocodoc
import kinodata.wandb_utils as wb
from kinodata.data import KinodataDocked, Filtered
from kinodata.configuration import Config
from kinodata.data.data_module import create_dataset
from kinodata.data.grouped_split import KinodataKFoldSplit
from kinodata.transform.filter_metadata import FilterDockingRMSD
from kinodata.transform import TransformToComplexGraph
from kinodata.evaluation.predict import predict_df

from enum import Enum

logger = logging.getLogger(__name__)


class SplitType(str, Enum):
    SCAFFOLD_K_FOLD = "scaffold-k-fold"
    POCKET_K_FOLD = "pocket-k-fold"
    RANDOM_K_FOLD = "random-k-fold"

    def __str__(self):
        return self.value


class ModelType(str, Enum):
    CGNN = "cgnn"
    CGNN3D = "cgnn3d"
    DIMENET = "dimenet"


def _append_dataframe_to(data_frame: pd.DataFrame, file_path: Path) -> None:
    logger.info("Storing dataframe to %s", str(file_path))
    file_path = Path(file_path)
    if not file_path.exists():
        logger.info("Creating new dataframe csv file")
        data_frame.to_csv(file_path, index=False, mode="w", header=True)
        return
    logger.info("Appending to exisiting csv file")
    data_frame.to_csv(file_path, index=False, mode="a", header=False)


def get_dataset_information(
    split_type: SplitType | None = None,
    split_fold: int | None = None,
    rmsd_threshold: float | None = 2.0,
    model_train_config: Optional[Config] = None,
) -> tuple[SplitType, int, float]:
    split_type = split_type or model_train_config["split_type"]
    split_fold = split_fold or model_train_config["split_index"]
    rmsd_threshold = rmsd_threshold or model_train_config["filter_rmsd_max_value"]
    return split_type, split_fold, rmsd_threshold


def get_inference_datasets(
    split_type: SplitType,
    split_fold: int,
    rmsd_threshold: float,
    train: bool = False,
    val: bool = False,
    test: bool = True,
) -> dict[str, KinodataDocked]:
    rmsd_filter = FilterDockingRMSD(maximum_rmsd=rmsd_threshold)
    filtered_dataset_cls = Filtered(
        KinodataDocked(),
        rmsd_filter,
    )
    dataset = filtered_dataset_cls()
    dataset.transform = TransformToComplexGraph(
        remove_heterogeneous_representation=True
    )
    splitter = KinodataKFoldSplit(str(split_type), 5)
    split = splitter.split(dataset)[split_fold]
    ret = {}
    if train:
        ret["train"] = dataset[split.train_split]
    if val:
        ret["val"] = dataset[split.val_split]
    if test:
        ret["test"] = dataset[split.test_split]
    return ret


def get_state_dict(
    model_checkpoint: Optional[Path] = None,
    wandb_run_id: Optional[str] = None,
    device: str = "cpu",
) -> dict:
    if model_checkpoint is not None:
        logging.info(f"Loading model checkpoint from {model_checkpoint}")
        return torch.load(model_checkpoint, map_location=device)
    elif wandb_run_id is not None:
        logging.info(f"Loading model from Weights & Biases run ID: {wandb_run_id}")
    else:
        raise ValueError("Either model_checkpoint or wandb_run_id must be provided.")


def get_model_class(
    model_type: ModelType,
) -> type:
    if model_type == ModelType.CGNN:
        from kinodata.model.complex_transformer import make_model

        return make_model
    elif model_type == ModelType.CGNN3D:
        from kinodata.model.complex_transformer import make_model

        return make_model
    elif model_type == ModelType.DIMENET:
        from kinodata.model.dimenet import DimeNetWrapper

        return DimeNetWrapper
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_train_config(
    model_train_config_path: Optional[Path] = None,
    wandb_run_id: Optional[str] = None,
) -> Config:
    if model_train_config_path is not None:
        if model_train_config_path.name.endswith(".wandb.json"):
            json_text = model_train_config_path.read_text()
            return wb.load_wandb_config(json_text)
        elif model_train_config_path.suffix == ".json":
            logging.info(
                f"Loading model training config from {model_train_config_path}"
            )
            return Config(json.loads(model_train_config_path.read_text()))
    elif wandb_run_id is not None:
        train_run = wb.run_by_id(wandb_run_id)
        return wb.RunInfo(train_run).config
    raise ValueError("Either model_train_config_path or wandb_run_id must be provided.")


def main(
    model_checkpoint_path: Optional[Path] = Option(
        None,
        help="Path to the model checkpoint .pt file",
    ),
    wandb_run_id: Optional[str] = Option(
        None,
        help="Training run ID",
    ),
    model_type: ModelType = Option(
        ModelType.CGNN3D,
        help="Type of model to use for inference",
    ),
    model_train_config_path: Optional[Path] = Option(
        None,
        help="Path to the model training configuration file",
    ),
    wandb_model_alias: Optional[str] = Option(
        "best",
        help="Alias of the model in Weights & Biases",
    ),
    split_type: SplitType | None = typer.Option(
        SplitType.SCAFFOLD_K_FOLD,
        help="Type of split to use for splitting the dataset",
    ),
    split_fold: int = typer.Option(
        0,
        help="Fold number to use for the split",
    ),
    rmsd_threshold: float = typer.Option(
        2.0,
        help="RMSD threshold for filtering the dataset",
    ),
    on_train: bool = typer.Option(
        False,
        help="Run crocodoc inference on the training split",
    ),
    on_val: bool = typer.Option(
        False,
        help="Run crocodoc inference on the validation split",
    ),
    on_test: bool = typer.Option(
        True,
        help="Run crocodoc inference on the test split",
    ),
    batch_size: int = typer.Option(
        48,
        help="Batch size for inference",
    ),
    device: str = typer.Option(
        "cuda:0",
        help="Device to run the inference on (e.g., 'cuda:0' or 'cpu')",
    ),
    outfile: Optional[Path] = Option(
        None,
        help="Output file to save the results",
    ),
    predict_outfile: Optional[Path] = Option(
        None,
        help="Output file to save the predictions",
    ),
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA is not available, switching to CPU.")
        device = "cpu"
    match (outfile, wandb_run_id):
        case (None, None):
            outfile = Path("crocodoc_results.csv")
        case (None, _):
            outfile = Path(f"crocodoc_results_{wandb_run_id}.csv")
        case (_, None):
            pass

    model_train_config = get_train_config(
        model_train_config_path=model_train_config_path,
        wandb_run_id=wandb_run_id,
    )
    model_cls = get_model_class(model_type)
    state_dict = get_state_dict(
        model_checkpoint=model_checkpoint_path,
        wandb_run_id=wandb_run_id,
        device=device,
    )
    if "state_dict" in state_dict:
        # If the state_dict is wrapped in a 'state_dict' key, extract it
        state_dict = state_dict["state_dict"]
    model: torch.nn.Module = model_cls(model_train_config)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    split_type, split_fold, rmsd_threshold = get_dataset_information(
        split_type=split_type,
        split_fold=split_fold,
        rmsd_threshold=rmsd_threshold,
        model_train_config=model_train_config,
    )

    datasets = get_inference_datasets(
        split_type=split_type,
        split_fold=split_fold,
        rmsd_threshold=rmsd_threshold,
        train=on_train,
        val=on_val,
        test=on_test,
    )
    if predict_outfile is not None:
        predict_outfile = Path(predict_outfile)
        if not predict_outfile.parent.exists():
            logger.info(
                f"Creating directory for predictions: {predict_outfile.parent.absolute()}"
            )
            predict_outfile.parent.mkdir(parents=True, exist_ok=True)
        for key, dataset in datasets.items():
            logger.info(f"Running predictions on {key} split")
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            predictions = predict_df(
                model=model,
                loader=loader,
                trainer=Trainer(accelerator="cpu"),
                ckpt_path=None,
            )
            predictions["dataset"] = key
            _append_dataframe_to(
                data_frame=predictions,
                file_path=predict_outfile,
            )

    for dataset_key, dataset in datasets.items():
        logger.info(
            f"Running crocodoc on\n"
            f"  model_type = {model_type}\n"
            f"  rmsd_threshold = {rmsd_threshold}\n"
            f"  split_type = {split_type}\n"
            f"  cv_fold = {split_fold}\n"
            f"  split = {dataset_key}"
        )
        results = run_crocodoc(
            model=model,
            dataset=dataset,
            ckpt_path=None,
            mask_type="atom_objects",
            batch_size=batch_size,
            device=device,
        )
        results["dataset"] = dataset_key
        _append_dataframe_to(
            data_frame=results,
            file_path=outfile,
        )


if __name__ == "__main__":
    typer.run(main)
