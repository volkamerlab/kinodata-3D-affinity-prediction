import gzip
import json
import logging
import shutil
from functools import cached_property
from pathlib import Path

import captum
import pandas as pd
import torch
from pytorch_lightning import Callback
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from kinodata.data import KinodataDocked
from kinodata.model.complex_transformer import ComplexTransformer
from kinodata.types import COLS, NodeType

from .crocodoc import (
    _get_run_id,
    compress_directory_to_tar_gz,
    remove_augementation_transforms_from_dataset,
)

logger = logging.getLogger(__name__)


class WrappedComplexTransformer:
    def __init__(self, model: ComplexTransformer):
        self.model = model

    def compute_initial_embeds(self, data: HeteroData):
        node_store = data[NodeType.Complex]
        node_repr = self.model.initial_embed_nodes(data)
        if (batch := node_store.get("batch", None)) is None:
            batch = torch.zeros(
                node_repr.size(0), dtype=torch.long, device=node_repr.device
            )
            node_store["batch"] = batch
        edge_index, edge_repr = self.model.initial_embed_edges(data)

        return node_repr, edge_repr, edge_index, node_store.batch

    def forward_initial_embeds(
        self,
        node_embed: Tensor,
        edge_embed: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ):
        node_embed = node_embed.squeeze()
        for sparse_attention_block, norm in zip(
            self.model.attention_blocks, self.model.norm_layers
        ):
            node_embed, edge_embed = sparse_attention_block(
                node_embed, edge_embed, edge_index
            )
            node_embed = norm(node_embed, batch)

        graph_repr = self.model.aggr(node_embed, batch)
        return self.model.out(graph_repr)


def compute_ig_attributions(
    model: WrappedComplexTransformer,
    loader: DataLoader,
    resolution: int = 30,
    handle_device: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if handle_device:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            print("CUDA is available, using GPU")
            device = torch.device("cuda")
        model = model.to(device)
    ig = captum.attr.IntegratedGradients(
        model.forward_initial_embeds, multiply_by_inputs=True
    )
    node_data_dict = {
        "attribution": [],
        "activity_id": [],
        "prediction": [],
        "delta": [],
        "node_index": [],
        "atomic_number": [],
        "is_pocket": [],
    }
    edge_data_dict = {
        "attribution": [],
        "activity_id": [],
        "prediction": [],
        "delta": [],
        "edge_index": [],
        "source_index": [],
        "target_index": [],
    }
    for data in tqdm(loader):
        if handle_device:
            data = data.to(device)
        with torch.no_grad():
            node_embed, edge_embed, edge_index, batch = model.compute_initial_embeds(
                data
            )
            node_embed.unsqueeze_(0)
            pred = model.forward_initial_embeds(
                node_embed, edge_embed, edge_index, batch
            )
        pred = pred.detach().cpu()
        delta = None

        (node_attr, edge_attr), delta = ig.attribute(
            inputs=(node_embed, edge_embed),
            additional_forward_args=(edge_index, batch),
            return_convergence_delta=True,
            internal_batch_size=1,
            n_steps=resolution,
        )
        node_attr = node_attr.sum(dim=-1).detach().cpu().squeeze()
        edge_attr = edge_attr.sum(dim=-1).detach().cpu().squeeze()
        delta = delta.detach().cpu() if delta is not None else None
        activity_id = data.chembl_activity_id
        if isinstance(activity_id, Tensor):
            activity_id = int(activity_id.detach().cpu().item())
        if isinstance(activity_id, list):
            activity_id = int(activity_id[0])
        else:
            activity_id = int(activity_id)

        # Node attributions
        node_data = data[NodeType.Complex]
        node_data_dict["attribution"].extend(node_attr.tolist())
        node_data_dict["activity_id"].extend([activity_id] * len(node_attr))
        node_data_dict["prediction"].extend([pred.item()] * len(node_attr))
        node_data_dict["delta"].extend([delta.item()] * len(node_attr))
        node_data_dict["node_index"].extend(range(len(node_attr)))
        node_data_dict["atomic_number"].extend(node_data.z.cpu().tolist())
        node_data_dict["is_pocket"].extend(
            node_data.is_pocket_atom.bool().cpu().squeeze().tolist()
        )

        # Edge attributions
        source_indices, target_indices = edge_index.cpu().tolist()
        edge_data_dict["attribution"].extend(edge_attr.tolist())
        edge_data_dict["activity_id"].extend([activity_id] * len(edge_attr))
        edge_data_dict["prediction"].extend([pred.item()] * len(edge_attr))
        edge_data_dict["delta"].extend([delta.item()] * len(edge_attr))
        edge_data_dict["edge_index"].extend(range(len(edge_attr)))
        edge_data_dict["source_index"].extend(source_indices.tolist())
        edge_data_dict["target_index"].extend(target_indices.tolist())

    node_df = pd.DataFrame(node_data_dict)
    edge_df = pd.DataFrame(edge_data_dict)
    return node_df, edge_df


class IntegratedGradientsCallback(Callback):
    def __init__(
        self,
        datasets: dict[str, KinodataDocked],
        frequency: int,
        start_epoch: int = 0,
        outdir: str | Path | None = None,
    ):
        super().__init__()
        self.datasets = datasets
        self.frequency = frequency
        self.start_epoch = start_epoch
        self._outdir = outdir
        if self.frequency == 0:
            logger.warning(
                "IntegratedGradientsCallback frequency is set to 0, IntegratedGradients will only run once at the end of training."
            )

    def log(self, *args, **kwargs):
        self._pl_module.log(*args, **kwargs)

    @cached_property
    def outdir(self) -> Path:
        assert hasattr(self, "_trainer"), (
            "IntegratedGradientsCallback must be setup before use"
        )
        if self._outdir is None:
            return (
                Path(self._trainer.log_dir)
                / "integrated_gradients"
                / _get_run_id(self._trainer)
            )

    @property
    def node_attribution_file(self) -> Path:
        return self.outdir / "node_attributions.csv"

    @property
    def edge_attribution_file(self) -> Path:
        return self.outdir / "edge_attributions.csv"

    def _write_config(self):
        config = {}
        with open(self.outdir / "ig_config.json", "w") as f:
            json.dump(config, f, indent=4)

    def _store_data_frame(self, data_frame: pd.DataFrame, file_path: str | Path):
        logger.info("Storing dataframe to %s", str(file_path))
        file_path = Path(file_path)
        if not file_path.exists():
            logger.info("Creating new dataframe csv file")
            data_frame.to_csv(file_path, index=False, mode="w", header=True)
            return
        logger.info("Appending to exisiting csv file")
        data_frame.to_csv(file_path, index=False, mode="a", header=False)

    def _handle_ig_result(
        self,
        node_attr: pd.DataFrame,
        edge_attr: pd.DataFrame,
        dataset_key: str,
        epoch: int,
        log: bool = False,
    ):
        node_attr["epoch"] = epoch
        edge_attr["epoch"] = epoch
        node_attr["dataset"] = dataset_key
        edge_attr["dataset"] = dataset_key
        self._store_data_frame(node_attr, self.node_attribution_file)
        self._store_data_frame(edge_attr, self.edge_attribution_file)
        if log:
            logger.warning(
                "Logging ig results to wandb is not implemented yet, skipping"
            )

    def _run_integrated_gradients(self, epoch, pl_module, **kwargs):
        for dataset_key, dataset in self.datasets.items():
            logger.info(f"Running IG on {dataset_key} dataset")
            dataset, orig_transform = remove_augementation_transforms_from_dataset(
                dataset, return_orig_transform=True
            )
            node_attr, edge_attr = compute_ig_attributions(
                WrappedComplexTransformer(pl_module),
                DataLoader(dataset, batch_size=1, shuffle=False),
                handle_device=True,
            )
            dataset.transform = orig_transform
            self._handle_ig_result(node_attr, edge_attr, dataset_key, epoch, **kwargs)

    def _compress_results(self):
        logger.info("Compressing IntegratedGradients results")
        compress_directory_to_tar_gz(self.outdir)

    def on_train_start(self, trainer, pl_module):
        self._trainer = trainer
        self._pl_module = pl_module
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
        logger.info("IntegratedGradientsCallback outdir: %s", self.outdir)
        self._write_config()
        return

    def on_train_epoch_end(self, trainer, pl_module):
        if self.frequency == 0:
            return
        if trainer.current_epoch < self.start_epoch:
            return
        if trainer.current_epoch % self.frequency != 0:
            return
        self._run_integrated_gradients(trainer.current_epoch, pl_module)
        return

    def on_fit_end(self, trainer, pl_module):
        self._run_integrated_gradients(trainer.current_epoch, pl_module, log=False)
        self._compress_results()
        return
