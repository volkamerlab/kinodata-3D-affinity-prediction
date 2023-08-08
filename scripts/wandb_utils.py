import torch
from pathlib import Path
from functools import partial
import wandb
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime
from functools import cached_property
import pandas as pd

from argparse import ArgumentParser


@dataclass
class Split:
    type: str
    index: int

    @classmethod
    def from_path(cls, path: str):
        elements = path.split("/")
        try:
            if len(elements) > 0:
                index = int(elements[-1].split("_")[-1].split(".")[0])
        except:
            index = -1
        try:
            type = elements[2]
        except IndexError:
            type = ""
        return cls(type, index)


class RunInfo:
    def __init__(self, run) -> None:
        self.run = run

    def __repr__(self) -> str:
        return f"{self.name}({self.run.id})"

    @property
    def name(self) -> str:
        return self.run.name

    @cached_property
    def config(self) -> Dict[str, Any]:
        config = json.loads(self.run.json_config)
        return config

    def get(self, key, default=None):
        try:
            return self.config[key]["value"]
        except KeyError:
            return default

    @cached_property
    def split(self) -> Split:
        if "split_type" in self.config:
            return Split(
                self.config["split_type"]["value"],
                self.config["split_index"]["value"],
            )
        if "data_split" in self.config:
            split_path = self.config["data_split"]["value"]
            return Split.from_path(split_path)

    @property
    def is_egnn(self) -> bool:
        return self.config.get("mp_type") == "rbf"

    @property
    def model_name(self) -> str:
        """Don't judge me xdd"""

        if "dti" in self.run.tags:
            return "DTI"
        elif "ligand-only" in self.run.tags:
            return "GIN"
        elif self.config.get("model_type", None) == "rel-egnn":
            return "REL-EGNN"
        elif "transformer" in self.run.tags:
            prefix = ""
            if "interaction_modes" in self.config:
                prefix = "Covalent"
                if "structural" in self.get("interaction_modes"):
                    prefix = "Structural"
            return f"{prefix} Transformer"
        elif self.is_egnn:
            suffix = ""
            if "pocket_residue" in self.config["node_types"]:
                if "residue_interaction_radius" in self.config:
                    suffix = f"(R/{self.config['residue_interaction_radius']})"
                else:
                    suffix = "(R)"
            return f"EGNN {suffix}"
        elif "kissim_size" in self.config and self.config["kissim_size"] == 12:
            return "DTI"
        return ""

    def rename_run(self, name: str = None):
        if name is None:
            name = self.new_name
        self.run.name = name
        self.run.update()

    def retrieve_predictions(self) -> Optional[pd.DataFrame]:
        artifacts = [
            artifact
            for artifact in self.run.logged_artifacts()
            if "predictions" in artifact.name
        ]
        if len(artifacts) == 0:
            return None
        if len(artifacts) > 1:
            raise ValueError("More than one prediction artifact.")
        artifact_path = Path(artifacts[0].download()) / "predictions.table.json"
        predictions_dict = json.loads(artifact_path.read_text())
        predictions = pd.DataFrame(
            data=predictions_dict["data"], columns=predictions_dict["columns"]
        )
        predictions["ident"] = predictions["ident"].astype(int)
        return predictions

    @classmethod
    def fetch(
        cls,
        path="nextaids/kinodata-docked-rescore",
        since: Optional[datetime] = None,
    ) -> List["RunInfo"]:
        filters = list()
        if since is not None:
            filters.append({"createdAt": {"$gt": str(since)}})

        api = wandb.Api()
        if len(filters) == 1:
            filters = filters[0]
        elif len(filters) > 1:
            filters = {"$and": filters}
        else:
            filters = None
        runs = api.runs(path, filters=filters)
        return [cls(run) for run in runs]


_sweep_parser = ArgumentParser()
_sweep_parser.add_argument("--sweep_id")


def try_parse_sweep():
    args, _ = _sweep_parser.parse_known_args()
    return args.sweep_id


def sweepable(func, sweep_id=None):
    get_sweep = lambda: sweep_id
    if sweep_id is None:
        get_sweep = try_parse_sweep

    def maybe_sweep(*args, **kwargs):
        sweep_id = get_sweep()
        if sweep_id is None:
            return func(*args, **kwargs)
        else:
            return wandb.agent(sweep_id, function=func)

    return maybe_sweep


def sweep(sweep_id):
    return partial(sweepable, sweep_id=sweep_id)


def retrieve_model_artifact(run, alias: str):
    for artifact in run.logged_artifacts():
        if artifact.type != "model":
            continue
        if alias in artifact.aliases:
            return artifact
    return None


retrieve_best_model_artifact = partial(retrieve_model_artifact, alias="best_k")


def load_state_dict(artifact):
    artifact_dir = artifact.download()
    ckpt = torch.load(
        Path(artifact_dir) / "model.ckpt", map_location=torch.device("cpu")
    )
    return ckpt
