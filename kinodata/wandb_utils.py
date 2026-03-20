import torch
from torch import nn
from pathlib import Path
from functools import partial
import wandb
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from functools import cached_property
import pandas as pd

from argparse import ArgumentParser
from kinodata.configuration import Config

import os

api = wandb.Api()


def run_by_name(name, project="nextaids/kinodata-docked-rescore"):
    return list(api.runs(project, filters={"display_name": name}))[0]


def run_by_id(run_id, project="nextaids/kinodata-docked-rescore"):
    return api.run(f"{project}/{run_id}")


def latest_k_runs(k: int, state="finished", project="nextaids/kinodata-docked-rescore"):
    return api.runs(project, filters={"state": state})[:k]


def load_wandb_config(json_config):
    config = json.loads(json_config)
    config = {k: v["value"] for k, v in config.items()}
    return Config(config)


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
        return load_wandb_config(self.run.json_config)

    def rename_run(self, name: str = None):
        if name is None:
            name = self.new_name
        self.run.name = name
        self.run.update()


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


retrieve_best_model_artifact = partial(retrieve_model_artifact, alias="best")


def load_state_dict(
    artifact,
    return_artifact_dir: bool = False,
):
    artifact_dir = artifact.download()
    ckpt = torch.load(
        Path(artifact_dir) / "model.ckpt",
        map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    )
    if return_artifact_dir:
        return ckpt, artifact_dir
    return ckpt


def load_model_lazy(
    run_name: str = None,
    run_id: str = None,
    model_cls: Callable[[Config], nn.Module] = None,
    alias: str = None,
    override_config: Dict[str, Any] = None,
    return_config: bool = False,
):
    if run_id is not None:
        run = run_by_id(run_id)
    elif run_name is not None:
        run = run_by_name(run_name)
    else:
        raise ValueError("run_name or run_id must be provided")
    assert model_cls is not None
    if alias is None:
        artifact = retrieve_best_model_artifact(run)
    else:
        artifact = retrieve_model_artifact(run, alias)
    state_dict = load_state_dict(artifact)
    config = RunInfo(run).config
    if override_config is not None:
        config.update(override_config)
    model = model_cls(config)
    model.load_state_dict(state_dict["state_dict"])
    if return_config:
        return model, config
    return model


def _model_from_run(run):
    config = load_wandb_config(run.json_config)
    interaction_modes = config.get("interaction_modes", [])
    if "dimenet" in run.tags:
        return "DimeNet"
    if "covalent" in interaction_modes and "structural" in interaction_modes:
        return "CGNN-3D"
    if "covalent" in interaction_modes:
        return "CGNN"
    return None


def current_metric(run, key):
    try:
        return run.history(keys=[key]).tail(5)[key].mean()
    except:
        print(f"Failed getting {key}for {run.id}")
        return float("nan")


def run_ids_to_df(*run_ids):
    if len(run_ids) == 0:
        return None
    if len(run_ids) == 1 and isinstance(run_ids[0], list):
        run_ids = run_ids[0]
    runs = [run_by_id(run_id) for run_id in run_ids]
    run_df = pd.DataFrame(
        {
            "run": [run for run in runs],
            "run_id": [run.id for run in runs],
            "state": [run.state for run in runs],
            "split_type": [run.config["split_type"] for run in runs],
            "split_index": [run.config["split_index"] for run in runs],
            "test/corr": [run.summary.get("test/corr", float("nan")) for run in runs],
            "val/corr.max": [
                run.summary.get("val/corr", float("nan")).get("max") for run in runs
            ],
            "val/corr": [current_metric(run, "val/corr") for run in runs],
            "train/corr": [current_metric(run, "val/corr") for run in runs],
            "epoch": [run.summary.get("epoch", float("nan")) for run in runs],
            "model": [_model_from_run(run) for run in runs],
        }
    )
    return run_df


def load_wandb_table_as_pandas_data_frame(artifact_dir: Path): ...
