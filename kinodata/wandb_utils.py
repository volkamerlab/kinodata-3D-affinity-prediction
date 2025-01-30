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

api = wandb.Api()


def run_by_name(name):
    return list(
        api.runs(f"nextaids/kinodata-docked-rescore", filters={"display_name": name})
    )[0]


def run_by_id(run_id):
    return api.run(f"nextaids/kinodata-docked-rescore/{run_id}")


def latest_k_runs(k: int, state="finished"):
    return api.runs(f"nextaids/kinodata-docked-rescore", filters={"state": state})[:k]


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
        config = {k: v["value"] for k, v in config.items()}
        return Config(config)

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


retrieve_best_model_artifact = partial(retrieve_model_artifact, alias="best_k")


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
    return model
