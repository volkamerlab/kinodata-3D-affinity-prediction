from functools import partial
import wandb
import json
from dataclasses import dataclass
from typing import Any, Dict, List
import json
from dataclasses import dataclass

from argparse import ArgumentParser


@dataclass
class RunInfo:
    run: Any
    name: str
    split_path: List[str]
    mp_type: str
    tags: list
    config: Dict[str, Any]

    def __repr__(self) -> str:
        return f"{self.name}({self.run.id})"

    @classmethod
    def from_run(cls, run):
        config = json.loads(run.json_config)
        if "data_split" in config:
            split = config["data_split"]["value"].split("/")
        else:
            split = []
        if "mp_type" in config:
            mp_type = config["mp_type"]["value"]
        else:
            mp_type = None
        return cls(
            run,
            run.name,
            split,
            mp_type,
            run.tags,
            {key: obj["value"] for key, obj in json.loads(run.json_config).items()},
        )

    def has_likely_generic_name(self) -> bool:
        return self.name.count("-") == 2

    def split_known(self):
        return len(self.split_path) > 0

    @property
    def is_egnn(self) -> bool:
        return self.mp_type == "rbf"

    @property
    def model_name(self) -> str:
        if "dti" in self.tags:
            return "DTI"
        elif "ligand-only" in self.tags:
            return "GIN"
        elif self.is_egnn:
            suffix = ""
            if "pocket_residue" in self.config["node_types"]:
                if "residue_interaction_radius" in self.config:
                    suffix = f"(R/{self.config['residue_interaction_radius']})"
                else:
                    suffix = "(R)"
            return f"EGNN {suffix}"
        elif "kissim_size" in self.run.config and self.run.config["kissim_size"] == 12:
            return "DTI"
        return ""

    @property
    def split_seed(self) -> int:
        if len(self.split_path) > 0:
            return int(self.split_path[-1].split("_")[-1].split(".")[0])
        return None

    @property
    def split_type(self) -> str:
        try:
            return self.split_path[2]
        except IndexError:
            return None

    @property
    def new_name(self) -> str:
        if not self.split_known():
            raise ValueError(self)
        return f"{self.model_name} ({self.split_type}/{self.split_seed})"

    def rename_run(self, name: str = None):
        if name is None:
            name = self.new_name
        self.run.name = name
        self.run.update()

    @classmethod
    def fetch_all(cls, path="nextaids/kinodata-docked-rescore") -> List["RunInfo"]:
        api = wandb.Api()
        runs = api.runs(path)
        return [cls.from_run(run) for run in runs]


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
