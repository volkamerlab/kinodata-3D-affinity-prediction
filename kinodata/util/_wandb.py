from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
import shutil
import os

import torch
import wandb

_here = Path(__file__)
_root = _here.parent.parent.parent
if (f_wandb_api_key := (_root / "wandb_api_key")).exists():
    api_key = f_wandb_api_key.read_text().strip()
    print("Setting wandb api key")
    os.environ["WANDB_API_KEY"] = api_key

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

def parse_wandb_json_config(
    arg: Path | dict,
):
    if isinstance(arg, Path):
        arg = json.loads(arg.read_text())
    assert isinstance(arg, dict)
    return {
        key: val_and_desc["value"]
        for key, val_and_desc in arg.items()
    }

def find_model_ckpt(path: Path) -> Path | None:
    for fp in path.glob("**/*.ckpt"):
        return fp
    return None

@dataclass(frozen=True)
class ModelInfo:
    config: dict
    fp_model_ckpt: Path
   
    @property 
    def model_type(self) -> str:
        # TODO derive from config
        return "CGNN-3D"
   
    @classmethod 
    def from_dir(cls, path: Path):
        return cls(
            config=parse_wandb_json_config(path / "config.json"),
            fp_model_ckpt=find_model_ckpt(path),
        )

class WandbInterface:
    
    def __init__(self) -> None:
        self.api = wandb.Api()
    
    def download_model_from_wandb(
        self,
        run_id: int | str,
        out_dir: Path | None = None
    ) -> Path:
        out_dir = out_dir or Path(str(run_id))
        if not out_dir.exists():
            out_dir.mkdir()
        run = self.api.run(f"nextaids/kinodata-docked-rescore/{run_id}")
        json_config = run.json_config
        artifact = retrieve_best_model_artifact(run)
        artifact_path = Path(artifact.download())
        for subdir in out_dir.iterdir():
            shutil.rmtree(subdir)
        shutil.move(artifact_path, out_dir)
        with open(
            (fp_config := out_dir / "config.json"), "w"
        ) as f_config:
            f_config.write(json_config)
        ckpt_path = find_model_ckpt(out_dir)
        assert ckpt_path is not None
        return ModelInfo(
            config=parse_wandb_json_config(fp_config),
            fp_model_ckpt=ckpt_path
        ) 

wandb_interface = WandbInterface() 