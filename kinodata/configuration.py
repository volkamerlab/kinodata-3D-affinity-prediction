from typing import Any, Dict, Optional, Union
import traceback
from collections import defaultdict
from pathlib import Path
from warnings import warn

import torch
import yaml

from kinodata.typing import Config


configs: Dict[str, Config] = dict()


def register(config_name: str, config: Optional[Config] = None, **kwargs: Any):
    if config is None:
        config = kwargs
    configs[config_name] = config


def extend(config_name: str, **kwargs: Any):
    configs[config_name] |= kwargs


def load_from_file(fp: Path) -> Config:
    try:
        file_stream = fp.read_text()
        config = yaml.full_load(file_stream)
    except:
        print(f"Not a valid config yaml file: {fp}")
        traceback.print_exc()
        exit(1)
    return config


def overwrite_from_file(config: Config, fp: Union[str, Path], verbose: bool = True):
    fp = Path(fp)
    if not fp.exists():
        warn(f"Config file does not exist: {fp}")
        return config
    file_config = load_from_file(fp)
    if verbose:
        print(f"Reading additional config from {fp}:")
        print("\n".join([f"{key}: {value}" for key, value in file_config.items()]))
    return config | file_config


def get(*config_names: str) -> Config:
    key_counts = defaultdict(int)
    for name in config_names:
        assert name in configs, f'Unknown config: "{name}"'
        for key in configs[name]:
            key_counts[key] += 1

    duplicate_keys = [key for key in key_counts if key_counts[key] > 1]
    if len(duplicate_keys) > 0:
        raise ValueError(f"Duplicate config keys: {duplicate_keys}")

    config: Config = dict()
    for name in config_names:
        config |= configs[name]
    return config


register(
    "data",
    interaction_radius=5.0,
    seed=420,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    use_bonds=True,
    add_artificial_decoys=True,
    cold_split=False,
)

register(
    "model",
    num_mp_layers=4,
    hidden_channels=64,
    act="silu",
    mp_type="rbf",
    mp_reduce="sum",
    rbf_size=64,
    readout_type="sum",
)

register(
    "training",
    lr=3e-4,
    weight_decay=1e-5,
    batch_size=32,
    accumulate_grad_batches=2,
    epochs=100,
    num_workers=32,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    loss_type="mse",
    lr_factor=0.7,
    lr_patience=5,
    min_lr=1e-7,
)
