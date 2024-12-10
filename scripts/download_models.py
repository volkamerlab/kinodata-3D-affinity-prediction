from datetime import datetime
from pathlib import Path
import json
import shutil

from wandb_utils import RunInfo, retrieve_best_model_artifact


def filter_cgnn(run: RunInfo) -> bool: ...


if __name__ == "__main__":
    run_filter = lambda run: True
    since = datetime(2024, 10, 30)
    runs = RunInfo.fetch(since=since)
    dry_run = False
    move_to = Path("models") / "cgnn_new"
    print(f"Move to path is {move_to.absolute()}, exists? {move_to.exists()}")
    if not move_to.exists():
        move_to.mkdir(parents=True)
    print(f"Found {len(runs)} runs since {since}")
    runs = [run for run in runs if run_filter(run)]
    runs = {run.run.id: run for run in runs}
    print(f"{len(runs)} runs remain after filtering")
    model_artifacts = [retrieve_best_model_artifact(run.run) for run in runs.values()]

    artifacts_paths = {}
    for artifact, run_id in zip(model_artifacts, runs.keys()):
        if artifact is None:
            print(f"Run {run_id}: no model artifact found")
            artifacts_paths[run_id] = None
            continue
        if dry_run:
            print(f"Dry run: would download {artifact.name}")
            print(f"Would move to {move_to / run_id}")
            continue
        if (move_to / run_id).exists():
            artifacts_paths[run_id] = move_to / run_id
            continue
        artifacts_paths[run_id] = artifact.download()
        shutil.move(artifacts_paths[run_id], move_to / run_id)
        artifacts_paths[run_id] = move_to / run_id

    if dry_run:
        exit(0)

    for run_id, path in artifacts_paths.items():
        if path is None:
            continue
        run = runs[run_id]
        config_fp = Path(path) / "config.json"
        config_fp.write_text(json.dumps(run.config))
