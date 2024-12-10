from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import time

DATA = Path("data") / "remote_ig_attrbibutions"

df = pd.read_csv("data/processed/kinodata_docked.csv").set_index("ident")


def get_time(path: Path):
    # read a timestamp in the format YYYY-MM-DD_HH-MM-SS
    print(path)
    timestamp = path.name
    ptime = time.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
    print(ptime)
    return ptime


def analyze(path: Path | None = None):
    # use newest attributions if path is None
    if path is None:
        path = max(DATA.iterdir(), key=lambda p: get_time(p))

    print(f"The attributions are loaded from {path}")
    attrs = torch.load(path / "attrs.pt")
    deltas = torch.load(path / "deltas.pt")
    idents = torch.load(path / "idents.pt")

    print(f"Number of complexes considered: {len(attrs)}")
    print(deltas[:10])
    print(idents[:10])

    all_ligand_attr = []
    all_pocket_attr = []
    all_deltas = []
    for ident, attr, delta in zip(idents, attrs, deltas):
        ident = ident.item()
        row = df.loc[ident]
        ls = row["ligand_size"]
        attr = attr.squeeze()
        ligand_attr = attr[-ls:]
        pocket_attr = attr[0:-ls]
        all_ligand_attr.append(
            ligand_attr,
        )
        all_pocket_attr.append(pocket_attr)
        all_deltas.append(delta)

    all_ligand_attr = torch.cat(all_ligand_attr, 0)
    all_pocket_attr = torch.cat(all_pocket_attr, 0)

    attr_df = defaultdict(list)
    for a in all_ligand_attr.tolist():
        attr_df["attr"].append(a)
        attr_df["kind"].append("ligand")
    for a in all_pocket_attr.tolist():
        attr_df["attr"].append(a)
        attr_df["kind"].append("pocket")
    attr_df = pd.DataFrame(attr_df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    all_ligand_attr = all_ligand_attr.numpy()
    sns.kdeplot(x=all_ligand_attr, ax=ax1)
    sns.rugplot(x=all_ligand_attr, ax=ax1)
    q1, q9 = np.quantile(all_ligand_attr, [0.1, 0.9])
    std = np.std(all_ligand_attr)
    mean = np.mean(all_ligand_attr)
    ax1.set_title(
        f"ligand attr.: mean={mean:.2f}, q1={q1:.2f}, q9={q9:.2f}, std={std:.2f}"
    )

    all_ligand_attr = all_pocket_attr.numpy()
    sns.kdeplot(x=all_ligand_attr, ax=ax2)
    sns.rugplot(x=all_ligand_attr, ax=ax2)
    q1, q9 = np.quantile(all_ligand_attr, [0.1, 0.9])
    std = np.std(all_ligand_attr)
    mean = np.mean(all_ligand_attr)
    ax2.set_title(
        f"pocket attr.: mean={mean:.2f}, q1={q1:.2f}, q9={q9:.2f}, std={std:.2f}"
    )
    plt.show()


def attr_distr(attrs):
    attrs = attrs.sum(dim=-1)
    print(attrs.sum())
    print(attrs.mean())
    print(attrs.std())
    sns.displot(x=attrs.flatten().numpy())
    plt.show()


if __name__ == "__main__":
    analyze()
