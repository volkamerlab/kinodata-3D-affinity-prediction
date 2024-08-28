from pathlib import Path
import time
from types import MethodType

import captum
import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from kinodata.data.data_module import create_dataset
from kinodata.data.dataset import Filtered, KinodataDocked
from kinodata.data.grouped_split import KinodataKFoldSplit
from kinodata.model.complex_transformer import ComplexTransformer
from kinodata.transform.filter_metadata import FilterDockingRMSD
from kinodata.transform.to_complex_graph import TransformToComplexGraph
from kinodata.types import NodeType
from kinodata.configuration import Config

from crocodoc_residues import load_model_from_checkpoint


def prepare_data(rmsd_thresh, split_type, split_fold):
    to_cplx = TransformToComplexGraph(remove_heterogeneous_representation=True)
    rmsd_filter = FilterDockingRMSD(rmsd_thresh)
    data_cls = Filtered(KinodataDocked(), rmsd_filter)
    test_split = KinodataKFoldSplit(split_type, 5)
    split = test_split.split(data_cls())[split_fold]
    test_data = create_dataset(
        data_cls,
        dict(),
        split.test_split,
        None,
    )
    forbidden_seq = set(
        [
            "KALGKGLFSMVIRITLKVVGLRILNLPHLILEYCKAKDIIRFLQQKNFLLLINWGIR",
            "LIGKGDSARLDYLVVGRLLQLVREP",
            "LIIGKGDFGKVELSALKVVDIIRLILDYLVVGRLLQLVRE",
            "NKMGEGGFGVVYKVAVKKLQFDQEIKVMAKCQENLVELLGFCLVYVYMPNGSLLDRLSCFLHENHHIHRDIKSANILLISDFGLA",
            "_ALNVLDMSQKLYLLSSLDPYLLEMYSYLILEAPEGEIFNLLRQYLHSAMIIYRDLKPHNVLFIAA",
        ]
    )
    return [
        to_cplx(data) for data in test_data if data.pocket_sequence not in forbidden_seq
    ]


# this is far from pretty but it works
def inject_partial_forward(model: ComplexTransformer) -> ComplexTransformer:
    def compute_initial_embeds(self: ComplexTransformer, data: HeteroData):
        node_store = data[NodeType.Complex]
        node_repr = self.initial_embed_nodes(data)
        if (batch := node_store.get("batch", None)) is None:
            batch = torch.zeros(
                node_repr.size(0), dtype=torch.long, device=node_repr.device
            )
            node_store["batch"] = batch
        edge_index, edge_repr = self.initial_embed_edges(data)

        return node_repr, edge_repr, edge_index, node_store.batch

    def forward_initial_embeds(
        self: ComplexTransformer,
        node_embed: Tensor,
        edge_embed: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ):
        node_embed = node_embed.squeeze()
        for sparse_attention_block, norm in zip(
            self.attention_blocks, self.norm_layers
        ):
            node_embed, edge_embed = sparse_attention_block(
                node_embed, edge_embed, edge_index
            )
            node_embed = norm(node_embed, batch)

        graph_repr = self.aggr(node_embed, batch)
        return self.out(graph_repr)

    model.compute_initial_embeds = MethodType(compute_initial_embeds, model)
    model.forward_initial_embeds = MethodType(forward_initial_embeds, model)
    return model


def compute_attributions(
    model: ComplexTransformer,
    loader: DataLoader,
    collapse_hidden_dim: bool = True,
):
    ig = captum.attr.IntegratedGradients(model.forward_initial_embeds)
    attrs = []
    deltas = []
    idents = []
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("CUDA is available, using GPU")
        device = torch.device("cuda")
    model = model.to(device)
    for data in tqdm(loader[:5]):
        data = data.to(device)
        with torch.no_grad():
            node_embed, edge_embed, edge_index, batch = model.compute_initial_embeds(
                data
            )
            node_embed.unsqueeze_(0)
        attr, delta = ig.attribute(
            node_embed,
            additional_forward_args=(edge_embed, edge_index, batch),
            return_convergence_delta=True,
            internal_batch_size=1,
            n_steps=150,
        )
        if collapse_hidden_dim:
            attr = attr.sum(dim=-1)
        attrs.append(attr.detach().cpu())
        deltas.append(delta.detach().cpu())
        idents.append(data.ident.detach().cpu())
    return attrs, deltas, idents


if __name__ == "__main__":
    config = Config()
    config = config.update(
        {
            "rmsd": 2,
            "split_type": "scaffold-k-fold",
            "fold": 0,
            "save_to": "data/ig_attributions",
        }
    )
    config = config.update_from_args()

    rmsd, split_type, fold = config["rmsd"], config["split_type"], config["fold"]

    save_to = Path(config["save_to"])
    # create a timestamp in the format YYYY-MM-DD_HH-MM-SS
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_to = save_to / timestamp
    if not save_to.exists():
        save_to.mkdir(parents=True)
    model, train_config = load_model_from_checkpoint(rmsd, split_type, fold, "CGNN-3D")
    model = inject_partial_forward(model)
    model.eval()
    data = prepare_data(rmsd, split_type, fold)
    attrs, deltas, idents = compute_attributions(model, data)

    # save config as a json file
    with open(save_to / "config.json", "w") as f:
        f.write(config.to_json())

    torch.save(attrs, save_to / "attrs.pt")
    torch.save(deltas, save_to / "deltas.pt")
    torch.save(idents, save_to / "idents.pt")
    pass
