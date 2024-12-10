from functools import partial
from typing import List, Optional, Tuple
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from torch.nn import (
    ModuleList,
    Embedding,
    Sequential,
    Linear,
    Module,
    Parameter,
    Dropout,
    BatchNorm1d,
)
from torch_geometric.data import HeteroData
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.utils import coalesce
from torch_geometric.utils import to_dense_batch
from torch_cluster import knn_graph
import wandb

from kinodata.configuration import Config

from ..types import NodeEmbedding, NodeType, RelationType, MASK_RESIDUE_KEY
from .shared.dist_embedding import GaussianDistEmbedding
from .sparse_transformer import SPAB
from .regression import RegressionModel, cat_many
from .resolve import resolve_act, resolve_loss
from ..data.featurization.atoms import AtomFeatures
from ..data.featurization.bonds import NUM_BOND_TYPES
from .dti import make_model, Encoder, KissimTransformer

OptTensor = Optional[Tensor]


def FF(din, dout, act):
    return Sequential(Linear(din, dout), act)


class InteractionModule(Module):
    hidden_channels: int

    def __init__(
        self, hidden_channels: int, act: str = "none", bias: bool = False
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.act = resolve_act(act)
        self.bias = Parameter(torch.zeros(self.hidden_channels), requires_grad=bias)

    def out(self, edge_repr: Tensor) -> Tensor:
        return self.act(edge_repr + self.bias)

    def interactions(self, data: HeteroData) -> Tuple[Tensor, OptTensor, OptTensor]: ...

    def process_attr(self, edge_attr: Tensor) -> Tensor: ...

    def process_weight(self, edge_weight: Tensor) -> Tensor: ...

    def forward(self, data: HeteroData) -> Tuple[Tensor, Tensor]:
        edge_index, edge_attr, edge_weight = self.interactions(data)
        device = [x.device for x in (edge_attr, edge_weight) if x is not None][0]
        edge_repr = torch.zeros(edge_index.size(1), self.hidden_channels, device=device)
        if edge_attr is not None:
            edge_repr = edge_repr + self.process_attr(edge_attr)
        if edge_weight is not None:
            edge_repr = edge_repr + self.process_weight(edge_weight)
        return edge_index, self.out(edge_repr)


class CovalentInteractions(InteractionModule):
    def __init__(
        self,
        hidden_channels: int,
        act: str = "none",
        bias: bool = False,
        node_type: str = None,
        edge_size: int = NUM_BOND_TYPES,
    ) -> None:
        super().__init__(hidden_channels, act, bias)
        assert node_type is not None
        self.node_type = node_type
        self.lin = Linear(edge_size, hidden_channels)

    def interactions(self, data: HeteroData) -> Tuple[Tensor, OptTensor, OptTensor]:
        edge_store = data[self.node_type, RelationType.Covalent, self.node_type]
        return edge_store.edge_index, edge_store.edge_attr, None

    def process_attr(self, edge_attr: Tensor) -> Tensor:
        return self.lin(edge_attr.float())


class StructuralInteractions(InteractionModule):
    def __init__(
        self,
        hidden_channels: int,
        act: str = "none",
        bias: bool = False,
        interaction_radius: float = 8.0,
        max_num_neighbors: int = 16,
        rbf_size: int = None,
        mask_pl_edges: bool = False,
    ) -> None:
        super().__init__(hidden_channels, act, bias)
        self.interaction_radius = interaction_radius
        self.max_num_neighbors = max_num_neighbors
        self.rbf_size = rbf_size if rbf_size else hidden_channels
        self.mask_pl_edges = mask_pl_edges
        self.distance_embedding = GaussianDistEmbedding(rbf_size, interaction_radius)
        self.lin = Linear(rbf_size, hidden_channels, bias=False)

        self.hacky_mask = None

    def interactions(self, data: HeteroData) -> Tuple[Tensor, OptTensor, OptTensor]:
        self.hacky_mask = None
        pos = data[NodeType.Complex].pos
        batch = data[NodeType.Complex].batch
        edge_index = knn_graph(pos, self.max_num_neighbors + 1, batch, loop=True)
        distances = (pos[edge_index[0]] - pos[edge_index[1]]).pow(2).sum(dim=1).sqrt()
        mask = distances <= self.interaction_radius
        edge_index = edge_index[:, mask]
        distances = distances[mask]
        row, col = edge_index
        node_store = data[NodeType.Complex]

        # masking
        if MASK_RESIDUE_KEY in node_store:
            is_part_of_masked_residue = node_store[MASK_RESIDUE_KEY].squeeze()
            is_ligand_atom = (~node_store.is_pocket_atom).squeeze()
            is_pl_edge_at_masked_residue = torch.logical_or(
                torch.logical_and(is_part_of_masked_residue[row], is_ligand_atom[col]),
                torch.logical_and(is_ligand_atom[row], is_part_of_masked_residue[col]),
            )
            self.hacky_mask = ~is_pl_edge_at_masked_residue
        elif self.mask_pl_edges:
            is_pl_egde = torch.logical_xor(
                node_store.is_pocket_atom[row].squeeze(),
                node_store.is_pocket_atom[col].squeeze(),
            )
            self.hacky_mask = ~is_pl_egde

        if self.hacky_mask is not None:
            edge_index = edge_index.T[self.hacky_mask].T
            distances = distances[self.hacky_mask]
        return edge_index, None, distances

    def process_weight(self, edge_weight: Tensor) -> Tensor:
        return self.distance_embedding(edge_weight)


class CombinedInteractions(Module):
    def __init__(self, interactions: List[InteractionModule], act: str) -> None:
        super().__init__()
        self.interactions = ModuleList(interactions)
        self.act = resolve_act(act)
        assert (
            len(set([intr.hidden_channels for intr in interactions])) == 1
        ), "Interactions should not map edge representation to different number of hidden channels."
        self.bias = Parameter(torch.zeros(interactions[0].hidden_channels))

    def forward(self, data: HeteroData) -> Tuple[Tensor, Tensor]:
        edge_indices, edge_reprs = zip(*[intr(data) for intr in self.interactions])
        edge_index, edge_repr = coalesce(
            torch.cat(edge_indices, 1), torch.cat(edge_reprs, 0), reduce="add"
        )
        return edge_index, self.act(edge_repr + self.bias)


class ComplexTransformer(RegressionModel):
    def __init__(
        self,
        config: Config,
        hidden_channels: int,
        num_heads: int,
        num_attention_blocks: int,
        interaction_radius: float,
        max_num_neighbors: int,
        act: str,
        max_atomic_number: int = 100,
        atom_attr_size: int = AtomFeatures.size,
        ln1: bool = True,
        ln2: bool = False,
        ln3: bool = True,
        graph_norm: bool = True,
        decoder_hidden_layers: int = 1,
        interaction_modes: List[str] = [],
        dropout: float = 0.1,
        mask_pl_edges: bool = False,
        edge_size: int = NUM_BOND_TYPES,
    ) -> None:
        super().__init__(config)
        assert len(config["node_types"]) == 1
        assert config["node_types"][0] == NodeType.Complex
        self.act = resolve_act(act)
        self.d_cut = interaction_radius
        self.max_num_neighbors = max_num_neighbors
        intr_bias = len(interaction_modes) == 1
        intr = []
        for mode in interaction_modes:
            if mode == "covalent":
                module = CovalentInteractions(
                    hidden_channels,
                    act,
                    intr_bias,
                    config["node_types"][0],
                    edge_size=edge_size,
                )
            elif mode == "structural":
                module = StructuralInteractions(
                    hidden_channels,
                    act,
                    intr_bias,
                    interaction_radius,
                    max_num_neighbors,
                    hidden_channels,
                    mask_pl_edges,
                )
            else:
                raise ValueError(mode)
            intr.append(module)
        if len(intr) > 1:
            self.interaction_module = CombinedInteractions(intr, act)
        else:
            self.interaction_module = intr[0]

        self.atomic_num_embedding = Embedding(max_atomic_number, hidden_channels)
        self.lin_atom_features = Linear(atom_attr_size, hidden_channels)
        self.attention_blocks = ModuleList(
            [
                SPAB(hidden_channels, num_heads, self.act, ln1, ln2, ln3)
                for _ in range(num_attention_blocks)
            ]
        )
        if graph_norm:
            self.norm_layers = ModuleList(
                [GraphNorm(hidden_channels) for _ in range(num_attention_blocks)]
            )
        else:
            self.norm_layers = [lambda x, b: x] * num_attention_blocks
        self.aggr = SoftmaxAggregation(learn=True, channels=hidden_channels)
        self.out = Sequential(
            *(
                [Dropout(dropout), BatchNorm1d(hidden_channels)]
                + [
                    FF(hidden_channels, hidden_channels, self.act)
                    for _ in range(decoder_hidden_layers)
                ]
                + [Linear(hidden_channels, 1)]
            )
        )

        config["residue_size"] = 12
        kissim_encoder = config.init(KissimTransformer)
        self.pocket_baseline = Sequential(
            kissim_encoder, self.act, Linear(hidden_channels, 1)
        )

    def initial_embed_nodes(self, data: HeteroData) -> Tensor:
        node_store = data[NodeType.Complex]
        node_repr = self.act(
            self.atomic_num_embedding(node_store.z)
            + self.lin_atom_features(node_store.x)
        )
        return node_repr

    def initial_embed_edges(self, data: HeteroData):
        edge_index, edge_repr = self.interaction_module(data)
        return edge_index, edge_repr

    def final_prediction(self, node_repr, edge_repr, edge_index, batch):
        for sparse_attention_block, norm in zip(
            self.attention_blocks, self.norm_layers
        ):
            node_repr, edge_repr = sparse_attention_block(
                node_repr, edge_repr, edge_index
            )
            node_repr = norm(node_repr, batch)
        graph_repr = self.aggr(node_repr, batch)
        return self.out(graph_repr)
            node_repr = norm(node_repr, node_store.batch)

        graph_repr = self.aggr(node_repr, node_store.batch)
        affinity_prediction = self.out(graph_repr)
        pocket_baseline_affinity = self.pocket_baseline(data.kissim_fp)
        return affinity_prediction, pocket_baseline_affinity

    def forward(self, data: HeteroData) -> Tensor:
        node_store = data[NodeType.Complex]
        node_repr = self.initial_embed_nodes(data)
        edge_index, edge_repr = self.initial_embed_edges(data)
        return self.final_prediction(node_repr, edge_repr, edge_index, node_store.batch)


class ComplexTransformerWithDebiasingBaseline(ComplexTransformer):

    def set_criterion(self):
        self.criterion = resolve_loss(self.config.loss_type, with_baseline=True)

    def remove_pl_interactions(self, edge_index, edge_repr, node_store):
        row, col = edge_index
        is_source_pocket = node_store.is_pocket_atom[row]
        is_target_pocket = node_store.is_pocket_atom[col]
        is_pl_edge = torch.logical_xor(is_source_pocket, is_target_pocket).squeeze()

        return edge_index[:, ~is_pl_edge], edge_repr[~is_pl_edge]

    def forward(self, data: HeteroData) -> Tensor:
        node_store = data[NodeType.Complex]
        node_repr = self.initial_embed_nodes(data)
        edge_index, edge_repr = self.initial_embed_edges(data)

        pred_activity = self.final_prediction(
            node_repr, edge_repr, edge_index, node_store.batch
        )

        masked_edge_index, masked_edge_repr = self.remove_pl_interactions(
            edge_index, edge_repr, node_store
        )
        pred_baseline = self.final_prediction(
            node_repr, masked_edge_repr, masked_edge_index, node_store.batch
        )

        return pred_activity, pred_baseline

    def training_step(self, batch, *args) -> Tensor:
        pred, baseline = self.forward(batch)
        pred = pred.view(-1, 1)
        baseline = baseline.view(-1, 1)
        loss = self.criterion(pred, baseline, batch.y.view(-1, 1))
        self.log("train/loss", loss, batch_size=pred.size(0), on_epoch=True)
        return loss

    def validation_step(self, batch, *args, key: str = "val"):
        pred, baseline = self.forward(batch)
        pred = pred.flatten()
        baseline = baseline.flatten()
        val_mae = (pred - batch.y).abs().mean()
        abs_baseline = baseline.abs()
        self.log(f"{key}/mae", val_mae, batch_size=pred.size(0), on_epoch=True)
        self.log(
            f"{key}/abs_baseline",
            abs_baseline.mean(),
            batch_size=pred.size(0),
            on_epoch=True,
        )
        return {
            f"{key}/mae": val_mae,
            "pred": pred,
            "abs_baseline": abs_baseline,
            "target": batch.y,
            "ident": batch.ident,
        }

    def process_eval_outputs(self, outputs) -> float:
        pred = torch.cat([output["pred"] for output in outputs], 0)
        target = torch.cat([output["target"] for output in outputs], 0)
        abs_baseline = torch.cat([output["abs_baseline"] for output in outputs], 0)
        corr = ((pred - pred.mean()) * (target - target.mean())).mean() / (
            pred.std() * target.std()
        ).cpu().item()
        mae = (pred - target).abs().mean()
        return pred, target, corr, mae, abs_baseline

    def validation_epoch_end(self, outputs, *args, **kwargs) -> None:
        pred, target, corr, mae, ab = self.process_eval_outputs(outputs)
        self.log("val/corr", corr)

        if self.log_scatter_plot:
            y_min = min(pred.min().cpu().item(), target.min().cpu().item()) - 1
            y_max = max(pred.max().cpu().item(), target.max().cpu().item()) + 1
            fig, ax = plt.subplots()
            ax.scatter(target.cpu().numpy(), pred.cpu().numpy(), s=0.7)
            ax.set_xlim(y_min, y_max)
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel("Pred")
            ax.set_xlabel("Target")
            ax.set_title(f"corr={corr}")
            wandb.log({"scatter_val": wandb.Image(fig)})

    def predict_step(self, batch, *args):
        pred = self.forward(batch).flatten()
        return {"pred": pred, "target": batch.y.flatten()}

    def test_epoch_end(self, outputs, *args, **kwargs) -> None:
        pred, target, corr, mae, ab = self.process_eval_outputs(outputs)
        self.log("test/mae", mae)
        self.log("test/abs_baseline", ab)
        self.log("test/corr", corr)

        if self.log_test_predictions:
            test_predictions = wandb.Artifact("test_predictions", type="predictions")
            data = cat_many(outputs, subset=["pred", "ident"])
            values = [t.detach().cpu() for t in data.values()]
            values = torch.stack(values, dim=1)
            table = wandb.Table(columns=list(data.keys()), data=values.tolist())
            test_predictions.add(table, "predictions")
            wandb.log_artifact(test_predictions)
            pass


def make_model(config: Config):
    if config.get("with_debiasing_baseline", False):
        cls = partial(ComplexTransformerWithDebiasingBaseline, config)
    else:
        cls = partial(ComplexTransformer, config)
    return config.init(cls)
