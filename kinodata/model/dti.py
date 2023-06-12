from typing import Any, Callable, Optional, Protocol, Tuple

import torch
from torch import Tensor
from torch.nn import (
    Linear,
    Module,
    ModuleList,
    Sequential,
    Embedding,
    SiLU,
    BatchNorm1d,
    LayerNorm,
)
import pytorch_lightning as pl

from kinodata.configuration import Config
from kinodata.model.regression import RegressionModel
from kinodata.model.resolve import resolve_act, resolve_loss
from kinodata.model.shared import GINE, SetAttentionBlock
from torch_geometric.nn.pool import global_add_pool


class Encoder(Protocol):
    def __call__(self, batch: Any) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        batch : Any
            pyg data object

        Returns
        -------
        Tuple[Tensor, Optional[Tensor]]
            Embeddings of size (N, d) (float)
            Index tensor of size (N,) (long), assigns embeddings to their batch.
        """
        ...


class Decoder(Protocol):
    def __call__(
        self,
        ligand_embeddings: Tensor,
        pocket_embeddings: Tensor,
        ligand_batch: Optional[Tensor] = None,
        pocket_batch: Optional[Tensor] = None,
    ) -> Tensor:
        ...


class DTIModel(RegressionModel):
    def __init__(
        self,
        config: Config,
        ligand_encoder_cls: Callable[..., Encoder],
        pocket_encoder_cls: Callable[..., Encoder],
        decoder_cls: Callable[..., Decoder],
    ) -> None:
        super().__init__(config)
        self.criterion = resolve_loss(config.loss_type)
        self.ligand_encoder = config.init(ligand_encoder_cls)
        self.pocket_encoder = config.init(pocket_encoder_cls)
        self.decoder = config.init(decoder_cls)

    def forward(self, batch) -> Tensor:
        x_ligand, batch_ligand = self.ligand_encoder(batch)
        x_pocket, batch_pocket = self.pocket_encoder(batch)
        prediction = self.decoder(x_ligand, x_pocket, batch_ligand, batch_pocket)
        return prediction


class LigandGINE(Module):
    def __init__(self, hidden_channels: int, num_layers: int, act: str) -> None:
        super().__init__()
        self.atom_embedding = Embedding(100, hidden_channels)
        self.gine = GINE(hidden_channels, num_layers, edge_channels=4, act=act)

    def forward(self, batch) -> Tuple[Tensor, Tensor]:
        x = self.atom_embedding(batch["ligand"].z)
        h = self.gine(
            x=x,
            edge_index=batch["ligand", "bond", "ligand"].edge_index,
            edge_attr=batch["ligand", "bond", "ligand"].edge_attr,
            batch=batch["ligand"].batch,
        )
        return h, batch["ligand"].batch


class KissimPocketTransformer(Module):
    def __init__(
        self,
        kissim_size: int,
        hidden_channels: int,
        num_attention_blocks: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.lin1 = Sequential(
            Linear(kissim_size, hidden_channels), SiLU(), LayerNorm(hidden_channels)
        )
        self.attention_blocks = ModuleList(
            [
                SetAttentionBlock(hidden_channels, num_heads)
                for _ in range(num_attention_blocks)
            ]
        )

    def forward(self, batch) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.lin1(batch.kissim_fp.float())
        for attn in self.attention_blocks:
            x = attn(x)
        return x, None


def _sum_aggr(
    x: Tensor, index: Optional[Tensor] = None, feature_dim: int = 1
) -> Tensor:
    if index is not None:
        return global_add_pool(x, index)
    else:
        return x.sum(dim=feature_dim)


class GlobalSumDecoder(Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int = 1,
        act: str = "silu",
        feature_dim: int = -1,
    ) -> None:
        super().__init__()
        self.act = resolve_act(act)
        self.f_ligand = Sequential(
            Linear(hidden_channels, hidden_channels),
            self.act,
            BatchNorm1d(hidden_channels),
        )
        self.f_pocket = Sequential(
            Linear(hidden_channels, hidden_channels),
            self.act,
            BatchNorm1d(hidden_channels),
        )
        self.f_combined = Sequential(
            Linear(hidden_channels * 2, hidden_channels),
            self.act,
            Linear(hidden_channels, out_channels),
        )
        self.feature_dim = feature_dim

    def forward(
        self,
        ligand_embeddings: Tensor,
        pocket_embeddings: Tensor,
        ligand_batch: Optional[Tensor] = None,
        pocket_batch: Optional[Tensor] = None,
    ):
        ligand_repr = self.f_ligand(_sum_aggr(ligand_embeddings, ligand_batch))
        pocket_repr = self.f_pocket(_sum_aggr(pocket_embeddings, pocket_batch))
        combined_repr = torch.cat((ligand_repr, pocket_repr), dim=self.feature_dim)
        return self.f_combined(combined_repr)
