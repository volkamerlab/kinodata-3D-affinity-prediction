from typing import List, Union
from torch_geometric.data import HeteroData
from torch.nn import (
    Embedding,
    Module,
    Linear,
    LayerNorm,
    Identity,
    ModuleDict,
    ModuleList,
)
from torch import Tensor, cat

from kinodata.types import NodeEmbedding
from kinodata.model.resolve import resolve_act


class CategoricalEmbedding(Module):
    @property
    def out_channels(self) -> int:
        return self.embedding.embedding_dim

    def __init__(
        self,
        node_type: str,
        hidden_chanels: int,
        category_key: str,
        max_num_categories: int = 100,
    ) -> None:
        super().__init__()
        self.node_type = node_type
        self.category_key = category_key
        self.embedding = Embedding(max_num_categories, hidden_chanels)

    def forward(self, data: HeteroData) -> Tensor:
        category = getattr(data[self.node_type], self.category_key)
        return self.embedding(category)


class AtomTypeEmbedding(CategoricalEmbedding):
    def __init__(
        self,
        node_type: str,
        hidden_chanels: int,
        category_key: str = "z",
        max_num_categories: int = 100,
    ) -> None:
        super().__init__(node_type, hidden_chanels, category_key, max_num_categories)


class FeatureEmbedding(Module):
    @property
    def out_channels(self) -> int:
        return self.embedding.embedding_dim

    def __init__(
        self,
        node_type: str,
        in_channels: int,
        hidden_channels: int,
        act: str,
        embedding_norm: bool = True,
    ) -> None:
        super().__init__()
        self.node_type = node_type
        self.lin = Linear(in_channels, hidden_channels)
        self.act = resolve_act(act)
        if embedding_norm:
            self.norm = LayerNorm(hidden_channels)
        else:
            self.norm = Identity()

    def forward(self, data: HeteroData) -> Tensor:
        return self.norm(self.act(self.lin(data[self.node_type].x)))


class HeteroEmbedding(Module):
    class CombinedEmbedding(Module):
        def __init__(self, embeddings: List[Module]) -> None:
            super().__init__()
            self.embeddings = ModuleList(embeddings)
            combined_channels = sum(emb.output_channels for emb in embeddings)
            self.lin = Linear(combined_channels, embeddings[0].output_channels)

        def forward(self, data: HeteroData):
            return self.lin(cat([emb(data) for emb in self.embeddings]))

    def __init__(self, **embeddings: Union[Module, List[Module]]) -> None:
        super().__init__()
        self.embeddings = ModuleDict()
        for node_type, embedding in embeddings.items():
            if isinstance(embedding, list):
                self.embeddings[node_type] = HeteroEmbedding.CombinedEmbedding(
                    embedding
                )
            else:
                self.embeddings[node_type] = embedding

    def forward(self, data: HeteroData) -> NodeEmbedding:
        return {node_type: emb(data) for node_type, emb in self.embeddings.items()}
