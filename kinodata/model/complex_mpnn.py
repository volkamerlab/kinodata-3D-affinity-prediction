from typing import Any, Callable, Dict, List, Optional, Protocol

from torch import Tensor
from torch_geometric.data import HeteroData

from kinodata.types import NodeEmbedding
from kinodata.configuration import Config
from kinodata.model.regression import RegressionModel


class Embedding(Protocol):
    def __call__(self, data: HeteroData) -> NodeEmbedding:
        ...


class MessagePassing(Protocol):
    def __call__(
        self, data: HeteroData, initial_node_embedding: NodeEmbedding
    ) -> NodeEmbedding:
        ...


class Readout(Protocol):
    def __call__(self, data: HeteroData, node_embedding: NodeEmbedding) -> Tensor:
        ...


class MessagePassingModel(RegressionModel):
    def __init__(
        self,
        config: Config,
        embedding_cls: Callable[..., Embedding],
        message_passing_cls: Callable[..., MessagePassing],
        readout_cls: Callable[..., Readout],
    ) -> None:
        super().__init__(config)
        self.embedding = config.init(embedding_cls)
        self.message_passing = config.init(message_passing_cls)
        self.readout = config.init(readout_cls)

    def forward(self, batch) -> Tensor:
        node_embed = self.embedding(batch)
        node_embed = self.message_passing(batch, node_embed)
        return self.readout(node_embed, batch)
