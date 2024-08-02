from functools import partial

from torch_geometric.data import HeteroData

from .complex_transformer import ComplexTransformer
from .complex_transformer import make_model as make_cgnn
from .regression import RegressionModel
from ..types import NodeEmbedding, NodeType

from kinodata.configuration import Config


class PairTransformer(RegressionModel):
    def __init__(
        self,
        config: Config,
        gnn: ComplexTransformer = None,
    ) -> None:
        super().__init__(config)
        self.gnn = make_cgnn(config) if gnn is None else gnn

    def forward(self, data: HeteroData) -> NodeEmbedding:
        self.gnn.set_node_type(NodeType.ComplexA)
        node_store = data[NodeType.ComplexA]
        node_repr_a = self.gnn.aggr(self.gnn.message_passing(data), node_store.batch)

        self.gnn.set_node_type(NodeType.ComplexB)
        node_store = data[NodeType.ComplexB]
        node_repr_b = self.gnn.aggr(self.gnn.message_passing(data), node_store.batch)

        return self.out(node_repr_a - node_repr_b)


def make_model(config: Config):
    cls = partial(PairTransformer, config)
    return config.init(cls)
