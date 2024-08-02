from .complex_transformer import ComplexTransformer


class PairTransformer(RegressionModel):
    def __init__(
            self,
            config: Config,
            gnn: ComplexTransformer,
    ) -> None:
        super().__init__(config)
        self.gnn = gnn

    def forward(self, data: HeteroData) -> NodeEmbedding:
        self.gnn.interaction_module.set_node_type(NodeType.ComplexA)
        node_store = data[NodeType.ComplexA]
        node_repr_a = self.aggr(self.gnn.message_passing(data), node_store.batch)

        self.gnn.interaction_module.set_node_type(NodeType.ComplexB)
        node_store = data[NodeType.ComplexB]
        node_repr_b = self.aggr(self.gnn.message_passing(data), node_store.batch)

        return self.out(node_repr_a - node_repr_b)
