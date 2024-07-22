from .complex_transformer import ComplexTransformer


class LigandPairModel(RegressionModel):
    def __init__(
            self,
            config: Config,
            gnn: ComplexTransformer,
    ) -> None:
        super().__init__(config)
        self.gnn = gnn

    def forward(self, data: HeteroData) -> NodeEmbedding:
        return
