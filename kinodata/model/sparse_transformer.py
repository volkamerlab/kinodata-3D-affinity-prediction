from torch import Tensor
from torch.nn import Module, Linear, LayerNorm, Sequential, Identity
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class SparseAttention(Module):
    """
    Sparse Query/Key/Value attention.
    """

    hidden_channels: int = 8
    num_heads: int = 4

    def __init__(self, hidden_channels, num_heads) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.projected_channels = hidden_channels // num_heads
        self.num_heads = num_heads
        self.lin_query = Linear(
            self.hidden_channels, self.projected_channels * self.num_heads, bias=False
        )
        self.lin_key_value = Linear(
            self.hidden_channels,
            self.projected_channels * self.num_heads * 2,
            bias=False,
        )
        self.lin_bias = Linear(
            self.hidden_channels,
            self.projected_channels * self.num_heads * 2,
            bias=False,
        )
        self.lin_out = Linear(
            self.projected_channels * self.num_heads, self.hidden_channels, bias=False
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        query_index: Tensor,
        key_index: Tensor,
        paired_repr: Tensor,
    ):
        """
        Perform sparse attention based on `M` Query/Key interactions.
        Interactions 0...M-1 are encoded by the aligned index tensors
        `query_index` and `key_index`.


        Parameters
        ----------
        query : Tensor
            (N, d)
        key : Tensor
            (N, d)
        query_index : Tensor
            (M,)
        key_index : Tensor
            (M,)
        paired_repr : Tensor
            (M, d)
        """
        N = query.size(0)
        M = paired_repr.size(0)
        d = self.projected_channels
        H = self.num_heads

        # (N, d) -> (N, d, H)
        query = self.lin_query(query).view(N, d, H)
        key, value = self.lin_key_value(key).chunk(2, dim=1)
        key = key.view(N, d, H)
        value = value.view(N, d, H)

        # (M,)
        query = query[query_index]
        key = key[key_index]
        value = value[key_index]

        # (M, d) -> ((M, d * H)
        bias_mul, bias_add = self.lin_bias(paired_repr).chunk(2, dim=1)

        # (M, d, H)
        bias_mul = bias_mul.view(M, d, H)
        bias_add = bias_add.view(M, d, H)

        # (M * H, d)  -> (M * H,)
        logit = (query * (key * (1 + bias_mul) + bias_add)).sum(dim=1)
        attention = softmax(logit, query_index, dim=0)
        result = scatter_add(attention.unsqueeze(1) * value, query_index, 0)
        result = result.view(N, -1)
        result = self.lin_out(result)
        pair_logits = logit.view(M, -1)
        return result, pair_logits


class SPAB(Module):
    """
    Self-attention with sparse topology and explicit interaction representations.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        act: Module,
        ln1: bool = True,
        ln2: bool = False,
        ln3: bool = True,
    ) -> None:
        super().__init__()
        self.attention = SparseAttention(hidden_channels, num_heads)
        self.ln1 = LayerNorm(hidden_channels) if ln1 else Identity()
        self.ff = Sequential(
            Linear(hidden_channels, hidden_channels),
            act,
            Linear(hidden_channels, hidden_channels),
        )
        self.ln2 = LayerNorm(hidden_channels) if ln2 else Identity()

        self.ff_edge = Sequential(
            Linear(num_heads, hidden_channels),
            act,
            Linear(hidden_channels, hidden_channels),
        )
        self.ln3 = LayerNorm(hidden_channels) if ln3 else Identity()

    def forward(self, x, edge_repr, edge_index):
        x_, pair_logits = self.attention(x, x, edge_index[0], edge_index[1], edge_repr)
        x = self.ln1(x_ + x)
        x = self.ln2(self.ff(x) + x)
        edge_repr = self.ln3(self.ff_edge(pair_logits) + edge_repr)
        return x, edge_repr
