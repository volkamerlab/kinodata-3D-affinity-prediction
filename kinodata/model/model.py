from typing import Any, Dict, List, Optional

import pytorch_lightning as pl

from kinodata.model.egnn import EGNN
from kinodata.typing import NodeType, EdgeType


class Model(pl.LightningModule):
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        hidden_channels: int,
        num_mp_layers: int,
        act: str,
        lr: float,
        batch_size: int,
        weight_decay: float,
        mp_type: str,
        mp_kwargs: Optional[Dict[str, Any]] = None,
        use_bonds: bool = False,
    ) -> None:
        super().__init__()

        # dirty
        edge_attr_size = {("ligand", "interacts", "ligand"): 4} if use_bonds else dict()

        self.egnn = EGNN(
            edge_attr_size=edge_attr_size,
            hidden_channels=hidden_channels,
            final_embedding_size=hidden_channels,
            target_size=1,
            num_mp_layers=num_mp_layers,
            act=act,
            node_types=node_types,
            edge_types=edge_types,
        )

    def load_pretrained_encoder(self, pretrained_egnn: EGNN):
        self.egnn.load_state_dict(pretrained_egnn.state_dict())
