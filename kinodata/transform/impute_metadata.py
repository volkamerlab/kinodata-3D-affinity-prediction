import torch
from torch_geometric.transforms import BaseTransform
from kinodata.data.dataset import process_raw_data, _DATA


class ImputeMetdata(BaseTransform):

    def __init__(self, remove_hydrogen: bool = True, **kwargs):
        super().__init__()
        print("Loading source df for metadata imputation. This may take a while.")
        self.source_df = process_raw_data(
            _DATA / "raw", remove_hydrogen=remove_hydrogen, **kwargs
        ).set_index("ident")

    def forward(self, data):
        row = self.source_df.loc[data["ident"].item()]
        data["chembl_activity_id"] = torch.tensor([int(row["activities.activity_id"])])
        data["klifs_structure_id"] = torch.tensor(
            [int(row["similar.klifs_structure_id"])]
        )
        return data
