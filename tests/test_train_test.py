import pandas as pd
from kinodata.model.complex_transformer import make_model, ComplexTransformer
from kinodata.data.dataset import KinodataDocked
from kinodata.transform import TransformToComplexGraph
from kinodata.training.predict import predict_df
import kinodata.configuration as cfg

from torch_geometric.loader import DataLoader


def test_complex_transformer():

    cfg.register(
        "sparse_transformer",
        max_num_neighbors=16,
        hidden_channels=32,
        num_attention_blocks=2,
        num_heads=2,
        act="relu",
        edge_attr_size=4,
        ln1=False,
        ln2=False,
        ln3=False,
        graph_norm=False,
        interaction_modes=["covalent", "structural"],
    )
    config = cfg.get("data", "training", "sparse_transformer")
    config["need_distances"] = False
    config["perturb_ligand_positions"] = 0.0
    config["perturb_pocket_positions"] = 0.0
    config["perturb_complex_positions"] = 0.1
    model = make_model(config)

    to_cplx = TransformToComplexGraph()
    dataset = KinodataDocked(transform=to_cplx)
    dataset = dataset[420:520]

    reference_df = pd.DataFrame(
        [
            {
                "target": data.y.item(),
                "chembl_activity_id": int(data.chembl_activity_id),
            }
            for data in dataset
        ]
    )

    loader = DataLoader(
        dataset,
        batch_size=20,
    )

    df = predict_df(model, loader)
    print(df.dtypes)
    print(df.shape)
    print(df.head())
    assert df.shape[0] == 100

    validation = pd.merge(df, reference_df, on="chembl_activity_id")
    assert validation.shape[0] == 100

    # check that targets match
    assert (validation["target_x"] - validation["target_y"]).abs().max() < 1e-6


if __name__ == "__main__":
    test_complex_transformer()
