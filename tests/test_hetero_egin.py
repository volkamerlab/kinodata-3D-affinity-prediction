from kinodata.model.egin import HeteroEGIN
from kinodata.data.data_module import make_data_module
from kinodata.data.dataset import KinodataDocked
from kinodata.transform import AddDistancesAndInteractions
from kinodata.data.data_split import RandomSplit


def test_hetero_egin():
    dataset = KinodataDocked(
        transform=AddDistancesAndInteractions(5.0, distance_key="edge_weight")
    )
    data_module = make_data_module(dataset, RandomSplit(), 32, 0)

    batch = next(iter(data_module.train_dataloader()))
    hetero_gin = HeteroEGIN(
        ["ligand", "pocket"],
        [
            ("ligand", "interacts", "ligand"),
            ("pocket", "interacts", "ligand"),
            ("ligand", "interacts", "pocket"),
        ],
        32,
        2,
        5.0,
        edge_dim=4,
    )

    output = hetero_gin.encode(batch)

    print(output)


if __name__ == "__main__":
    test_hetero_egin()
