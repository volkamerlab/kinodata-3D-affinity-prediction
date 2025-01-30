from kinodata.data.dataset import KinodataDocked
from pathlib import Path


if __name__ == "__main__":
    dataset = KinodataDocked()
    processed_dir = Path(dataset.processed_dir)
    file = processed_dir / "kinodata3d.csv"
    if not file.exists():
        dataset.df.to_csv(file)
