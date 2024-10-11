from typing import Type, Callable

from kinodata.data.dataset import KinodataDocked, apply_transform_instance_permament


def patch_pyg_kinodata3d_dataset(
    dataset: KinodataDocked,
    source_col_to_add: str,
    new_col_name: str | None = None,
    transform: Type | Callable | None = None,
) -> KinodataDocked:
    """Patch a KinodataDocked dataset to include a new column from the source dataframe.

    Args:
        dataset (KinodataDocked): The dataset instance to patch.
        source_col_to_add (str): Name of the column in the source dataframe to add.
        new_col_name (str | None, optional): Renames the new column if not None. Defaults to None.
        transform (Type | Callable | None, optional): Type to cast or transform to apply to the source column values. Defaults to None.

    Raises:
        ValueError: If the source column is not found in the source dataframe.

    Returns:
        KinodataDocked: The patched dataset instance.
    """
    new_col_name = new_col_name or source_col_to_add
    print(
        f"Attempting to patch {dataset} to include source column",
        f"'{source_col_to_add}' as data.{new_col_name}.",
    )
    df = dataset.df
    df = df.set_index("ident")
    if source_col_to_add not in df.columns:
        raise ValueError(f"Column '{source_col_to_add}' not found in source dataframe.")

    class RetrieveAndAddColumn:
        def __init__(self, source_col_to_add, new_col_name, transform):
            self.source_col_to_add = source_col_to_add
            self.new_col_name = new_col_name
            self.transform = transform

        def __call__(self, data):
            source_val = df.loc[data.ident.item(), self.source_col_to_add]
            if self.transform:
                source_val = self.transform(source_val)
            data[self.new_col_name] = source_val
            return data

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({self.source_col_to_add}, {self.new_col_name}, transform={self.transform})"

    return apply_transform_instance_permament(
        dataset,
        RetrieveAndAddColumn(source_col_to_add, new_col_name, transform),
    )
