from collections import defaultdict
from enum import Enum
from typing import Callable, Literal, TypeVar
import torch
from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.data import Data, InMemoryDataset

T = TypeVar("T")


class HANDLERS(Enum):
    raise_error = "raise_error"
    use_global = "use_global"


def add_raise_error_handler(norm: "GroupNormalizeTarget"):
    def get_group_mean(group: T) -> float:
        mean = norm._group_means.get(group, None)
        if mean is None:
            raise ValueError(f"Unknown group: {group}")
        return mean

    def get_group_std(group: T) -> float:
        std = norm._group_stds.get(group, None)
        if std is None:
            raise ValueError(f"Unknown group: {group}")
        return std

    norm.get_group_mean = get_group_mean
    norm.get_group_std = get_group_std
    return norm


def add_global_mean_handler(norm: "GroupNormalizeTarget"):
    global_mean = sum(norm._group_means.values()) / len(norm._group_means)
    global_std = sum(norm._group_stds.values()) / len(norm._group_stds)

    def get_group_mean(group: T) -> float:
        return norm._group_means.get(group, global_mean)

    def get_group_std(group: T) -> float:
        return norm._group_stds.get(group, global_std)

    norm.get_group_mean = get_group_mean
    norm.get_group_std = get_group_std
    return norm


class GroupNormalizeTarget(BaseTransform):
    group_key: str
    raw_target_key: str
    target_delta_key: str
    group_mean_key: str
    group_std_key: str
    scale: bool
    unknown_group_handler: Literal[HANDLERS.raise_error, HANDLERS.use_global] | None = (
        None
    )

    _group_means: dict[T, float] = dict()
    _group_stds: dict[T, float] = dict()

    def __init__(
        self,
        group_key: str,
        raw_target_key: str = "y",
        target_delta_key: str = "y_delta",
        group_mean_key: str = "y_group_mean",
        group_std_key: str = "y_group_std",
        scale: bool = False,
        unknown_group_handler: (
            Literal[HANDLERS.raise_error, HANDLERS.use_global] | None
        ) = None,
    ):
        super().__init__()
        self.group_key = group_key
        self.raw_target_key = raw_target_key
        self.target_delta_key = target_delta_key
        self.group_mean_key = group_mean_key
        self.group_std_key = group_std_key
        self.scale = scale
        self.unknown_group_handler = unknown_group_handler

        if self.unknown_group_handler == HANDLERS.raise_error:
            add_raise_error_handler(self)
        elif self.unknown_group_handler == HANDLERS.use_global:
            add_global_mean_handler(self)

    @classmethod
    def apply(
        cls,
        group_key: str,
        train_dataset: InMemoryDataset,
        *other_datasets: InMemoryDataset,
        raw_target_key: str = "y",
        target_delta_key: str = "y_delta",
        group_mean_key: str = "y_group_mean",
        group_std_key: str = "y_group_std",
        scale: bool = False,
        add_transform_inplace: bool = True,
    ):
        transform = cls(
            group_key,
            raw_target_key,
            target_delta_key,
            group_mean_key,
            group_std_key,
            scale,
        )
        transform._fit(train_dataset)

        dataset_with_transform = [train_dataset] + list(other_datasets)
        if not add_transform_inplace:
            dataset_with_transform = [
                dataset.clone() for dataset in dataset_with_transform
            ]
        for dataset in dataset_with_transform:
            existing_transform = dataset.transform
            t = transform
            if existing_transform is not None:
                t = Compose([existing_transform, t])
            dataset.transform = t
        return transform, dataset_with_transform

    def get_group_mean(self, group: T) -> float | None:
        return self._group_means.get(group) if group in self._group_means else None

    def get_group_std(self, group: T) -> float | None:
        return self._group_stds.get(group) if group in self._group_stds else None

    def get_target_value(self, data: Data) -> torch.Tensor:
        obj = getattr(data, self.raw_target_key)
        if isinstance(obj, float):
            return torch.tensor([obj])
        if isinstance(obj, list):
            return torch.tensor(obj)
        if isinstance(obj, torch.Tensor):
            return obj
        raise ValueError(f"Invalid target value type: {type(obj)}")

    def get_group(self, data: Data) -> T:
        obj = getattr(data, self.group_key)
        if isinstance(obj, list):
            return obj[0]
        if isinstance(obj, torch.Tensor):
            return obj.item()
        return obj

    def get_groups(self, data: Data) -> T:
        obj = getattr(data, self.group_key)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        raise ValueError(f"Invalid group value type: {type(obj)}")

    def _fit(self, dataset: list[Data] | InMemoryDataset):
        print("Fitting group target normalization..")
        target_by_group = defaultdict(list)
        for data in dataset:
            value = self.get_target_value(data).item()
            target_by_group[self.get_group(data)].append(value)

        target_by_group = {
            group: torch.tensor(values) for group, values in target_by_group.items()
        }
        self._group_means = {
            group: values.mean().item() for group, values in target_by_group.items()
        }
        print(f"\t {len(self._group_means)} groups encountered.")
        if self.scale:
            self._group_stds = {
                group: values.std().item() for group, values in target_by_group.items()
            }

    def forward(self, data: Data):
        group = self.get_group(data)
        group_mean = torch.tensor(self.get_group_mean(group))
        setattr(data, self.group_mean_key, group_mean)
        if self.scale:
            group_std = torch.tensor(self.get_group_std(group))
            setattr(data, self.group_std_key, group_std)

        normalized_target = self.get_target_value(data) - group_mean
        if self.scale:
            normalized_target /= group_std
        setattr(data, self.target_delta_key, normalized_target)
        return data

    def get_raw_predictor(
        self,
        delta_model: torch.nn.Module | Callable[[Data], torch.Tensor],
        expect_transformed_data: bool = False,
    ) -> torch.Tensor:
        def predict_raw(data: Data) -> torch.Tensor:
            if expect_transformed_data:
                group_mean = getattr(data, self.group_mean_key)
            else:
                groups = self.get_groups(data)
                group_mean = torch.tensor(
                    [self.get_group_mean(group) for group in groups]
                )
            pred_delta = delta_model(data)
            pred = group_mean + pred_delta
            if self.scale:
                if expect_transformed_data:
                    group_std = getattr(data, self.group_std_key)
                else:
                    group_std = torch.tensor(
                        [self.get_group_std(group) for group in groups]
                    )
                pred = pred * group_std
            return pred

        return predict_raw
