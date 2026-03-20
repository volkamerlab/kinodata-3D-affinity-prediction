import logging

import torch
from torch import nn
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader  # Preferred for PyG datasets

logger = logging.getLogger(__name__)


def find_max_batch_size_inference(
    model: nn.Module,
    dataset: Dataset,
    device: str = "cuda",
    low: int = 16,
    high: int = 256,
    tolerance: int = 1,
) -> int:
    """
    Finds the maximum batch size that can be used without OOM error.

    Args:
        model: An instance of torch.nn.Module (PyG model).
        dataset: A torch_geometric Dataset.
        device: Device to use ('cuda' or 'cpu').
        low: Minimum batch size to test.
        high: Maximum batch size to test.
        tolerance: Stop when (high - low) <= tolerance.

    Returns:
        The largest safe batch size.
    """
    try:
        was_training = model.training
        model.eval()

        def test_batch_size(bs: int) -> bool:
            try:
                model = model.to(device)
                loader = DataLoader(dataset[:bs], batch_size=bs)
                batch = next(iter(loader)).to(device)
                with torch.inference_mode():
                    model(batch)
                del loader, batch
                torch.cuda.empty_cache()
                return True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return False
                else:
                    raise e

        while high - low > tolerance:
            mid = (low + high) // 2
            if test_batch_size(mid):
                low = mid
            else:
                high = mid - 1
    except RuntimeError as e:
        logger.warning(
            "RuntimeError encountered during batch size search: %s.",
            e,
        )
    finally:
        if was_training:
            model.train()

    return low
