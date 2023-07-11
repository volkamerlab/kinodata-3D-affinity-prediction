from pathlib import Path
from typing import Callable
from typing import TypeVar

T = TypeVar("T")


def cached_execution(
    cache_path: Path,
    task: Callable[..., T],
    write_operation: Callable[[T, Path], None],
    read_operation: Callable[[Path], T],
) -> T:
    if cache_path.exists():
        return read_operation(cache_path)
    result = task()
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True)
    write_operation(result, cache_path)
    return result
