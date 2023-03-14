import multiprocessing as mp
import os
from typing import Callable, List, Optional, TypeVar

import pandas as pd
from tqdm import tqdm

T = TypeVar("T")

# adapted from
# https://stackoverflow.com/questions/40357434/pandas-df-iterrows-parallelization
def map_over_dataframe(
    func: Callable[[pd.DataFrame], List[T]],
    df: pd.DataFrame,
    num_processes: Optional[int] = None,
) -> List[T]:
    num_processes = num_processes if num_processes else os.cpu_count()
    num_total = df.shape[0]
    chunk_size = int(num_total / num_processes)
    chunks = [
        df.iloc[i : min(i + chunk_size, num_total - 1)]
        for i in range(0, num_total, chunk_size)
    ]

    with mp.Pool(num_processes) as pool:
        results: List[List[T]] = pool.map(func, tqdm(chunks))

    return [item for sublist in results for item in sublist]


# needs to be defined module-level due to multiprocessing pickling
def _test_func(chunk: pd.DataFrame) -> List[str]:
    return [f"Div3: {x['b']}, G8: {x['c']}" for _, x in chunk.iterrows()]


def _test():
    df = pd.DataFrame({"a": list(range(10000))})
    df["b"] = (df["a"] % 3 == 0).astype(int)
    df["c"] = (df["a"] > 8).astype(int)

    print(df)
    for s in map_over_dataframe(_test_func, df, 8):
        print(s)


if __name__ == "__main__":
    _test()
