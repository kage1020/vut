import time

import numpy as np
import torch
from rich.box import HORIZONTALS
from rich.console import Console
from rich.table import Table

from vut import unique


def main():
    table = Table(box=HORIZONTALS)
    console = Console()
    table.add_column("Type")
    table.add_column("Time (s)", justify="right")
    table.add_column("Time/iter (ms)", justify="right")
    iter_count = 10000
    data_count = 10000
    data = list(range(data_count))

    start = time.perf_counter()
    target = data.copy()
    for i in range(iter_count):
        set(target)
    end_time = time.perf_counter() - start
    table.add_row("set", f"{end_time:.4f}", f"{end_time / iter_count * 1000:.4f}")

    start = time.perf_counter()
    target = np.array(data.copy())
    for i in range(iter_count):
        np.unique(target)
    end_time = time.perf_counter() - start
    table.add_row(
        "numpy.unique", f"{end_time:.4f}", f"{end_time / iter_count * 1000:.4f}"
    )

    start = time.perf_counter()
    target = torch.tensor(data.copy())
    for i in range(iter_count):
        torch.unique(target)
    end_time = time.perf_counter() - start
    table.add_row(
        "torch.unique", f"{end_time:.4f}", f"{end_time / iter_count * 1000:.4f}"
    )

    start = time.perf_counter()
    target = data.copy()
    for i in range(iter_count):
        unique(target)
    end_time = time.perf_counter() - start
    table.add_row(
        "implement(list)",
        f"{end_time:.4f}",
        f"{end_time / iter_count * 1000:.4f}",
    )

    start = time.perf_counter()
    target = np.array(data.copy())
    for i in range(iter_count):
        unique(target)
    end_time = time.perf_counter() - start
    table.add_row(
        "implement(ndarray)", f"{end_time:.4f}", f"{end_time / iter_count * 1000:.4f}"
    )

    start = time.perf_counter()
    target = torch.tensor(data.copy())
    for i in range(iter_count):
        unique(target)
    end_time = time.perf_counter() - start
    table.add_row(
        "implement(tensor)", f"{end_time:.4f}", f"{end_time / iter_count * 1000:.4f}"
    )

    console.print(f"iter_count: {iter_count}")
    console.print(f"data_count: {data_count}")
    console.print(table)


if __name__ == "__main__":
    main()
