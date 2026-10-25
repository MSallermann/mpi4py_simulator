"""Run one blocking MPI-style program with real or simulated communicators."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from mpi4py_simulator import ANY_SOURCE, Comm, Simulator


class Communicator(Protocol):
    """The small part of mpi4py's communicator used by this example."""

    def Get_rank(self) -> int: ...  # noqa: D102

    def Get_size(self) -> int: ...  # noqa: D102

    def send(self, value: Any, dest: int, tag: int = 0) -> None: ...  # noqa: D102

    def recv(self, source: int = -1, tag: int = -1) -> Any: ...  # noqa: D102


@dataclass(frozen=True)
class Task:
    """A value and its measured or estimated work duration."""

    value: int
    seconds: float


def rank_main(
    comm: Communicator,
    work: Callable[[Task], int],
) -> list[tuple[int, int]] | None:
    """Application code shared by the real and virtual executions."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        for worker in range(1, size):
            comm.send(Task(worker, 0.0001 * worker), dest=worker, tag=10)
        results = [comm.recv(source=ANY_SOURCE, tag=11) for _ in range(size - 1)]
        return sorted(results)

    task = comm.recv(source=0, tag=10)
    result = work(task)
    comm.send((rank, result), dest=0, tag=11)
    return None


def run_virtual(size: int) -> None:
    """Run the application with modeled work and print its virtual result."""

    def virtual_rank(comm: Comm) -> list[tuple[int, int]] | None:
        def modeled_work(task: Task) -> int:
            comm.compute(task.seconds)
            return task.value**2

        return rank_main(comm, modeled_work)

    result = Simulator(size, latency=2e-6).run(virtual_rank)
    print(f"virtual ranks: {size}")
    print(f"virtual makespan: {result.time:.6f} s")
    print(f"messages: {result.stats.messages_sent}")
    print(f"first results: {result.rank_results[0][:3]}")


def main() -> None:
    """Parse the example's command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, default=8)
    args = parser.parse_args()
    run_virtual(args.ranks)


if __name__ == "__main__":
    main()
