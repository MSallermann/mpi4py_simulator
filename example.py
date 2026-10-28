"""Run ChemFit's normal MPI scheduler with virtual ranks in one process."""

from __future__ import annotations

import argparse
from typing import Any

import chemfit
from chemfit import mpi_graph_schedule
from chemfit.mpi_graph_schedule import MPIGraphScheduler

from mpi4py_simulator import MPI, Comm, Simulator

# ChemFit accepts ``comm`` directly, but uses this namespace for Status, wildcards,
# and Request.Waitall during shutdown. No ChemFit implementation is replaced.
mpi_graph_schedule.MPI = MPI


def rank_main(comm: Comm, tasks: int, task_seconds: float) -> list[Any] | None:
    """Run the same scheduler branches that a real MPI process would run."""

    @chemfit.component()
    def square(value: int) -> int:
        comm.compute(task_seconds)
        return value * value

    scheduler = MPIGraphScheduler(comm=comm)
    with scheduler.prepare(square) as schedule:
        if comm.rank != 0:
            schedule.worker_loop()
            return None

        requests = [
            chemfit.EvaluationRequest(value, chemfit.EvaluateContext())
            for value in range(tasks)
        ]

        completed = sorted(
            schedule.evaluate_many(requests), key=lambda item: item.index
        )
        return [item.value for item in completed]


def main() -> None:
    """Run the example and report modeled work and measured coordinator work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--tasks", type=int, default=1_024)
    parser.add_argument("--task-ms", type=float, default=1.0)
    args = parser.parse_args()

    simulation = Simulator(
        size=16,
        latency=1.4106714e-06,
        send_overhead=0,
        receive_overhead=5.62192819e-06,
        collective_latency=6.6768e-07,
        bandwidth=1.42672844e10,
        serialization_bandwidth=1.13014049e09,
        deserialization_bandwidth=2.26584621e10,
        measure_python=True,
    )
    result = simulation.run(
        lambda comm: rank_main(comm, args.tasks, args.task_ms / 1_000),
    )

    worker_busy = sum(result.stats.rank_compute_seconds[1:])
    utilization = worker_busy / (args.workers * result.time)
    print(f"completed: {len(result.rank_results[0])}")
    print(f"simulated runtime: {result.time:.6f} s")
    print(
        f"measured rank-0 Python: {result.stats.rank_python_seconds[0]:.6f} s",
    )
    print(f"modeled worker utilization: {utilization:.1%}")


if __name__ == "__main__":
    main()
