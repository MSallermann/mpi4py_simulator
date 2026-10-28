"""Calibrate, run, and simulate the same MPI coordinator/worker workload."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from mpi4py_simulator import ANY_SOURCE, Comm, Simulator, Status

_CALIBRATION_FORMAT = "mpi4py-simulator.calibration"
_ACTUAL_FORMAT = "mpi4py-simulator.actual-run"
_DOCUMENT_VERSION = 1
_MINIMUM_RANKS = 2
_MINIMUM_CALIBRATION_SAMPLES = 2
_DEFAULT_CALIBRATION_SIZES = "0,64,1024,16384,262144,1048576"
_TIMING_SETTINGS = (
    "latency",
    "send_overhead",
    "receive_overhead",
    "collective_latency",
    "bandwidth",
    "serialization_bandwidth",
    "deserialization_bandwidth",
)


@dataclass(frozen=True, slots=True)
class Task:
    """One deterministic square operation and its modeled duration."""

    index: int
    value: int
    seconds: float


# rank-main-start
def rank_main(
    comm: Any,
    tasks: Sequence[Task],
    work: Callable[[Task], int],
    *,
    any_source: int,
    status_factory: Callable[[], Any],
) -> list[int] | None:
    """Run a dynamic coordinator/worker protocol with either communicator."""
    if comm.Get_rank() != 0:
        while True:
            task = comm.recv(source=0, tag=10)
            if task is None:
                return None
            comm.send((task.index, work(task)), dest=0, tag=11)

    results = [0] * len(tasks)
    next_task = 0
    active_workers: set[int] = set()
    for worker in range(1, comm.Get_size()):
        if next_task < len(tasks):
            comm.send(tasks[next_task], dest=worker, tag=10)
            next_task += 1
            active_workers.add(worker)
        else:
            comm.send(None, dest=worker, tag=10)

    while active_workers:
        status = status_factory()
        index, value = comm.recv(source=any_source, tag=11, status=status)
        results[index] = value
        worker = status.Get_source()
        if next_task < len(tasks):
            comm.send(tasks[next_task], dest=worker, tag=10)
            next_task += 1
        else:
            comm.send(None, dest=worker, tag=10)
            active_workers.remove(worker)
    return results


# rank-main-end


def _tasks(count: int, base_seconds: float) -> tuple[Task, ...]:
    return tuple(
        Task(
            index=index,
            value=index,
            seconds=base_seconds * (0.5 + (index % 7) / 6),
        )
        for index in range(count)
    )


def _run_calibration(args: argparse.Namespace) -> None:
    from mpi4py import MPI

    from mpi4py_simulator.calibration import calibrate

    payloads = (
        [bytes(size) for size in args.sizes] if MPI.COMM_WORLD.rank == 0 else None
    )
    result = calibrate(
        MPI.COMM_WORLD,
        payloads,
        iterations=args.iterations,
        warmups=args.warmups,
    )
    if MPI.COMM_WORLD.rank != 0:
        return
    document = {
        "format": _CALIBRATION_FORMAT,
        "version": _DOCUMENT_VERSION,
        "ranks": result.ranks,
        "settings": result.simulator_kwargs,
        "samples": [asdict(sample) for sample in result.samples],
    }
    _write_json(args.output, document)
    print(f"wrote calibration to {args.output}")


def _run_actual(args: argparse.Namespace) -> None:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() < _MINIMUM_RANKS:
        msg = "the actual run requires at least two MPI ranks"
        raise ValueError(msg)
    tasks = _tasks(args.tasks, args.task_ms / 1_000)

    comm.Barrier()
    started = MPI.Wtime()
    results = rank_main(
        comm,
        tasks,
        lambda task: _actual_work(task),
        any_source=MPI.ANY_SOURCE,
        status_factory=MPI.Status,
    )
    comm.Barrier()
    elapsed = MPI.Wtime() - started

    if comm.rank != 0:
        return
    document = {
        "format": _ACTUAL_FORMAT,
        "version": _DOCUMENT_VERSION,
        "ranks": comm.Get_size(),
        "tasks": args.tasks,
        "task_ms": args.task_ms,
        "elapsed_seconds": elapsed,
        "rank_zero_result": results,
    }
    _write_json(args.output, document)
    print(f"actual MPI runtime: {elapsed:.6f} s")
    print(f"wrote actual run to {args.output}")


def _actual_work(task: Task) -> int:
    time.sleep(task.seconds)
    return task.value**2


def _run_simulation(args: argparse.Namespace) -> None:
    actual = _read_json(args.actual)
    calibration = _read_json(args.calibration)
    _validate_document(actual, _ACTUAL_FORMAT)
    _validate_document(calibration, _CALIBRATION_FORMAT)

    ranks = _integer_field(actual, "ranks", minimum=2)
    task_count = _integer_field(actual, "tasks", minimum=1)
    task_ms = _number_field(actual, "task_ms", positive=True)
    actual_seconds = _number_field(actual, "elapsed_seconds", positive=True)
    calibration_ranks = _integer_field(calibration, "ranks", minimum=2)
    if calibration_ranks != ranks:
        msg = (
            f"calibration used {calibration_ranks} ranks but the actual run used "
            f"{ranks}; rerun calibration with the same placement"
        )
        raise ValueError(msg)
    settings = _timing_settings(calibration)
    tasks = _tasks(task_count, task_ms / 1_000)

    def simulated_rank(comm: Comm) -> list[int] | None:
        def work(task: Task) -> int:
            comm.compute(task.seconds)
            return task.value**2

        values = rank_main(
            comm,
            tasks,
            work,
            any_source=ANY_SOURCE,
            status_factory=Status,
        )
        comm.barrier()
        return values

    result = Simulator(
        size=ranks,
        measure_python=True,
        **settings,
    ).run(simulated_rank)
    expected = actual.get("rank_zero_result")
    if result.rank_results[0] != expected:
        msg = "real and simulated rank-zero results differ"
        raise RuntimeError(msg)

    args.stats.write_text(result.stats.to_json(indent=2) + "\n", encoding="utf-8")
    difference = result.time - actual_seconds
    percent = difference / actual_seconds * 100
    print(f"actual MPI runtime: {actual_seconds:.6f} s")
    print(f"simulated runtime:  {result.time:.6f} s")
    print(f"difference:         {difference:+.6f} s ({percent:+.1f}%)")
    print("rank-zero results:  identical")
    print(f"wrote simulated statistics to {args.stats}")


def _timing_settings(document: Mapping[str, object]) -> dict[str, float]:
    values = document.get("settings")
    if not isinstance(values, Mapping):
        msg = "calibration document must contain a 'settings' object"
        raise TypeError(msg)
    missing = set(_TIMING_SETTINGS) - set(values)
    if missing:
        msg = f"calibration settings are missing {sorted(missing)!r}"
        raise ValueError(msg)
    return {
        name: _number_field(values, name, positive=False) for name in _TIMING_SETTINGS
    }


def _validate_document(document: Mapping[str, object], expected_format: str) -> None:
    if document.get("format") != expected_format:
        msg = f"expected {expected_format!r}, got {document.get('format')!r}"
        raise ValueError(msg)
    if document.get("version") != _DOCUMENT_VERSION:
        msg = f"unsupported document version: {document.get('version')!r}"
        raise ValueError(msg)


def _integer_field(
    document: Mapping[str, object],
    name: str,
    *,
    minimum: int,
) -> int:
    value = document.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name!r} must be an integer"
        raise TypeError(msg)
    if value < minimum:
        msg = f"{name!r} must be at least {minimum}"
        raise ValueError(msg)
    return value


def _number_field(
    document: Mapping[str, object],
    name: str,
    *,
    positive: bool,
) -> float:
    value = document.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name!r} must be a number"
        raise TypeError(msg)
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        qualifier = "positive" if positive else "nonnegative"
        msg = f"{name!r} must be {qualifier}"
        raise ValueError(msg)
    return result


def _read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        msg = f"{path} must contain a JSON object"
        raise TypeError(msg)
    return cast("Mapping[str, object]", value)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _parse_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        msg = "sizes must be comma-separated integers"
        raise argparse.ArgumentTypeError(msg) from error
    if len(sizes) < _MINIMUM_CALIBRATION_SAMPLES or any(size < 0 for size in sizes):
        msg = "sizes must contain at least two nonnegative integers"
        raise argparse.ArgumentTypeError(msg)
    return sizes


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        msg = "must be an integer"
        raise argparse.ArgumentTypeError(msg) from error
    if result < 1:
        msg = "must be positive"
        raise argparse.ArgumentTypeError(msg)
    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        msg = "must be a number"
        raise argparse.ArgumentTypeError(msg) from error
    if not math.isfinite(result) or result <= 0.0:
        msg = "must be positive"
        raise argparse.ArgumentTypeError(msg)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    calibration = commands.add_parser("calibrate", help="measure simulator inputs")
    calibration.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=_parse_sizes(_DEFAULT_CALIBRATION_SIZES),
    )
    calibration.add_argument("--iterations", type=_positive_integer, default=100)
    calibration.add_argument("--warmups", type=int, default=10)
    calibration.add_argument("--output", type=Path, default=Path("calibration.json"))

    actual = commands.add_parser("actual", help="run the real MPI workload")
    actual.add_argument("--tasks", type=_positive_integer, default=64)
    actual.add_argument("--task-ms", type=_positive_float, default=5.0)
    actual.add_argument("--output", type=Path, default=Path("actual-run.json"))

    simulation = commands.add_parser("simulate", help="run and compare simulation")
    simulation.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration.json"),
    )
    simulation.add_argument("--actual", type=Path, default=Path("actual-run.json"))
    simulation.add_argument(
        "--stats",
        type=Path,
        default=Path("simulation-stats.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one stage of the comparison workflow."""
    args = _build_parser().parse_args(argv)
    if args.command == "calibrate":
        _run_calibration(args)
    elif args.command == "actual":
        _run_actual(args)
    else:
        _run_simulation(args)


if __name__ == "__main__":
    main()
