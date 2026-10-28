"""Measure simulator timing inputs with a real mpi4py communicator."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Sequence

_MINIMUM_SAMPLES = 2


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """Measurements for one representative Python payload.

    Args:
        serialized_bytes: Number of bytes produced by ``MPI.pickle.dumps``.
        serialization_seconds: Mean time to pickle the payload once on rank zero.
        deserialization_seconds: Mean time to unpickle the payload once on rank zero.
        transport_one_way_seconds: Mean one-way time of a raw-buffer MPI ping-pong.
        object_one_way_seconds: Mean one-way time of a lowercase object ping-pong.
        predicted_one_way_seconds: One-way object time predicted by the fitted
            simulator settings.

    """

    serialized_bytes: int
    serialization_seconds: float
    deserialization_seconds: float
    transport_one_way_seconds: float
    object_one_way_seconds: float
    predicted_one_way_seconds: float


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Fitted simulator settings and the measurements behind them.

    Args:
        ranks: Number of MPI ranks participating in calibration. Point-to-point
            measurements use ranks zero and one; the barrier uses every rank.
        latency: Intercept fitted to raw-buffer one-way ping-pong time.
        send_overhead: Fixed intercept fitted to serialization on rank zero.
        receive_overhead: Fixed intercept fitted to deserialization on rank zero.
        collective_latency: Mean time of a barrier across all participating ranks.
        bandwidth: Raw-buffer point-to-point bytes per second.
        serialization_bandwidth: Pickled bytes produced per second on rank zero.
        deserialization_bandwidth: Pickled bytes consumed per second on rank zero.
        samples: Per-payload measurements and model predictions.

    Notes:
        A ping-pong cannot uniquely separate MPI endpoint overhead from network
        latency. This calibration assigns the raw-buffer intercept to ``latency``
        and the fixed pickle costs to ``send_overhead`` and ``receive_overhead``.
        Validate the predicted object times in :attr:`samples` before extrapolating.

    """

    ranks: int
    latency: float
    send_overhead: float
    receive_overhead: float
    collective_latency: float
    bandwidth: float
    serialization_bandwidth: float
    deserialization_bandwidth: float
    samples: tuple[CalibrationSample, ...]

    @property
    def simulator_kwargs(self) -> dict[str, float]:
        """Return timing arguments that can be passed directly to ``Simulator``.

        The returned dictionary excludes ``size`` and ``measure_python`` because
        those describe the virtual experiment rather than the calibrated MPI run.
        """
        return {
            "latency": self.latency,
            "send_overhead": self.send_overhead,
            "receive_overhead": self.receive_overhead,
            "collective_latency": self.collective_latency,
            "bandwidth": self.bandwidth,
            "serialization_bandwidth": self.serialization_bandwidth,
            "deserialization_bandwidth": self.deserialization_bandwidth,
        }

    def format_simulator(self, *, size: int | str = "...") -> str:
        """Format the measured settings as a readable ``Simulator`` call.

        Args:
            size: Virtual rank count to display. The default placeholder emphasizes
                that calibration rank count and simulation rank count are different.

        Returns:
            A multiline Python expression suitable for copying into an experiment.

        """
        lines = [f"Simulator(size={size},"]
        lines.extend(
            f"    {name}={value:.9g}," for name, value in self.simulator_kwargs.items()
        )
        lines.append(")")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class _CalibrationFailure:
    message: str


def calibrate(
    comm: Any,
    payloads: Sequence[Any] | None,
    *,
    iterations: int = 100,
    warmups: int = 10,
) -> CalibrationResult:
    """Measure timing inputs using a real mpi4py communicator.

    Args:
        comm: Real mpi4py communicator containing at least two ranks. Ranks zero and
            one perform point-to-point measurements. All ranks measure a barrier.
        payloads: Representative Python objects supplied on rank zero. Other ranks
            may pass ``None``. At least two objects with different serialized sizes
            are required so fixed costs and byte rates can be fitted.
        iterations: Number of timed repetitions for every payload and operation.
            Increase this when measurements are noisy; runtime grows proportionally.
        warmups: Untimed repetitions before each measurement. These initialize MPI
            paths, memory allocation, and pickle machinery.

    Returns:
        The fitted settings on every participating rank. Usually only rank zero
        prints or saves the result.

    Raises:
        TypeError: If iteration counts are not integers.
        ValueError: If fewer than two ranks participate, counts are invalid, or the
            payloads do not contain at least two different serialized sizes.
        RuntimeError: If measurement noise produces a nonpositive fitted byte rate.

    Notes:
        Run this function under ``mpiexec`` on the target hardware using the same
        process placement intended for the real application. It measures lowercase
        mpi4py object communication. It does not model congestion, eager/rendezvous
        transitions, or differing performance between coordinator and worker nodes.

    """
    _validate_count("iterations", iterations, minimum=1)
    _validate_count("warmups", warmups, minimum=0)
    if comm.Get_size() < _MINIMUM_SAMPLES:
        msg = "calibration requires at least two MPI ranks"
        raise ValueError(msg)

    from mpi4py import MPI  # Imported lazily; simulation itself does not require MPI.

    root_payloads = tuple(payloads) if comm.Get_rank() == 0 and payloads else None
    values = comm.bcast(root_payloads, root=0)
    if values is None or len(values) < _MINIMUM_SAMPLES:
        msg = "rank zero must provide at least two representative payloads"
        raise ValueError(msg)

    pickle_measurements: list[tuple[int, float, float]] = []
    if comm.Get_rank() == 0:
        pickle_measurements = [
            _measure_pickle(MPI, value, iterations, warmups) for value in values
        ]
    pickle_measurements = comm.bcast(pickle_measurements, root=0)

    sizes = [measurement[0] for measurement in pickle_measurements]
    if len(set(sizes)) < _MINIMUM_SAMPLES:
        msg = "payloads must have at least two different serialized sizes"
        raise ValueError(msg)

    transport_times = [
        _measure_raw_pingpong(comm, MPI, size, iterations, warmups) for size in sizes
    ]
    object_times = [
        _measure_object_pingpong(comm, MPI, value, iterations, warmups)
        for value in values
    ]
    collective_latency = _measure_barrier(comm, MPI, iterations, warmups)

    outcome: CalibrationResult | _CalibrationFailure | None = None
    if comm.Get_rank() == 0:
        try:
            outcome = _fit_result(
                comm.Get_size(),
                pickle_measurements,
                transport_times,
                object_times,
                collective_latency,
            )
        except RuntimeError as error:
            outcome = _CalibrationFailure(str(error))
    outcome = comm.bcast(outcome, root=0)
    if isinstance(outcome, _CalibrationFailure):
        _raise_calibration_failure(outcome.message)
    if outcome is None:  # pragma: no cover - defensive against a broken communicator
        msg = "rank zero did not produce a calibration result"
        raise RuntimeError(msg)
    return outcome


def _raise_calibration_failure(message: str) -> NoReturn:
    msg = f"calibration failed: {message}"
    raise RuntimeError(msg)


def _measure_pickle(
    mpi: Any,
    value: Any,
    iterations: int,
    warmups: int,
) -> tuple[int, float, float]:
    data = mpi.pickle.dumps(value)
    for _ in range(warmups):
        data = mpi.pickle.dumps(value)
    started = mpi.Wtime()
    for _ in range(iterations):
        data = mpi.pickle.dumps(value)
    serialization = (mpi.Wtime() - started) / iterations

    restored: Any = None
    for _ in range(warmups):
        restored = mpi.pickle.loads(data)
    started = mpi.Wtime()
    for _ in range(iterations):
        restored = mpi.pickle.loads(data)
    deserialization = (mpi.Wtime() - started) / iterations
    del restored
    return len(data), serialization, deserialization


def _measure_raw_pingpong(
    comm: Any,
    mpi: Any,
    size: int,
    iterations: int,
    warmups: int,
) -> float:
    sent = bytearray(size)
    received = bytearray(size)
    comm.Barrier()
    _raw_pingpong_rounds(comm, mpi, sent, received, warmups)
    comm.Barrier()
    started = mpi.Wtime()
    _raw_pingpong_rounds(comm, mpi, sent, received, iterations)
    elapsed = mpi.Wtime() - started if comm.Get_rank() == 0 else 0.0
    comm.Barrier()
    return comm.bcast(elapsed / (2 * iterations), root=0)


def _raw_pingpong_rounds(
    comm: Any,
    mpi: Any,
    sent: bytearray,
    received: bytearray,
    rounds: int,
) -> None:
    rank = comm.Get_rank()
    for _ in range(rounds):
        if rank == 0:
            comm.Send([sent, mpi.BYTE], dest=1, tag=27100)
            comm.Recv([received, mpi.BYTE], source=1, tag=27101)
        elif rank == 1:
            comm.Recv([received, mpi.BYTE], source=0, tag=27100)
            comm.Send([sent, mpi.BYTE], dest=0, tag=27101)


def _measure_object_pingpong(
    comm: Any,
    mpi: Any,
    value: Any,
    iterations: int,
    warmups: int,
) -> float:
    comm.Barrier()
    _object_pingpong_rounds(comm, value, warmups)
    comm.Barrier()
    started = mpi.Wtime()
    _object_pingpong_rounds(comm, value, iterations)
    elapsed = mpi.Wtime() - started if comm.Get_rank() == 0 else 0.0
    comm.Barrier()
    return comm.bcast(elapsed / (2 * iterations), root=0)


def _object_pingpong_rounds(comm: Any, value: Any, rounds: int) -> None:
    rank = comm.Get_rank()
    for _ in range(rounds):
        if rank == 0:
            comm.send(value, dest=1, tag=27200)
            comm.recv(source=1, tag=27201)
        elif rank == 1:
            received = comm.recv(source=0, tag=27200)
            comm.send(received, dest=0, tag=27201)


def _measure_barrier(
    comm: Any,
    mpi: Any,
    iterations: int,
    warmups: int,
) -> float:
    for _ in range(warmups):
        comm.Barrier()
    comm.Barrier()
    started = mpi.Wtime()
    for _ in range(iterations):
        comm.Barrier()
    local_average = (mpi.Wtime() - started) / iterations
    return float(comm.allreduce(local_average, op=mpi.MAX))


def _fit_result(
    ranks: int,
    pickle_measurements: Sequence[tuple[int, float, float]],
    transport_times: Sequence[float],
    object_times: Sequence[float],
    collective_latency: float,
) -> CalibrationResult:
    sizes = [float(measurement[0]) for measurement in pickle_measurements]
    serialization_times = [measurement[1] for measurement in pickle_measurements]
    deserialization_times = [measurement[2] for measurement in pickle_measurements]
    latency, bandwidth = _fit_cost(sizes, transport_times, "transport")
    send_overhead, serialization_bandwidth = _fit_cost(
        sizes,
        serialization_times,
        "serialization",
    )
    receive_overhead, deserialization_bandwidth = _fit_cost(
        sizes,
        deserialization_times,
        "deserialization",
    )

    samples = tuple(
        CalibrationSample(
            serialized_bytes=int(size),
            serialization_seconds=serialization,
            deserialization_seconds=deserialization,
            transport_one_way_seconds=transport,
            object_one_way_seconds=object_time,
            predicted_one_way_seconds=(
                send_overhead
                + size / serialization_bandwidth
                + latency
                + size / bandwidth
                + receive_overhead
                + size / deserialization_bandwidth
            ),
        )
        for size, serialization, deserialization, transport, object_time in zip(
            sizes,
            serialization_times,
            deserialization_times,
            transport_times,
            object_times,
            strict=True,
        )
    )
    return CalibrationResult(
        ranks=ranks,
        latency=latency,
        send_overhead=send_overhead,
        receive_overhead=receive_overhead,
        collective_latency=collective_latency,
        bandwidth=bandwidth,
        serialization_bandwidth=serialization_bandwidth,
        deserialization_bandwidth=deserialization_bandwidth,
        samples=samples,
    )


def _fit_cost(
    sizes: Sequence[float],
    seconds: Sequence[float],
    component: str,
) -> tuple[float, float]:
    mean_size = sum(sizes) / len(sizes)
    mean_seconds = sum(seconds) / len(seconds)
    denominator = sum((size - mean_size) ** 2 for size in sizes)
    slope = (
        sum(
            (size - mean_size) * (duration - mean_seconds)
            for size, duration in zip(sizes, seconds, strict=True)
        )
        / denominator
    )
    if slope <= 0.0 or not math.isfinite(slope):
        msg = (
            f"{component} measurements did not produce a positive byte cost; "
            "increase --iterations or use a wider payload-size range"
        )
        raise RuntimeError(msg)
    intercept = max(0.0, mean_seconds - slope * mean_size)
    return intercept, 1.0 / slope


def _validate_count(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    if value < minimum:
        msg = f"{name} must be at least {minimum}"
        raise ValueError(msg)


def _parse_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        msg = "sizes must be comma-separated integers"
        raise argparse.ArgumentTypeError(msg) from error
    if len(sizes) < _MINIMUM_SAMPLES or any(size < 0 for size in sizes):
        msg = "sizes must contain at least two nonnegative integers"
        raise argparse.ArgumentTypeError(msg)
    return sizes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure mpi4py-simulator timing settings with a real MPI communicator."
        ),
    )
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=_parse_sizes("0,64,1024,16384,262144,1048576"),
        help="comma-separated byte-payload sizes",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of a Python configuration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run calibration from ``python -m mpi4py_simulator.calibration``.

    Args:
        argv: Optional command-line arguments without the executable name. ``None``
            reads arguments from :data:`sys.argv` through :mod:`argparse`.

    """
    try:
        from mpi4py import MPI
    except ImportError as error:  # pragma: no cover - depends on local installation
        msg = "calibration requires mpi4py; install mpi4py-simulator[calibration]"
        raise SystemExit(msg) from error

    args = _build_parser().parse_args(argv)
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
    if args.json:
        document = {
            "ranks": result.ranks,
            "settings": result.simulator_kwargs,
            "samples": [asdict(sample) for sample in result.samples],
        }
        print(json.dumps(document, indent=2))
        return

    print(result.format_simulator())
    print("\nMeasured object round trips (one-way seconds):")
    for sample in result.samples:
        print(
            f"  {sample.serialized_bytes:>9} bytes: "
            f"measured={sample.object_one_way_seconds:.6g}, "
            f"predicted={sample.predicted_one_way_seconds:.6g}",
        )


if __name__ == "__main__":
    main()
