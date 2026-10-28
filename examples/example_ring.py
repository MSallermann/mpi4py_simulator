"""Send one token per rank around a simulated communication ring."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from mpi4py_simulator import Comm, Simulator


@dataclass(frozen=True, slots=True)
class Token:
    """A token's origin rank and the ranks it has visited."""

    origin: int
    path: tuple[int, ...]


def ring_rank(comm: Comm, *, laps: int, hop_seconds: float) -> Token:
    """Circulate this rank's token clockwise and return it after complete laps."""
    predecessor = (comm.rank - 1) % comm.size
    successor = (comm.rank + 1) % comm.size
    token = Token(origin=comm.rank, path=(comm.rank,))

    comm.compute(1)

    for step in range(comm.size * laps):
        # Posting the send before receiving lets every rank use the same code without
        # depending on buffered blocking-send behavior.
        sending = comm.isend(token, dest=successor, tag=step)
        received = comm.recv(source=predecessor, tag=step)
        sending.wait()

        if hop_seconds:
            comm.compute(hop_seconds)
        token = Token(
            origin=received.origin,
            path=(*received.path, comm.rank),
        )

    return token


def _expected_path(origin: int, *, size: int, laps: int) -> tuple[int, ...]:
    return tuple((origin + step) % size for step in range(size * laps + 1))


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


def _nonnegative_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        msg = "must be a number"
        raise argparse.ArgumentTypeError(msg) from error
    if not math.isfinite(result) or result < 0:
        msg = "must be nonnegative"
        raise argparse.ArgumentTypeError(msg)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=_positive_integer, default=8)
    parser.add_argument("--laps", type=_positive_integer, default=1)
    parser.add_argument(
        "--latency-us",
        type=_nonnegative_float,
        default=2.0,
        help="one-way message latency in microseconds",
    )
    parser.add_argument(
        "--hop-us",
        type=_nonnegative_float,
        default=0.0,
        help="modeled computation at each token hop in microseconds",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        help="optional path for versioned simulation-statistics JSON",
    )
    return parser


def main() -> None:
    """Run the ring, verify every token path, and print a short summary."""
    args = _build_parser().parse_args()
    result = Simulator(
        size=args.ranks,
        latency=args.latency_us / 1_000_000,
    ).run(
        lambda comm: ring_rank(
            comm,
            laps=args.laps,
            hop_seconds=args.hop_us / 1_000_000,
        ),
    )

    for rank, token in enumerate(result.rank_results):
        expected = _expected_path(rank, size=args.ranks, laps=args.laps)
        if token != Token(origin=rank, path=expected):
            msg = f"rank {rank} received an invalid token path: {token}"
            raise RuntimeError(msg)

    first = result.rank_results[0]
    path = " -> ".join(map(str, first.path))
    print(f"verified {args.ranks} tokens across {args.laps} lap(s)")
    print(f"rank 0 token path: {path}")
    print(f"messages sent: {result.stats.messages_sent}")
    print(f"simulated runtime: {result.time:.9f} s")

    if args.stats is not None:
        args.stats.write_text(result.stats.to_json(indent=2) + "\n", encoding="utf-8")
        print(f"wrote statistics to {args.stats}")


if __name__ == "__main__":
    main()
