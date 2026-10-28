from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from mpi4py_simulator import Comm, SimulationStats, Simulator
from mpi4py_simulator.visualization import main, render_stats_svg

if TYPE_CHECKING:
    from pathlib import Path


def _stats() -> SimulationStats:
    def program(comm: Comm) -> None:
        comm.compute(float(comm.rank + 1))
        if comm.rank == 0:
            comm.send("done", dest=1)
        else:
            comm.recv(source=0)

    return Simulator(2, latency=0.25).run(program).stats


def test_simulation_stats_json_round_trip() -> None:
    stats = _stats()

    document = stats.to_dict()
    assert document["format"] == "mpi4py-simulator.stats"
    assert document["version"] == 1
    values = document["stats"]
    assert isinstance(values, dict)
    assert values["rank_finish_seconds"] == [1.0, 2.0]
    assert SimulationStats.from_dict(document) == stats
    assert SimulationStats.from_json(stats.to_json(indent=2)) == stats


def test_simulation_stats_reject_unsupported_version() -> None:
    document = deepcopy(_stats().to_dict())
    document["version"] = 2

    with pytest.raises(ValueError, match="unsupported statistics format version"):
        SimulationStats.from_dict(document)


def test_simulation_stats_reject_mismatched_rank_lengths() -> None:
    document = deepcopy(_stats().to_dict())
    values = document["stats"]
    assert isinstance(values, dict)
    values["rank_compute_seconds"] = [0.0]

    with pytest.raises(ValueError, match="equal lengths"):
        SimulationStats.from_dict(document)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("time", float("nan"), "finite and nonnegative"),
        ("messages_sent", -1, "must be nonnegative"),
        ("rank_finish_seconds", "not an array", "must be an array"),
    ],
)
def test_simulation_stats_reject_invalid_values(
    field: str,
    value: object,
    match: str,
) -> None:
    document = deepcopy(_stats().to_dict())
    values = document["stats"]
    assert isinstance(values, dict)
    values[field] = value

    with pytest.raises((TypeError, ValueError), match=match):
        SimulationStats.from_dict(document)


def test_render_stats_svg() -> None:
    svg = render_stats_svg(_stats(), width=800, max_ranks=2)

    assert svg.startswith("<svg")
    assert "Simulation statistics" in svg
    assert "rank 0" in svg
    assert "rank 1" in svg
    assert "Compute" in svg


def test_visualization_cli_writes_svg(tmp_path: Path) -> None:
    input_path = tmp_path / "stats.json"
    output_path = tmp_path / "stats.svg"
    input_path.write_text(_stats().to_json(), encoding="utf-8")

    main([str(input_path), "--output", str(output_path)])

    assert output_path.read_text(encoding="utf-8").startswith("<svg")
