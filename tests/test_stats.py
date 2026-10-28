from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from mpi4py_simulator import Comm, SimulationStats, Simulator, TraceEvent
from mpi4py_simulator.visualization import main, render_stats_html

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
    assert document["version"] == 2
    values = document["stats"]
    assert isinstance(values, dict)
    assert values["rank_finish_seconds"] == [1.0, 2.0]
    assert values["trace_events"] == []
    assert SimulationStats.from_dict(document) == stats
    assert SimulationStats.from_json(stats.to_json(indent=2)) == stats


def test_simulation_stats_reject_unsupported_version() -> None:
    document = deepcopy(_stats().to_dict())
    document["version"] = 3

    with pytest.raises(ValueError, match="unsupported statistics format version"):
        SimulationStats.from_dict(document)


def test_simulation_stats_loads_version_one_without_trace() -> None:
    document = deepcopy(_stats().to_dict())
    document["version"] = 1
    values = document["stats"]
    assert isinstance(values, dict)
    del values["trace_events"]

    restored = SimulationStats.from_dict(document)

    assert restored.trace_events == ()


def test_simulation_stats_serializes_trace_events() -> None:
    stats = _stats()
    traced = SimulationStats(
        stats.time,
        stats.messages_sent,
        stats.messages_received,
        stats.pending_messages,
        stats.rank_finish_seconds,
        stats.rank_compute_seconds,
        stats.rank_python_seconds,
        stats.rank_communication_seconds,
        stats.rank_compute_idle_seconds,
        stats.rank_last_compute_finish_seconds,
        (TraceEvent(0, "compute", 0.0, 1.0, "modeled compute"),),
    )

    assert SimulationStats.from_json(traced.to_json()) == traced


def test_simulation_stats_rejects_invalid_trace_event() -> None:
    document = deepcopy(_stats().to_dict())
    values = document["stats"]
    assert isinstance(values, dict)
    values["trace_events"] = [
        {
            "rank": 0,
            "kind": "network",
            "start": 0.0,
            "finish": 1.0,
            "detail": "unsupported",
        },
    ]

    with pytest.raises(ValueError, match="unsupported kind"):
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


def test_render_stats_html() -> None:
    report = render_stats_html(_stats(), max_ranks=2)

    assert report.startswith("<!doctype html>")
    assert "Simulation statistics" in report
    assert 'id="filter"' in report
    assert "Waiting / other" in report
    assert '"rank_count":2' in report
    assert '"rank":0' in report
    assert "trace unavailable" in report


def test_render_stats_html_embeds_compute_window_diagnostics() -> None:
    def program(comm: Comm) -> None:
        comm.compute(1.0)
        comm.barrier()
        if comm.rank == 0:
            comm.compute(1.0)

    report = render_stats_html(
        Simulator(2, collective_latency=0.5).run(program).stats,
    )

    assert '"gaps":0.5' in report
    assert '"tail":0.5' in report
    assert '"other":0.5' in report


def test_render_stats_html_embeds_trace_timeline() -> None:
    def program(comm: Comm) -> None:
        comm.compute(float(comm.rank + 1))
        comm.barrier()

    stats = Simulator(2, collective_latency=0.5, trace=True).run(program).stats
    report = render_stats_html(stats)

    assert '"trace_event_count":4' in report
    assert '"kind":"compute"' in report
    assert '"kind":"collective"' in report
    assert "Event timeline" in report
    assert "Trace events" in report


def test_visualization_cli_writes_html_and_opens_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "stats.json"
    input_path.write_text(_stats().to_json(), encoding="utf-8")
    opened: list[tuple[str, int]] = []

    def open_browser(url: str, new: int = 0) -> bool:
        opened.append((url, new))
        return True

    monkeypatch.setattr("mpi4py_simulator.visualization.webbrowser.open", open_browser)

    main([str(input_path)])

    output_path = input_path.with_suffix(".html")
    report = output_path.read_text(encoding="utf-8")
    assert report.startswith("<!doctype html>")
    assert opened == [(output_path.resolve().as_uri(), 2)]


def test_visualization_cli_can_skip_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "stats.json"
    input_path.write_text(_stats().to_json(), encoding="utf-8")

    def fail_to_open(_url: str, new: int = 0) -> bool:
        pytest.fail(f"browser unexpectedly opened with new={new}")

    monkeypatch.setattr("mpi4py_simulator.visualization.webbrowser.open", fail_to_open)

    main([str(input_path), "--no-open"])


def test_visualization_cli_rejects_non_html_output(tmp_path: Path) -> None:
    input_path = tmp_path / "stats.json"
    input_path.write_text(_stats().to_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="must use an .html"):
        main([str(input_path), "--output", str(tmp_path / "stats.svg")])
