"""Render saved simulation statistics as a self-contained HTML report."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from ._engine import SimulationStats

_DEFAULT_MAX_RANKS = 64


def render_stats_html(
    stats: SimulationStats,
    *,
    max_ranks: int = _DEFAULT_MAX_RANKS,
) -> str:
    """Render aggregate simulation statistics as an interactive HTML report.

    Args:
        stats: Completed simulation statistics.
        max_ranks: Maximum rank rows to include. When a simulation has more ranks,
            rank zero and the slowest remaining ranks are retained.

    Returns:
        A standalone UTF-8 HTML document with embedded data, styles, and JavaScript.
        It can be opened directly without a server or external dependencies.

    Raises:
        TypeError: If ``max_ranks`` is not an integer.
        ValueError: If ``max_ranks`` is not positive.

    Notes:
        ``Waiting / other``, first-compute delay, gaps, and tail are derived from
        aggregate counters. They are useful diagnostics but not an event trace.

    """
    _validate_max_ranks(max_ranks)
    ranks = _select_ranks(stats, max_ranks)
    report = {
        "time": stats.time,
        "rank_count": len(stats.rank_finish_seconds),
        "displayed_rank_count": len(ranks),
        "messages_sent": stats.messages_sent,
        "messages_received": stats.messages_received,
        "pending_messages": stats.pending_messages,
        "ranks": [_rank_record(stats, rank) for rank in ranks],
    }
    # Keep the embedded JSON inert if future formats gain user-controlled strings.
    payload = json.dumps(report, separators=(",", ":"), allow_nan=False).replace(
        "<",
        "\\u003c",
    )
    return _REPORT_TEMPLATE.replace("__REPORT_DATA__", payload)


def _rank_record(stats: SimulationStats, rank: int) -> dict[str, float | int | bool]:
    finish = stats.rank_finish_seconds[rank]
    compute = stats.rank_compute_seconds[rank]
    python = stats.rank_python_seconds[rank]
    communication = stats.rank_communication_seconds[rank]
    gaps = stats.rank_compute_idle_seconds[rank]
    has_compute = compute > 0.0
    last_compute = (
        min(finish, stats.rank_last_compute_finish_seconds[rank])
        if has_compute
        else 0.0
    )
    first_compute = max(0.0, last_compute - compute - gaps) if has_compute else 0.0
    tail = max(0.0, finish - last_compute)
    other = max(0.0, finish - compute - python - communication)
    return {
        "rank": rank,
        "finish": finish,
        "compute": compute,
        "python": python,
        "communication": communication,
        "other": other,
        "first_compute": first_compute,
        "gaps": gaps,
        "last_compute": last_compute,
        "tail": tail,
        "has_compute": has_compute,
    }


def _validate_max_ranks(max_ranks: int) -> None:
    if isinstance(max_ranks, bool) or not isinstance(max_ranks, int):
        msg = "max_ranks must be an integer"
        raise TypeError(msg)
    if max_ranks < 1:
        msg = "max_ranks must be positive"
        raise ValueError(msg)


def _select_ranks(stats: SimulationStats, maximum: int) -> list[int]:
    count = len(stats.rank_finish_seconds)
    if count <= maximum:
        return list(range(count))
    if maximum == 1:
        return [0]
    slowest = sorted(
        range(1, count),
        key=lambda rank: (stats.rank_finish_seconds[rank], -rank),
        reverse=True,
    )[: maximum - 1]
    return [0, *sorted(slowest)]


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render mpi4py-simulator statistics as an interactive HTML report.",
    )
    parser.add_argument("input", type=Path, help="versioned simulation-stats JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output HTML path (default: input path with an .html suffix)",
    )
    parser.add_argument(
        "--max-ranks",
        type=_positive_integer,
        default=_DEFAULT_MAX_RANKS,
        help="maximum rank rows; rank 0 and slowest ranks are retained",
    )
    parser.add_argument(
        "--no-open",
        action="store_false",
        dest="open_browser",
        help="write the report without opening the system browser",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Read saved statistics and write a self-contained HTML visualization.

    Args:
        argv: Optional command-line arguments without the executable name.

    """
    args = _build_parser().parse_args(argv)
    output = args.output or args.input.with_suffix(".html")
    if output.suffix.lower() not in {".html", ".htm"}:
        msg = "visualization output must use an .html or .htm suffix"
        raise ValueError(msg)
    stats = SimulationStats.from_json(args.input.read_bytes())
    report = render_stats_html(stats, max_ranks=args.max_ranks)
    output.write_text(report, encoding="utf-8")
    sys.stdout.write(f"wrote {output}\n")
    if args.open_browser:
        url = output.resolve().as_uri()
        try:
            opened = webbrowser.open(url, new=2)
        except (OSError, webbrowser.Error):
            opened = False
        if opened:
            sys.stdout.write(f"opened {url}\n")
        else:
            sys.stdout.write(f"could not open browser; open {url}\n")


_REPORT_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MPI simulation statistics</title>
<style>
:root {
  color-scheme: light;
  --ink: #172033;
  --muted: #64748b;
  --line: #dbe3ee;
  --panel: #ffffff;
  --page: #f4f7fb;
  --compute: #2563eb;
  --python: #7c3aed;
  --communication: #ea580c;
  --other: #cbd5e1;
  --selected: #eff6ff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--page);
  color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { max-width: 1440px; margin: 0 auto; padding: 28px; }
h1 { margin: 0; font-size: clamp(24px, 3vw, 36px); letter-spacing: -0.03em; }
h2 { margin: 0 0 14px; font-size: 18px; }
.lede { margin: 6px 0 22px; color: var(--muted); }
.summary {
  display: grid;
  grid-template-columns: repeat(4, minmax(130px, 1fr));
  gap: 12px;
  margin-bottom: 18px;
}
.card, .chart-panel, .details {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--panel);
  box-shadow: 0 8px 24px rgb(15 23 42 / 0.04);
}
.card { padding: 14px 16px; }
.card .value { display: block; font: 600 20px ui-monospace, monospace; }
.card .name { color: var(--muted); font-size: 12px; }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 18px; }
.chart-panel { min-width: 0; padding: 18px; }
.controls { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
label { color: var(--muted); font-size: 12px; }
input, select {
  display: block;
  min-width: 170px;
  margin-top: 4px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: white;
  color: var(--ink);
  padding: 7px 9px;
  font: inherit;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin: 8px 0 16px;
  color: var(--muted);
}
.swatch {
  display: inline-block;
  width: 11px;
  height: 11px;
  margin-right: 5px;
  border-radius: 2px;
}
.axis, .rank-row {
  display: grid;
  grid-template-columns: 62px minmax(260px, 1fr) 88px;
  gap: 10px;
  align-items: center;
}
.axis { margin-bottom: 5px; color: var(--muted); font: 10px ui-monospace, monospace; }
.ticks { display: flex; justify-content: space-between; }
.rank-row {
  width: 100%;
  border: 0;
  border-radius: 7px;
  background: transparent;
  color: inherit;
  padding: 6px 4px;
  text-align: left;
  cursor: pointer;
}
.rank-row:hover, .rank-row:focus-visible { background: #f8fafc; outline: none; }
.rank-row.selected { background: var(--selected); }
.rank-name, .finish { font: 11px ui-monospace, monospace; }
.finish { color: var(--muted); text-align: right; }
.track {
  position: relative;
  height: 18px;
  overflow: hidden;
  border-radius: 4px;
  background:
    linear-gradient(90deg, transparent 24.8%, var(--line) 25%, transparent 25.2%),
    linear-gradient(90deg, transparent 49.8%, var(--line) 50%, transparent 50.2%),
    linear-gradient(90deg, transparent 74.8%, var(--line) 75%, transparent 75.2%),
    #f8fafc;
}
.fill { display: flex; height: 100%; min-width: 1px; }
.segment { height: 100%; min-width: 0; }
.compute { background: var(--compute); }
.python { background: var(--python); }
.communication { background: var(--communication); }
.other { background: var(--other); }
.empty { padding: 24px; color: var(--muted); text-align: center; }
.omitted { margin: 12px 0 0; color: var(--muted); font-size: 12px; }
.details { align-self: start; position: sticky; top: 18px; padding: 18px; }
.detail-rank { color: var(--muted); font: 12px ui-monospace, monospace; }
.metrics {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 8px 12px;
  margin: 16px 0;
}
.metrics dt { color: var(--muted); }
.metrics dd { margin: 0; font: 12px ui-monospace, monospace; text-align: right; }
.note { margin: 14px 0 0; color: var(--muted); font-size: 12px; }
@media (max-width: 900px) {
  main { padding: 18px; }
  .summary { grid-template-columns: repeat(2, 1fr); }
  .layout { grid-template-columns: 1fr; }
  .details { position: static; }
}
@media (max-width: 560px) {
  .axis, .rank-row { grid-template-columns: 48px minmax(170px, 1fr); }
  .axis > :last-child, .finish { display: none; }
}
</style>
</head>
<body>
<main>
  <h1>Simulation statistics</h1>
  <p class="lede">Interactive aggregate timing and communication summary.</p>
  <section class="summary" id="summary" aria-label="Simulation summary"></section>
  <div class="layout">
    <section class="chart-panel">
      <div class="controls">
        <label>Filter ranks
          <input id="filter" type="search" placeholder="e.g. 0 or 12">
        </label>
        <label>Sort ranks<select id="sort">
          <option value="rank">Rank</option>
          <option value="finish">Finish time</option>
          <option value="other">Waiting / other</option>
          <option value="occupancy">Lowest compute occupancy</option>
        </select></label>
      </div>
      <div class="legend" aria-label="Timing categories">
        <span><i class="swatch compute"></i>Compute</span>
        <span><i class="swatch python"></i>Python</span>
        <span><i class="swatch communication"></i>Communication CPU</span>
        <span><i class="swatch other"></i>Waiting / other</span>
      </div>
      <div class="axis" aria-hidden="true">
        <span>Rank</span><div class="ticks" id="ticks"></div><span>Finish</span>
      </div>
      <div id="rows"></div>
      <p class="omitted" id="omitted"></p>
    </section>
    <aside class="details" id="details" aria-live="polite"></aside>
  </div>
</main>
<script id="report-data" type="application/json">__REPORT_DATA__</script>
<script>
"use strict";
const report = JSON.parse(document.getElementById("report-data").textContent);
const categories = [
  ["compute", "Compute"],
  ["python", "Measured Python"],
  ["communication", "Communication CPU"],
  ["other", "Waiting / other"],
];
let selectedRank = report.ranks.length ? report.ranks[0].rank : null;

function duration(value) {
  if (value === 0) return "0 s";
  if (value >= 1) return `${value.toPrecision(4)} s`;
  if (value >= 1e-3) return `${(value * 1e3).toPrecision(4)} ms`;
  if (value >= 1e-6) return `${(value * 1e6).toPrecision(4)} µs`;
  return `${(value * 1e9).toPrecision(4)} ns`;
}

function exactDuration(value) {
  return `${duration(value)} (${value.toPrecision(7)} s)`;
}

function card(name, value) {
  const item = document.createElement("div");
  item.className = "card";
  const shown = document.createElement("span");
  shown.className = "value";
  shown.textContent = value;
  const label = document.createElement("span");
  label.className = "name";
  label.textContent = name;
  item.append(shown, label);
  return item;
}

function renderSummary() {
  const summary = document.getElementById("summary");
  summary.append(
    card("Makespan", duration(report.time)),
    card("Ranks", String(report.rank_count)),
    card(
      "Messages",
      `${report.messages_sent} sent / ${report.messages_received} received`,
    ),
    card("Pending", String(report.pending_messages)),
  );
  const ticks = document.getElementById("ticks");
  for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
    const tick = document.createElement("span");
    tick.textContent = duration(report.time * fraction);
    ticks.append(tick);
  }
  const omitted = report.rank_count - report.displayed_rank_count;
  document.getElementById("omitted").textContent = omitted
    ? `Rank 0 and the ${report.displayed_rank_count - 1} slowest ranks are ` +
      `included; ${omitted} omitted.`
    : "";
}

function occupancy(rank) {
  return rank.finish > 0 ? rank.compute / rank.finish : 0;
}

function orderedRanks() {
  const query = document.getElementById("filter").value.trim();
  const mode = document.getElementById("sort").value;
  const ranks = report.ranks.filter(
    rank => !query || String(rank.rank).includes(query),
  );
  ranks.sort((left, right) => {
    if (mode === "finish") return right.finish - left.finish || left.rank - right.rank;
    if (mode === "other") return right.other - left.other || left.rank - right.rank;
    if (mode === "occupancy") {
      return occupancy(left) - occupancy(right) || left.rank - right.rank;
    }
    return left.rank - right.rank;
  });
  return ranks;
}

function renderRows() {
  const rows = document.getElementById("rows");
  rows.replaceChildren();
  const ranks = orderedRanks();
  if (!ranks.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No ranks match the filter.";
    rows.append(empty);
    return;
  }
  for (const rank of ranks) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `rank-row${rank.rank === selectedRank ? " selected" : ""}`;
    row.setAttribute("aria-label", `Show details for rank ${rank.rank}`);
    row.addEventListener("click", () => selectRank(rank.rank));

    const name = document.createElement("span");
    name.className = "rank-name";
    name.textContent = String(rank.rank);
    const track = document.createElement("span");
    track.className = "track";
    const fill = document.createElement("span");
    fill.className = "fill";
    fill.style.width = `${report.time ? 100 * rank.finish / report.time : 0}%`;
    for (const [field, label] of categories) {
      const segment = document.createElement("span");
      segment.className = `segment ${field}`;
      segment.style.width = `${rank.finish ? 100 * rank[field] / rank.finish : 0}%`;
      segment.title = `${label}: ${exactDuration(rank[field])}`;
      fill.append(segment);
    }
    track.append(fill);
    const finish = document.createElement("span");
    finish.className = "finish";
    finish.textContent = duration(rank.finish);
    row.append(name, track, finish);
    rows.append(row);
  }
}

function metric(list, name, value) {
  const term = document.createElement("dt");
  term.textContent = name;
  const detail = document.createElement("dd");
  detail.textContent = value;
  list.append(term, detail);
}

function renderDetails(rank) {
  const details = document.getElementById("details");
  details.replaceChildren();
  const label = document.createElement("div");
  label.className = "detail-rank";
  label.textContent = `rank ${rank.rank}`;
  const heading = document.createElement("h2");
  heading.textContent = "Timing details";
  const list = document.createElement("dl");
  list.className = "metrics";
  metric(list, "Finish", exactDuration(rank.finish));
  metric(list, "Compute", exactDuration(rank.compute));
  metric(list, "Measured Python", exactDuration(rank.python));
  metric(list, "Communication CPU", exactDuration(rank.communication));
  metric(list, "Waiting / other", exactDuration(rank.other));
  metric(list, "Compute occupancy", `${(occupancy(rank) * 100).toFixed(1)}%`);
  metric(
    list,
    "First compute",
    rank.has_compute ? exactDuration(rank.first_compute) : "—",
  );
  metric(list, "Between-compute gaps", exactDuration(rank.gaps));
  metric(
    list,
    "Last compute finish",
    rank.has_compute ? exactDuration(rank.last_compute) : "—",
  );
  metric(list, "Tail after compute", exactDuration(rank.tail));
  const note = document.createElement("p");
  note.className = "note";
  note.textContent =
    "First, gaps, and tail locate the compute window but can include Python " +
    "and communication work. These aggregates are not an event trace.";
  details.append(label, heading, list, note);
}

function selectRank(rankNumber) {
  selectedRank = rankNumber;
  const rank = report.ranks.find(item => item.rank === rankNumber);
  if (rank) renderDetails(rank);
  renderRows();
}

document.getElementById("filter").addEventListener("input", renderRows);
document.getElementById("sort").addEventListener("change", renderRows);
renderSummary();
renderRows();
if (report.ranks.length) renderDetails(report.ranks[0]);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
