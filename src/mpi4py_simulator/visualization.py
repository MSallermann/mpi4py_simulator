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
    events_by_rank = {rank: [] for rank in ranks}
    for event in stats.trace_events:
        if event.rank in events_by_rank:
            events_by_rank[event.rank].append(
                {
                    "kind": event.kind,
                    "start": event.start,
                    "finish": event.finish,
                    "detail": event.detail,
                },
            )
    report = {
        "time": stats.time,
        "rank_count": len(stats.rank_finish_seconds),
        "displayed_rank_count": len(ranks),
        "messages_sent": stats.messages_sent,
        "messages_received": stats.messages_received,
        "pending_messages": stats.pending_messages,
        "trace_event_count": len(stats.trace_events),
        "ranks": [_rank_record(stats, rank, events_by_rank[rank]) for rank in ranks],
    }
    # Keep the embedded JSON inert if future formats gain user-controlled strings.
    payload = json.dumps(report, separators=(",", ":"), allow_nan=False).replace(
        "<",
        "\\u003c",
    )
    return _REPORT_TEMPLATE.replace("__REPORT_DATA__", payload)


def _rank_record(
    stats: SimulationStats,
    rank: int,
    events: list[dict[str, object]],
) -> dict[str, object]:
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
        "events": events,
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
h3 { margin: 20px 0 0; font-size: 14px; }
.lede { margin: 6px 0 22px; color: var(--muted); }
.summary {
  display: grid;
  grid-template-columns: repeat(5, minmax(120px, 1fr));
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
.zoom-controls {
  display: flex;
  align-items: end;
  gap: 6px;
  margin-left: auto;
}
.zoom-controls[hidden] { display: none; }
.zoom-controls button {
  border: 1px solid var(--line);
  border-radius: 7px;
  background: white;
  color: var(--ink);
  padding: 7px 10px;
  font: inherit;
  cursor: pointer;
}
.zoom-controls button:hover:not(:disabled) { background: #f8fafc; }
.zoom-controls button:disabled { color: var(--muted); cursor: default; opacity: 0.55; }
.zoom-window {
  min-width: 145px;
  color: var(--muted);
  font: 11px ui-monospace, monospace;
  text-align: right;
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin: 8px 0 16px;
  color: var(--muted);
}
.view-note { margin: -8px 0 14px; color: var(--muted); font-size: 12px; }
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
.track.timeline { cursor: grab; touch-action: none; }
.track.timeline:active { cursor: grabbing; }
.fill { display: flex; height: 100%; min-width: 1px; }
.segment { height: 100%; min-width: 0; }
.compute { background: var(--compute); }
.python { background: var(--python); }
.communication { background: var(--communication); }
.other { background: var(--other); }
.wait { background: #94a3b8; }
.collective { background: #0d9488; }
.trace-event {
  position: absolute;
  top: 0;
  height: 100%;
  min-width: 1px;
  border-right: 1px solid rgb(255 255 255 / 0.45);
}
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
.event-list {
  max-height: 260px;
  overflow: auto;
  margin: 12px 0 0;
  padding: 0;
  list-style: none;
  border-top: 1px solid var(--line);
}
.event-list li { padding: 8px 0; border-bottom: 1px solid var(--line); }
.event-kind { font-weight: 600; text-transform: capitalize; }
.event-time { color: var(--muted); font: 11px ui-monospace, monospace; }
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
  <p class="lede">Interactive aggregate timing and optional event timeline.</p>
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
        <label>View<select id="view">
          <option value="aggregate">Aggregate composition</option>
          <option value="timeline">Event timeline</option>
        </select></label>
        <div class="zoom-controls" id="zoom-controls" hidden>
          <button type="button" id="zoom-out" title="Zoom out">&minus;</button>
          <button type="button" id="zoom-in" title="Zoom in">+</button>
          <button type="button" id="zoom-reset">Reset zoom</button>
          <span class="zoom-window" id="zoom-window"></span>
        </div>
      </div>
      <div class="legend" id="legend" aria-label="Timing categories"></div>
      <p class="view-note" id="view-note"></p>
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
const aggregateCategories = [
  ["compute", "Compute"],
  ["python", "Measured Python"],
  ["communication", "Communication CPU"],
  ["other", "Waiting / other"],
];
const traceCategories = [
  ["compute", "Compute"],
  ["python", "Measured Python"],
  ["communication", "Communication CPU"],
  ["wait", "Blocking wait"],
  ["collective", "Collective"],
];
const traceLabels = Object.fromEntries(traceCategories);
const maxDetailEvents = 250;
let selectedRank = report.ranks.length ? report.ranks[0].rank : null;
let timelineStart = 0;
let timelineEnd = report.time;
let panState = null;
let suppressRankSelection = false;
const minimumTimelineSpan = report.time > 0 ? report.time / 1e6 : 0;

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
    card("Trace events", String(report.trace_event_count)),
  );
  const timelineOption = document.querySelector('#view option[value="timeline"]');
  if (!report.trace_event_count) {
    timelineOption.disabled = true;
    timelineOption.textContent = "Event timeline (trace unavailable)";
  }
  const omitted = report.rank_count - report.displayed_rank_count;
  document.getElementById("omitted").textContent = omitted
    ? `Rank 0 and the ${report.displayed_rank_count - 1} slowest ranks are ` +
      `included; ${omitted} omitted.`
    : "";
}

function timelineActive() {
  return document.getElementById("view").value === "timeline";
}

function renderTicks() {
  const ticks = document.getElementById("ticks");
  ticks.replaceChildren();
  const start = timelineActive() ? timelineStart : 0;
  const finish = timelineActive() ? timelineEnd : report.time;
  for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
    const tick = document.createElement("span");
    tick.textContent = duration(start + (finish - start) * fraction);
    ticks.append(tick);
  }
}

function updateZoomControls() {
  const controls = document.getElementById("zoom-controls");
  controls.hidden = !timelineActive();
  const span = timelineEnd - timelineStart;
  const tolerance = report.time * 1e-12;
  document.getElementById("zoom-in").disabled =
    report.time <= 0 || span <= minimumTimelineSpan + tolerance;
  document.getElementById("zoom-out").disabled =
    report.time <= 0 || span >= report.time - tolerance;
  document.getElementById("zoom-reset").disabled =
    report.time <= 0 || span >= report.time - tolerance;
  document.getElementById("zoom-window").textContent =
    `${duration(timelineStart)} to ${duration(timelineEnd)}`;
}

function setTimelineWindow(start, finish) {
  if (report.time <= 0) {
    timelineStart = 0;
    timelineEnd = 0;
  } else {
    const span = Math.min(
      report.time,
      Math.max(minimumTimelineSpan, finish - start),
    );
    timelineStart = Math.max(0, Math.min(start, report.time - span));
    timelineEnd = timelineStart + span;
  }
  renderTicks();
  updateZoomControls();
  renderRows();
}

function zoomTimeline(factor, anchorFraction = 0.5) {
  const span = timelineEnd - timelineStart;
  if (report.time <= 0 || span <= 0) return;
  const fraction = Math.max(0, Math.min(1, anchorFraction));
  const nextSpan = Math.max(
    minimumTimelineSpan,
    Math.min(report.time, span * factor),
  );
  const anchor = timelineStart + span * fraction;
  setTimelineWindow(
    anchor - nextSpan * fraction,
    anchor + nextSpan * (1 - fraction),
  );
}

function renderLegend() {
  const legend = document.getElementById("legend");
  legend.replaceChildren();
  const categories = document.getElementById("view").value === "timeline"
    ? traceCategories
    : aggregateCategories;
  for (const [field, label] of categories) {
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.className = `swatch ${field}`;
    item.append(swatch, label);
    legend.append(item);
  }
  document.getElementById("view-note").textContent =
    timelineActive()
      ? "Use +/- or the mouse wheel to zoom; drag the timeline to pan. " +
        "Blank regions have no retained interval."
      : "Composition segments are totals, not chronological positions.";
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
    row.addEventListener("click", () => {
      if (!suppressRankSelection) selectRank(rank.rank);
    });

    const name = document.createElement("span");
    name.className = "rank-name";
    name.textContent = String(rank.rank);
    const track = document.createElement("span");
    track.className = "track";
    if (timelineActive()) {
      track.classList.add("timeline");
      const visibleSpan = timelineEnd - timelineStart;
      for (const event of rank.events) {
        const visibleStart = Math.max(event.start, timelineStart);
        const visibleFinish = Math.min(event.finish, timelineEnd);
        if (visibleFinish <= visibleStart) continue;
        const segment = document.createElement("span");
        segment.className = `trace-event ${event.kind}`;
        segment.style.left = visibleSpan
          ? `${100 * (visibleStart - timelineStart) / visibleSpan}%`
          : "0%";
        segment.style.width = visibleSpan
          ? `${100 * (visibleFinish - visibleStart) / visibleSpan}%`
          : "0%";
        segment.title =
          `${traceLabels[event.kind]}: ${event.detail} · ` +
          `${exactDuration(event.start)} to ${exactDuration(event.finish)}`;
        track.append(segment);
      }
    } else {
      const fill = document.createElement("span");
      fill.className = "fill";
      fill.style.width = `${report.time ? 100 * rank.finish / report.time : 0}%`;
      for (const [field, label] of aggregateCategories) {
        const segment = document.createElement("span");
        segment.className = `segment ${field}`;
        segment.style.width =
          `${rank.finish ? 100 * rank[field] / rank.finish : 0}%`;
        segment.title = `${label}: ${exactDuration(rank[field])}`;
        fill.append(segment);
      }
      track.append(fill);
    }
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
    "and communication work. Use the event timeline for chronological detail.";
  details.append(label, heading, list, note);
  if (rank.events.length) {
    const eventHeading = document.createElement("h3");
    eventHeading.textContent = `Trace events (${rank.events.length})`;
    const events = document.createElement("ul");
    events.className = "event-list";
    for (const event of rank.events.slice(0, maxDetailEvents)) {
      const item = document.createElement("li");
      const kind = document.createElement("div");
      kind.className = "event-kind";
      kind.textContent = `${traceLabels[event.kind]} · ${event.detail}`;
      const timing = document.createElement("div");
      timing.className = "event-time";
      timing.textContent =
        `${duration(event.start)} to ${duration(event.finish)} · ` +
        `${duration(event.finish - event.start)}`;
      item.append(kind, timing);
      events.append(item);
    }
    details.append(eventHeading, events);
    if (rank.events.length > maxDetailEvents) {
      const truncated = document.createElement("p");
      truncated.className = "note";
      truncated.textContent =
        `Showing the first ${maxDetailEvents} of ${rank.events.length} events. ` +
        "The timeline still includes every event.";
      details.append(truncated);
    }
  }
}

function selectRank(rankNumber) {
  selectedRank = rankNumber;
  const rank = report.ranks.find(item => item.rank === rankNumber);
  if (rank) renderDetails(rank);
  renderRows();
}

document.getElementById("filter").addEventListener("input", renderRows);
document.getElementById("sort").addEventListener("change", renderRows);
document.getElementById("view").addEventListener("change", () => {
  renderLegend();
  renderTicks();
  updateZoomControls();
  renderRows();
});
document.getElementById("zoom-in").addEventListener("click", () => {
  zoomTimeline(0.5);
});
document.getElementById("zoom-out").addEventListener("click", () => {
  zoomTimeline(2);
});
document.getElementById("zoom-reset").addEventListener("click", () => {
  setTimelineWindow(0, report.time);
});

const rowsElement = document.getElementById("rows");
rowsElement.addEventListener("wheel", event => {
  if (!timelineActive()) return;
  const track = event.target.closest(".track.timeline");
  if (!track) return;
  event.preventDefault();
  const bounds = track.getBoundingClientRect();
  const anchor = bounds.width ? (event.clientX - bounds.left) / bounds.width : 0.5;
  zoomTimeline(Math.exp(event.deltaY * 0.0015), anchor);
}, {passive: false});

rowsElement.addEventListener("pointerdown", event => {
  if (!timelineActive() || event.button !== 0) return;
  const track = event.target.closest(".track.timeline");
  if (!track) return;
  const bounds = track.getBoundingClientRect();
  panState = {
    pointerId: event.pointerId,
    lastX: event.clientX,
    width: bounds.width,
    moved: false,
  };
  rowsElement.setPointerCapture(event.pointerId);
  event.preventDefault();
});

rowsElement.addEventListener("pointermove", event => {
  if (!panState || event.pointerId !== panState.pointerId) return;
  const delta = event.clientX - panState.lastX;
  panState.lastX = event.clientX;
  if (!delta || !panState.width) return;
  panState.moved = true;
  const span = timelineEnd - timelineStart;
  const shift = -delta * span / panState.width;
  setTimelineWindow(timelineStart + shift, timelineEnd + shift);
});

function finishPan(event) {
  if (!panState || event.pointerId !== panState.pointerId) return;
  const moved = panState.moved;
  panState = null;
  if (rowsElement.hasPointerCapture(event.pointerId)) {
    rowsElement.releasePointerCapture(event.pointerId);
  }
  if (moved) {
    suppressRankSelection = true;
    window.setTimeout(() => { suppressRankSelection = false; }, 0);
  }
}

rowsElement.addEventListener("pointerup", finishPan);
rowsElement.addEventListener("pointercancel", finishPan);
renderSummary();
renderLegend();
renderTicks();
updateZoomControls();
renderRows();
if (report.ranks.length) renderDetails(report.ranks[0]);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
