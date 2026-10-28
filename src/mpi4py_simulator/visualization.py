"""Render saved simulation statistics as a dependency-free SVG summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from ._engine import SimulationStats

_MINIMUM_WIDTH = 600
_DEFAULT_WIDTH = 1_100
_DEFAULT_MAX_RANKS = 64
_ROW_HEIGHT = 24
_CHART_TOP = 124
_CHART_LEFT = 76
_CHART_RIGHT = 178
_CHART_BOTTOM = 52

_SEGMENTS = (
    ("Compute", "rank_compute_seconds", "#2563eb"),
    ("Python", "rank_python_seconds", "#7c3aed"),
    ("Communication", "rank_communication_seconds", "#ea580c"),
)
_OTHER_COLOR = "#cbd5e1"


def render_stats_svg(
    stats: SimulationStats,
    *,
    width: int = _DEFAULT_WIDTH,
    max_ranks: int = _DEFAULT_MAX_RANKS,
) -> str:
    """Render aggregate and per-rank statistics as a standalone SVG document.

    Args:
        stats: Completed simulation statistics.
        width: Output width in SVG units. Must be at least 600.
        max_ranks: Maximum rank rows to draw. When a simulation has more ranks, rank
            zero and the slowest remaining ranks are shown.

    Returns:
        A UTF-8 SVG string. Each rank row is a stacked time composition containing
        explicit compute, measured Python, communication CPU, and residual time.

    Raises:
        TypeError: If ``width`` or ``max_ranks`` is not an integer.
        ValueError: If ``width`` is too small or ``max_ranks`` is not positive.

    Notes:
        The residual ``Waiting / other`` segment is derived from rank finish time
        after subtracting the three explicitly accounted components. It can include
        message transit, blocking waits, collective delay, and idle time; the saved
        aggregate statistics do not contain a chronological event trace.

    """
    _validate_render_options(width, max_ranks)
    ranks = _select_ranks(stats, max_ranks)
    height = _CHART_TOP + len(ranks) * _ROW_HEIGHT + _CHART_BOTTOM
    chart_width = width - _CHART_LEFT - _CHART_RIGHT
    scale_seconds = max(stats.time, 1e-15, *stats.rank_finish_seconds)

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
        ),
        "<title>Simulation statistics</title>",
        (
            "<desc>Per-rank finish time decomposed into modeled compute, measured "
            "Python, communication CPU, and waiting or other time.</desc>"
        ),
        "<style>",
        "text { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }",
        (
            ".title { font-family: system-ui, sans-serif; font-size: 20px; "
            "font-weight: 700; fill: #0f172a; }"
        ),
        (
            ".summary { font-family: system-ui, sans-serif; font-size: 13px; "
            "fill: #334155; }"
        ),
        ".label { font-size: 11px; fill: #334155; }",
        ".tick { font-size: 10px; fill: #64748b; }",
        (
            ".legend { font-family: system-ui, sans-serif; font-size: 11px; "
            "fill: #334155; }"
        ),
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="24" y="30" class="title">Simulation statistics</text>',
        (
            f'<text x="24" y="54" class="summary">Makespan '
            f"{_seconds(stats.time)} · {len(stats.rank_finish_seconds)} ranks · "
            f"{stats.messages_sent} sent · {stats.messages_received} received · "
            f"{stats.pending_messages} pending</text>"
        ),
    ]
    lines.extend(_legend_lines())
    lines.extend(
        _axis_lines(
            chart_width=chart_width,
            height=height,
            scale_seconds=scale_seconds,
        ),
    )

    for row, rank in enumerate(ranks):
        lines.extend(
            _rank_lines(
                stats,
                rank,
                row,
                chart_width=chart_width,
                scale_seconds=scale_seconds,
            ),
        )

    if len(ranks) < len(stats.rank_finish_seconds):
        omitted = len(stats.rank_finish_seconds) - len(ranks)
        lines.append(
            f'<text x="{width - 24}" y="30" text-anchor="end" class="summary">'
            f"rank 0 + {len(ranks) - 1} slowest shown; {omitted} omitted</text>",
        )
    lines.append("</svg>")
    return "\n".join(lines)


def _validate_render_options(width: int, max_ranks: int) -> None:
    if isinstance(width, bool) or not isinstance(width, int):
        msg = "width must be an integer"
        raise TypeError(msg)
    if width < _MINIMUM_WIDTH:
        msg = f"width must be at least {_MINIMUM_WIDTH}"
        raise ValueError(msg)
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


def _legend_lines() -> list[str]:
    lines = []
    x = 24
    for label, _, color in (*_SEGMENTS, ("Waiting / other", "", _OTHER_COLOR)):
        lines.append(f'<rect x="{x}" y="70" width="12" height="12" fill="{color}"/>')
        lines.append(f'<text x="{x + 17}" y="80" class="legend">{label}</text>')
        x += 17 + len(label) * 7 + 18
    return lines


def _axis_lines(
    *,
    chart_width: int,
    height: int,
    scale_seconds: float,
) -> list[str]:
    lines = []
    chart_bottom = height - _CHART_BOTTOM + 6
    for tick in range(5):
        fraction = tick / 4
        x = _CHART_LEFT + chart_width * fraction
        value = scale_seconds * fraction
        lines.append(
            f'<line x1="{x:.2f}" y1="{_CHART_TOP - 18}" x2="{x:.2f}" '
            f'y2="{chart_bottom}" stroke="#e2e8f0" stroke-width="1"/>',
        )
        lines.append(
            f'<text x="{x:.2f}" y="{_CHART_TOP - 24}" text-anchor="middle" '
            f'class="tick">{_seconds(value)}</text>',
        )
    return lines


def _rank_lines(
    stats: SimulationStats,
    rank: int,
    row: int,
    *,
    chart_width: int,
    scale_seconds: float,
) -> list[str]:
    y = _CHART_TOP + row * _ROW_HEIGHT
    bar_y = y - 11
    finish = stats.rank_finish_seconds[rank]
    finish_width = chart_width * finish / scale_seconds
    lines = [
        f'<text x="{_CHART_LEFT - 8}" y="{y}" text-anchor="end" class="label">'
        f"rank {rank}</text>",
        f'<rect x="{_CHART_LEFT}" y="{bar_y}" width="{chart_width}" height="14" '
        'rx="2" fill="#f8fafc"/>',
    ]

    values = [getattr(stats, field)[rank] for _, field, _ in _SEGMENTS]
    remaining = finish
    x = float(_CHART_LEFT)
    for (label, _, color), value in zip(_SEGMENTS, values, strict=True):
        visible = min(max(value, 0.0), remaining)
        segment_width = chart_width * visible / scale_seconds
        if segment_width > 0.0:
            lines.append(
                f'<rect x="{x:.2f}" y="{bar_y}" width="{segment_width:.2f}" '
                f'height="14" fill="{color}"><title>{label}: '
                f"{_seconds(value)}</title></rect>",
            )
        x += segment_width
        remaining -= visible

    other_width = chart_width * remaining / scale_seconds
    if other_width > 0.0:
        lines.append(
            f'<rect x="{x:.2f}" y="{bar_y}" width="{other_width:.2f}" height="14" '
            f'fill="{_OTHER_COLOR}"><title>Waiting / other: '
            f"{_seconds(remaining)}</title></rect>",
        )
    lines.append(
        f'<line x1="{_CHART_LEFT + finish_width:.2f}" y1="{bar_y - 2}" '
        f'x2="{_CHART_LEFT + finish_width:.2f}" y2="{bar_y + 16}" '
        'stroke="#0f172a" stroke-width="1"/>',
    )
    lines.append(
        f'<text x="{_CHART_LEFT + chart_width + 8}" y="{y}" class="label">'
        f"{_seconds(finish)}</text>",
    )
    return lines


def _seconds(value: float) -> str:
    return f"{value:.6g} s"


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
        description="Render mpi4py-simulator statistics as a standalone SVG.",
    )
    parser.add_argument("input", type=Path, help="versioned simulation-stats JSON")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output SVG path (default: input path with an .svg suffix)",
    )
    parser.add_argument("--width", type=_positive_integer, default=_DEFAULT_WIDTH)
    parser.add_argument(
        "--max-ranks",
        type=_positive_integer,
        default=_DEFAULT_MAX_RANKS,
        help="maximum rank rows; rank 0 and slowest ranks are retained",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Read saved statistics and write an SVG visualization.

    Args:
        argv: Optional command-line arguments without the executable name.

    """
    args = _build_parser().parse_args(argv)
    output = args.output or args.input.with_suffix(".svg")
    stats = SimulationStats.from_json(args.input.read_bytes())
    svg = render_stats_svg(stats, width=args.width, max_ranks=args.max_ranks)
    output.write_text(svg, encoding="utf-8")
    sys.stdout.write(f"wrote {output}\n")


if __name__ == "__main__":
    main()
