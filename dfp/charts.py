"""Minimal dependency-free SVG charts (matplotlib is not installed).

Three chart types cover every figure the evaluation needs: a confusion-matrix
heatmap, a grouped/simple bar chart, and a multi-series line chart (the
minimum-evidence curve and the adversarial before/after chart).  Each returns a
standalone ``<svg>`` string that embeds directly in the HTML report or writes to
a ``.svg`` file.
"""

from __future__ import annotations

import html


def _esc(s: str) -> str:
    return html.escape(str(s))


def confusion_svg(
    matrix: list[list[float]],
    labels: list[str],
    title: str = "Confusion matrix",
    normalise: bool = True,
) -> str:
    n = len(labels)
    cell = 46
    left = 120
    top = 90
    w = left + n * cell + 30
    h = top + n * cell + 60
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'font-family="Segoe UI,Arial,sans-serif" font-size="12">',
        f'<text x="{w/2}" y="24" text-anchor="middle" font-size="16" '
        f'font-weight="bold">{_esc(title)}</text>',
        f'<text x="{left + n*cell/2}" y="{top-30}" text-anchor="middle" '
        f'font-size="11" fill="#555">predicted</text>',
    ]
    row_sums = [sum(r) or 1.0 for r in matrix]
    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            frac = v / row_sums[i] if normalise else v
            shade = int(255 - min(frac, 1.0) * 200)
            on_diag = i == j
            fill = (
                f"rgb({shade},{255 if on_diag else shade},{shade})"
                if on_diag
                else f"rgb(255,{shade},{shade})"
            )
            x = left + j * cell
            y = top + i * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{fill}" stroke="#ccc"/>'
            )
            disp = f"{frac:.2f}" if normalise else f"{int(v)}"
            parts.append(
                f'<text x="{x+cell/2}" y="{y+cell/2+4}" text-anchor="middle" '
                f'fill="#111">{disp}</text>'
            )
    for i, lab in enumerate(labels):
        parts.append(
            f'<text x="{left-8}" y="{top+i*cell+cell/2+4}" text-anchor="end">'
            f'{_esc(lab)}</text>'
        )
        parts.append(
            f'<text x="{left+i*cell+cell/2}" y="{top-8}" text-anchor="middle" '
            f'transform="rotate(-35 {left+i*cell+cell/2} {top-8})">{_esc(lab)}</text>'
        )
    parts.append(
        f'<text x="20" y="{top+n*cell/2}" text-anchor="middle" font-size="11" '
        f'fill="#555" transform="rotate(-90 20 {top+n*cell/2})">actual</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def bar_svg(
    labels: list[str],
    values: list[float],
    title: str = "",
    ymax: float | None = None,
    ylabel: str = "",
    fmt: str = "{:.2f}",
) -> str:
    w = max(360, 60 + len(labels) * 70)
    h = 300
    pad_b, pad_t, pad_l = 60, 50, 50
    ymax = ymax or (max(values) * 1.15 if values else 1.0)
    plot_h = h - pad_b - pad_t
    bar_w = (w - pad_l - 20) / max(len(labels), 1) * 0.6
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'font-family="Segoe UI,Arial,sans-serif" font-size="12">',
        f'<text x="{w/2}" y="24" text-anchor="middle" font-size="15" '
        f'font-weight="bold">{_esc(title)}</text>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h-pad_b}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{h-pad_b}" x2="{w-10}" y2="{h-pad_b}" stroke="#333"/>',
    ]
    if ylabel:
        parts.append(
            f'<text x="16" y="{pad_t+plot_h/2}" text-anchor="middle" font-size="11" '
            f'fill="#555" transform="rotate(-90 16 {pad_t+plot_h/2})">{_esc(ylabel)}</text>'
        )
    step = (w - pad_l - 20) / max(len(labels), 1)
    for i, (lab, val) in enumerate(zip(labels, values)):
        bh = (val / ymax) * plot_h if ymax else 0
        x = pad_l + i * step + (step - bar_w) / 2
        y = h - pad_b - bh
        parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" fill="#3b7dd8"/>'
        )
        parts.append(
            f'<text x="{x+bar_w/2}" y="{y-5}" text-anchor="middle" font-size="11">'
            f'{_esc(fmt.format(val))}</text>'
        )
        parts.append(
            f'<text x="{x+bar_w/2}" y="{h-pad_b+16}" text-anchor="middle" '
            f'font-size="10" transform="rotate(-20 {x+bar_w/2} {h-pad_b+16})">'
            f'{_esc(lab)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def line_svg(
    xs: list[float],
    series: dict[str, list[float]],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    logx: bool = False,
) -> str:
    import math

    w, h = 560, 340
    pad_l, pad_b, pad_t, pad_r = 60, 60, 50, 130
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_b - pad_t
    xv = [math.log10(max(x, 1)) for x in xs] if logx else list(xs)
    xmin, xmax = (min(xv), max(xv)) if xv else (0, 1)
    xrange = (xmax - xmin) or 1
    colours = ["#3b7dd8", "#d84b3b", "#2ca05a", "#a05aca", "#d8a53b"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'font-family="Segoe UI,Arial,sans-serif" font-size="12">',
        f'<text x="{(pad_l+plot_w)/2+pad_l/2}" y="24" text-anchor="middle" '
        f'font-size="15" font-weight="bold">{_esc(title)}</text>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h-pad_b}" stroke="#333"/>',
        f'<line x1="{pad_l}" y1="{h-pad_b}" x2="{pad_l+plot_w}" y2="{h-pad_b}" stroke="#333"/>',
    ]
    for gy in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + (1 - gy) * plot_h
        parts.append(
            f'<line x1="{pad_l}" y1="{y}" x2="{pad_l+plot_w}" y2="{y}" '
            f'stroke="#eee"/>'
        )
        parts.append(f'<text x="{pad_l-8}" y="{y+4}" text-anchor="end" '
                     f'font-size="10">{gy:.2f}</text>')
    for si, (name, ys) in enumerate(series.items()):
        col = colours[si % len(colours)]
        pts = []
        for x, y in zip(xv, ys):
            px = pad_l + (x - xmin) / xrange * plot_w
            py = pad_t + (1 - max(0.0, min(1.0, y))) * plot_h
            pts.append(f"{px:.1f},{py:.1f}")
        parts.append(
            f'<polyline fill="none" stroke="{col}" stroke-width="2" '
            f'points="{" ".join(pts)}"/>'
        )
        for pt in pts:
            cx, cy = pt.split(",")
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="3" fill="{col}"/>')
        ly = pad_t + si * 18
        parts.append(f'<rect x="{pad_l+plot_w+14}" y="{ly}" width="12" height="12" '
                     f'fill="{col}"/>')
        parts.append(f'<text x="{pad_l+plot_w+30}" y="{ly+11}" font-size="11">'
                     f'{_esc(name)}</text>')
    for x, raw in zip(xv, xs):
        px = pad_l + (x - xmin) / xrange * plot_w
        parts.append(f'<text x="{px}" y="{h-pad_b+16}" text-anchor="middle" '
                     f'font-size="10">{_esc(_fmtnum(raw))}</text>')
    if xlabel:
        parts.append(f'<text x="{pad_l+plot_w/2}" y="{h-14}" text-anchor="middle" '
                     f'font-size="11" fill="#555">{_esc(xlabel)}</text>')
    if ylabel:
        parts.append(f'<text x="16" y="{pad_t+plot_h/2}" text-anchor="middle" '
                     f'font-size="11" fill="#555" '
                     f'transform="rotate(-90 16 {pad_t+plot_h/2})">{_esc(ylabel)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _fmtnum(x: float) -> str:
    if x >= 1000:
        return f"{x/1000:.0f}k"
    return f"{x:g}"
