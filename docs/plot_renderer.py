#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_renderer.py
Parses ```plot``` fenced blocks from Markdown and renders them as inline SVG
using Matplotlib with typography that matches the EB Garamond document theme.

Supported chart types:
  line, bar, scatter, hist, box, heatmap, function
"""

import io
import re
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

# Use non-interactive Agg backend — no display needed
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Shared rcParams — mirrors the document's typographic style
# ---------------------------------------------------------------------------
PLOT_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["EB Garamond", "Georgia", "Times New Roman"],
    "mathtext.fontset": "cm",           # Computer Modern for math labels
    "axes.spines.top": False,           # IEEE / booktabs style
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#555555",
    "axes.labelcolor": "#111111",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.6,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "legend.fontsize": 9,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "legend.fancybox": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "patch.linewidth": 0.8,
}

# Default palette inspired by IEEE / Nature journal colors
DEFAULT_COLORS = [
    "#2b5797",  # deep blue
    "#d73a49",  # crimson
    "#22863a",  # forest green
    "#e36209",  # amber
    "#6f42c1",  # violet
    "#0969da",  # azure
    "#bf8700",  # gold
    "#1b7c83",  # teal
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_plot_blocks(md_content: str):
    """
    Find all ```plot ... ``` fenced blocks in Markdown.
    Returns a list of (full_match_string, yaml_body) tuples.
    """
    pattern = re.compile(
        r'```plot\s*\n(.*?)\n```',
        re.DOTALL
    )
    return [(m.group(0), m.group(1)) for m in pattern.finditer(md_content)]


def render_plot_block(yaml_body: str) -> str:
    """
    Parse a YAML plot specification and render it as an SVG string.
    Returns an HTML <figure> element containing the inline SVG.
    """
    try:
        spec = yaml.safe_load(yaml_body)
    except yaml.YAMLError as e:
        return f'<div class="plot-error"><strong>[plot error]</strong> YAML parse error: {e}</div>'

    chart_type = spec.get("type", "line").lower()

    with matplotlib.rc_context(PLOT_RCPARAMS):
        fig = _dispatch_chart(chart_type, spec)
        if fig is None:
            return f'<div class="plot-error"><strong>[plot error]</strong> Unknown chart type: <code>{chart_type}</code></div>'

        svg_buf = io.StringIO()
        fig.savefig(svg_buf, format="svg")
        plt.close(fig)

    svg_str = svg_buf.getvalue()
    # Strip the XML declaration so it embeds cleanly in HTML
    svg_str = re.sub(r'<\?xml[^?]*\?>', '', svg_str).strip()
    # Remove standalone DOCTYPE if present
    svg_str = re.sub(r'<!DOCTYPE[^>]*>', '', svg_str).strip()

    # Wrap in <figure> with optional caption
    caption = spec.get("caption", "")
    anchor = spec.get("anchor", "")
    anchor_attr = f' id="{anchor}"' if anchor else ""

    if caption:
        fig_html = (
            f'<figure class="plot-figure"{anchor_attr}>'
            f'{svg_str}'
            f'<figcaption class="plot-caption">{caption}</figcaption>'
            f'</figure>'
        )
    else:
        fig_html = f'<figure class="plot-figure"{anchor_attr}>{svg_str}</figure>'

    return fig_html


def preprocess_plots(md_content: str) -> str:
    """
    Replace every ```plot``` block in the Markdown string with its rendered SVG HTML.
    Called by md_to_pdf.py before Markdown → HTML conversion.
    """
    blocks = extract_plot_blocks(md_content)
    for raw_block, yaml_body in blocks:
        rendered = render_plot_block(yaml_body)
        md_content = md_content.replace(raw_block, rendered, 1)
    return md_content


# ---------------------------------------------------------------------------
# Chart dispatchers
# ---------------------------------------------------------------------------

def _apply_common(ax, spec):
    """Apply title, labels, and legend from spec to an Axes object."""
    title = spec.get("title", "")
    xlabel = spec.get("xlabel", "")
    ylabel = spec.get("ylabel", "")

    if title:
        ax.set_title(title, pad=10, color="#111111")
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=6)

    # Draw legend only if there are labelled artists
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()


def _make_fig(spec, default_width=6.5, default_height=3.8):
    width = spec.get("width", default_width)
    height = spec.get("height", default_height)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def _dispatch_chart(chart_type: str, spec: dict):
    dispatch = {
        "line":     _render_line,
        "bar":      _render_bar,
        "scatter":  _render_scatter,
        "hist":     _render_hist,
        "box":      _render_box,
        "heatmap":  _render_heatmap,
        "function": _render_function,
    }
    fn = dispatch.get(chart_type)
    return fn(spec) if fn else None


# ---------------------------------------------------------------------------
# Line chart
# ---------------------------------------------------------------------------

def _render_line(spec: dict):
    fig, ax = _make_fig(spec)

    datasets = spec.get("datasets", [])
    # Simple shorthand: single list of [x, y] pairs or y-values
    if not datasets and "data" in spec:
        datasets = [{"data": spec["data"], "label": spec.get("label", "")}]

    for i, ds in enumerate(datasets):
        color = ds.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        label = ds.get("label", "")
        raw = ds.get("data", [])

        if raw and isinstance(raw[0], (list, tuple)):
            xs = [p[0] for p in raw]
            ys = [p[1] for p in raw]
        else:
            xs = list(range(len(raw)))
            ys = raw

        marker = ds.get("marker", "o")
        ax.plot(xs, ys, color=color, label=label, marker=marker,
                linewidth=1.6, markersize=4)

    _apply_common(ax, spec)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def _render_bar(spec: dict):
    fig, ax = _make_fig(spec)

    datasets = spec.get("datasets", [])
    if not datasets and "data" in spec:
        datasets = [{"data": spec["data"], "label": spec.get("label", "")}]

    labels_x = spec.get("labels", [])
    n_ds = len(datasets)
    width = 0.7 / max(n_ds, 1)
    offsets = np.linspace(-(n_ds - 1) * width / 2, (n_ds - 1) * width / 2, n_ds)

    x_pos = np.arange(len(labels_x) if labels_x else
                      len(datasets[0].get("data", [])))

    for i, ds in enumerate(datasets):
        color = ds.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        label = ds.get("label", "")
        values = ds.get("data", [])
        ax.bar(x_pos + offsets[i], values, width=width * 0.92,
               color=color, label=label, alpha=0.88)

    if labels_x:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_x)

    ax.yaxis.grid(True, linestyle="--", alpha=0.45, color="#cccccc")
    ax.set_axisbelow(True)
    _apply_common(ax, spec)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Scatter chart
# ---------------------------------------------------------------------------

def _render_scatter(spec: dict):
    fig, ax = _make_fig(spec)

    datasets = spec.get("datasets", [])
    if not datasets and "data" in spec:
        datasets = [{"data": spec["data"], "label": spec.get("label", "")}]

    for i, ds in enumerate(datasets):
        color = ds.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        label = ds.get("label", "")
        raw = ds.get("data", [])
        if raw and isinstance(raw[0], (list, tuple)):
            xs = [p[0] for p in raw]
            ys = [p[1] for p in raw]
        else:
            xs = list(range(len(raw)))
            ys = raw
        ax.scatter(xs, ys, color=color, label=label, s=40, alpha=0.85,
                   edgecolors="white", linewidths=0.5)

    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    _apply_common(ax, spec)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def _render_hist(spec: dict):
    fig, ax = _make_fig(spec)

    datasets = spec.get("datasets", [])
    if not datasets and "data" in spec:
        datasets = [{"data": spec["data"], "label": spec.get("label", "")}]

    bins = spec.get("bins", 20)

    for i, ds in enumerate(datasets):
        color = ds.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        label = ds.get("label", "")
        values = ds.get("data", [])
        ax.hist(values, bins=bins, color=color, label=label,
                alpha=0.78, edgecolor="white", linewidth=0.5)

    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    _apply_common(ax, spec)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Box plot
# ---------------------------------------------------------------------------

def _render_box(spec: dict):
    fig, ax = _make_fig(spec)

    datasets = spec.get("datasets", [])
    data_arrays = [ds.get("data", []) for ds in datasets]
    labels = [ds.get("label", f"Group {i+1}") for i, ds in enumerate(datasets)]

    bp = ax.boxplot(data_arrays, labels=labels, patch_artist=True,
                    medianprops={"color": "#111111", "linewidth": 1.5},
                    whiskerprops={"linewidth": 0.8},
                    capprops={"linewidth": 0.8},
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": "#999999", "linestyle": "none"})

    for i, patch in enumerate(bp["boxes"]):
        color = datasets[i].get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    _apply_common(ax, spec)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def _render_heatmap(spec: dict):
    fig, ax = _make_fig(spec, default_height=4.2)

    matrix = spec.get("data", [[]])
    arr = np.array(matrix, dtype=float)

    cmap_name = spec.get("colormap", "Blues")
    vmin = spec.get("vmin", None)
    vmax = spec.get("vmax", None)

    im = ax.imshow(arr, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.85)

    row_labels = spec.get("row_labels", [])
    col_labels = spec.get("col_labels", [])

    if row_labels:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
    if col_labels:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right")

    # Annotate cells with values if requested
    if spec.get("annotate", False):
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                val = arr[r, c]
                text_color = "white" if val > (arr.max() * 0.6) else "#111111"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color)

    ax.grid(False)
    _apply_common(ax, spec)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Analytical function plot
# ---------------------------------------------------------------------------

def _render_function(spec: dict):
    """
    Plots one or more analytical functions defined as Python math expressions.
    Uses numpy (as `np`) in the evaluation context.

    Example spec:
      type: function
      xlabel: "$x$"
      ylabel: "$f(x)$"
      domain: [-5, 5]
      functions:
        - expr: "1 / (1 + np.exp(-x))"
          label: "Sigmoid $\\sigma(x)$"
          color: "#2b5797"
        - expr: "np.tanh(x)"
          label: "Tanh"
          color: "#d73a49"
    """
    fig, ax = _make_fig(spec)

    domain = spec.get("domain", [-5, 5])
    n_points = spec.get("points", 400)
    x = np.linspace(domain[0], domain[1], n_points)

    functions = spec.get("functions", [])
    if not functions and "expr" in spec:
        functions = [{"expr": spec["expr"], "label": spec.get("label", "")}]

    safe_globals = {"np": np, "__builtins__": {}}

    for i, fn in enumerate(functions):
        expr = fn.get("expr", "")
        label = fn.get("label", "")
        color = fn.get("color", DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        linestyle = fn.get("linestyle", "-")
        try:
            y = eval(expr, safe_globals, {"x": x})  # noqa: S307
        except Exception as e:
            y = np.zeros_like(x)
            label = f"[error: {e}]"
        ax.plot(x, y, color=color, label=label, linestyle=linestyle, linewidth=1.7)

    # Axes through origin if domain spans zero
    if domain[0] < 0 < domain[1]:
        ax.axhline(0, color="#888888", linewidth=0.6, linestyle="-")
        ax.axvline(0, color="#888888", linewidth=0.6, linestyle="-")

    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    _apply_common(ax, spec)
    fig.tight_layout()
    return fig
