"""Figures for the dashboard panels: data in, plotly figures out.

Kept apart from the app so that every panel can be built -- and tested --
without a browser, a server or a callback. A figure here is a pure function of
what ``data`` returned.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# One accent per role, reused across panels so a colour means the same thing
# everywhere: the candidate, its baseline, and the reference.
CANDIDATE = "#d95926"
BASELINE = "#3987e5"
REFERENCE = "#222222"

_LAYOUT = dict(
    margin=dict(l=60, r=20, t=40, b=45),
    template="plotly_white",
    hovermode="closest",
)


def _empty(message: str) -> go.Figure:
    """A figure that says why it is empty rather than rendering blank axes."""
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, x=0.5, y=0.5,
                          xref="paper", yref="paper")
    figure.update_layout(**_LAYOUT, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return figure


def history_strip(history: list[dict], selected: int | None = None) -> go.Figure:
    """Score per trial with the best-so-far envelope, and the scrubber's mark."""
    if not history:
        return _empty("no candidates in this run yet")

    trials = [row["trial"] for row in history]
    figure = go.Figure()
    figure.add_scatter(
        x=trials, y=[row["score"] for row in history], mode="markers",
        name="candidate", marker=dict(size=5, color=CANDIDATE, opacity=0.65),
        customdata=[row["candidate_id"] for row in history],
        hovertemplate="trial %{x}<br>score %{y:.3e}<br>candidate %{customdata}<extra></extra>",
    )
    figure.add_scatter(
        x=trials, y=[row["best_so_far"] for row in history], mode="lines",
        name="best so far", line=dict(color=REFERENCE, width=2),
    )
    if selected is not None:
        for row in history:
            if row["candidate_id"] == selected:
                figure.add_vline(x=row["trial"], line=dict(color=BASELINE, width=1.5,
                                                           dash="dot"))
                break
    figure.update_layout(
        **_LAYOUT, yaxis_type="log", xaxis_title="trial", yaxis_title="train score",
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return figure


def heatmap(rows: list[dict], candidate_label="candidate", baseline_label="baseline"):
    """Regime x metric, as the *difference* from the baseline.

    Differences rather than raw values, because the raw numbers of different
    metrics share no scale: an energy error and a potential error on one colour
    axis would make the larger one the only thing visible. Relative to a
    baseline, every cell reads the same way -- below zero is better than the
    baseline.
    """
    if not rows:
        return _empty("no evaluations for this candidate")

    bins = sorted({row["bin"] for row in rows})
    metrics = sorted({row["metric"] for row in rows})
    lookup = {(row["bin"], row["metric"]): row for row in rows}

    z, text = [], []
    for metric in metrics:
        z_row, text_row = [], []
        for bin_name in bins:
            row = lookup.get((bin_name, metric))
            if row is None or row["value"] is None or row["baseline"] is None:
                z_row.append(None)
                text_row.append("no data")
                continue
            value, baseline = float(row["value"]), float(row["baseline"])
            # Relative difference where the baseline is nonzero; absolute
            # difference where it is zero (a satisfied exact condition), since
            # dividing by it would be infinite rather than informative.
            z_row.append((value - baseline) / abs(baseline) if baseline else value - baseline)
            text_row.append(
                f"{candidate_label}: {value:.3e}<br>{baseline_label}: {baseline:.3e}"
            )
        z.append(z_row)
        text.append(text_row)

    figure = go.Figure(
        go.Heatmap(
            z=z, x=bins, y=metrics, text=text,
            hovertemplate="%{y} / %{x}<br>%{text}<extra></extra>",
            colorscale="RdBu", zmid=0.0, colorbar=dict(title="vs baseline"),
        )
    )
    figure.update_layout(**_LAYOUT, xaxis_title="regime", yaxis_title="metric")
    return figure


def density_figure(arrays: dict | None, wells: list[list[float]], name: str):
    """n(x) for one system, with its wells marked on the axis."""
    if arrays is None:
        return _empty("the .npz cache for this system is not reachable from here")

    figure = go.Figure()
    figure.add_scatter(
        x=arrays["x"], y=arrays["n_ref"], mode="lines", name="n_ref",
        line=dict(color=REFERENCE, width=2),
    )
    peak = float(np.max(arrays["n_ref"]))
    for charge, position in wells:
        figure.add_scatter(
            x=[position], y=[0.0], mode="markers+text", showlegend=False,
            marker=dict(symbol="triangle-up", size=8 + 3 * charge, color="#666"),
            text=[f"Z={charge:g}"], textposition="bottom center",
            hovertemplate=f"well Z={charge:g} at x={position:.2f}<extra></extra>",
        )

    visible = arrays["n_ref"] > 1e-4 * peak
    if visible.any():
        margin = 1.5
        figure.update_xaxes(
            range=[float(arrays["x"][visible].min()) - margin,
                   float(arrays["x"][visible].max()) + margin]
        )
    figure.update_layout(**_LAYOUT, title=name, xaxis_title="x (a.u.)", yaxis_title="n(x)")
    return figure


def potential_figure(arrays: dict | None, name: str):
    """The reference XC potential for one system, when 2.4's fields are cached.

    The candidate's own v_xc is deliberately absent: computing it here would be
    physics, and the dashboard does none. What the search measured about it --
    ``vxc_l2_error`` -- is a record, and the heatmap shows that.
    """
    if arrays is None or arrays.get("vxc_ref") is None:
        return _empty("no reference potential cached for this system (pre-2.4 build)")

    figure = go.Figure()
    figure.add_scatter(
        x=arrays["x"], y=arrays["vxc_ref"], mode="lines", name="vxc_ref",
        line=dict(color=REFERENCE, width=2),
    )
    n_ref = arrays["n_ref"]
    visible = n_ref > 1e-6 * float(np.max(n_ref))
    if visible.any():
        figure.update_xaxes(
            range=[float(arrays["x"][visible].min()), float(arrays["x"][visible].max())]
        )
    figure.update_layout(
        **_LAYOUT, title=f"{name}: reference XC potential",
        xaxis_title="x (a.u.)", yaxis_title="v_xc(x)",
    )
    return figure
