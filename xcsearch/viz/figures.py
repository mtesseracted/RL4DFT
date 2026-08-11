"""Figures for the dashboard panels: data in, plotly figures out.

Kept apart from the app so that every panel can be built -- and tested --
without a browser, a server or a callback. A figure here is a pure function of
what ``data`` returned.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

from .data import CONDITIONS

# One accent per role, reused across panels so a colour means the same thing
# everywhere: the candidate, its baseline, and the reference.
CANDIDATE = "#d95926"
BASELINE = "#3987e5"
REFERENCE = "#222222"
# The live diagnostic colour, matching the `live (not stored)` line's amber in
# `app.py`. A fourth role rather than a reuse of CANDIDATE: "this candidate"
# and "this number is not in any record" are different claims, and a plot that
# spelled them the same colour would be making the first while hiding the
# second (D34).
LIVE = "#b8860b"
# The same role at a second density. Lighter and dotted rather than a new hue,
# because it is not a new kind of thing -- it is the same live quantity
# evaluated somewhere else, and a distinct colour would suggest otherwise.
LIVE_FAINT = "#d9ad4a"

_LAYOUT = dict(
    margin=dict(l=60, r=20, t=40, b=45),
    template="plotly_white",
    hovermode="closest",
)


def _live_label(what: str, live: dict) -> str:
    """A live trace's legend entry: what it is, whose, and that it is not stored.

    The candidate id is in the label because the plot repaints on the tick and
    the solve is cached per candidate: without it, scrubbing to another
    candidate would leave a curve on screen that belongs to the one before.
    """
    candidate = live.get("candidate")
    whose = f", candidate {candidate}" if candidate is not None else ""
    return f"{what}{whose} -- live, not stored"


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


_SCALES = ("linear", "logabs")

# Which way a cell moved, for the log scale, where the colour no longer says.
# Down is better because every metric on this figure is an error: the arrow
# points the way the number went, and for errors down is the good direction.
BETTER, WORSE, TIE = "▼", "▲", "="

# Magnitude ramp for the log scale: pale where the candidate and the baseline
# agree, saturated where they diverge. Sequential rather than diverging
# because on this scale colour carries one thing only -- how big the change
# is -- and the arrows carry which way.
#
# Yellow-to-green specifically, on two constraints that between them rule out
# most of the obvious choices. It shares no hue with the linear scale's RdBu,
# so nothing means one thing on one setting of the toggle and another thing on
# the other -- a red ramp would have "big difference" wearing the colour that
# means "candidate winning" one click away. And it starts at pale yellow
# rather than at white, so the floor an exact tie is drawn at cannot be
# mistaken for the white of a cell with no data at all.
_MAGNITUDE_SCALE = "YlGn"


def log_floor(differences: list[float]) -> float:
    """The magnitude an exact zero is drawn at, given what else is on the map.

    A logarithm has no value at zero and a heatmap still has to draw the cell,
    so zeros are clamped to one decade below the smallest difference that is
    not zero. That is the useful place for them: strictly below everything the
    run actually distinguished, so an exact tie is unmistakably the foot of the
    ramp, and no lower than that, so the decades in between keep the rest of
    the colour. Because the floor sits a full decade under the smallest real
    difference, nothing but an exact zero can reach it -- which is what lets
    the colourbar label that end "0" and be telling the truth.
    """
    nonzero = [abs(d) for d in differences if d]
    # Every cell a tie: the floor is arbitrary, so pick one and let the ramp
    # collapse to a single colour, which is exactly what the figure means.
    return min(nonzero) / 10.0 if nonzero else 1.0


def _log_ticks(floor: float, top: float) -> dict:
    """Colourbar ticks at round magnitudes, placed at their log height.

    ``log10`` values are numbers about the transform; the differences they
    came from are numbers about the functional, so the bar is labelled in
    those and positioned in these. The bottom tick is the floor, labelled for
    what alone can reach it.
    """
    low = math.log10(floor)
    steps = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0)
    decades = [d for d in steps if low < math.log10(d) <= top]
    return dict(
        tickmode="array",
        tickvals=[low] + [math.log10(d) for d in decades],
        ticktext=["0"] + [f"{d:g}" for d in decades],
    )


def _difference(row: dict) -> tuple[float, bool]:
    """A cell's difference from its baseline, and whether it is a relative one.

    Relative where the baseline is nonzero; absolute where it is zero -- a
    satisfied exact condition -- since dividing by it would be infinite rather
    than informative.
    """
    value, baseline = float(row["value"]), float(row["baseline"])
    relative = bool(baseline)
    return ((value - baseline) / abs(baseline) if relative else value - baseline), relative


def heatmap(rows: list[dict], candidate_label="candidate", baseline_label="baseline",
            scale: str = "linear", against_zero: bool = False):
    """Regime x metric, as the *difference* from the baseline.

    Differences rather than raw values, because the raw numbers of different
    metrics share no scale: an energy error and a potential error on one colour
    axis would make the larger one the only thing visible. Relative to a
    baseline, every cell reads the same way.

    ``scale`` picks what colour means.

    ``"linear"`` is the signed difference on a diverging scale: colour is
    proportional to it and carries the direction as well, red below the
    baseline and blue above. Proportional is what you want when the cells are
    the same size; it is also why the largest cell owns the scale, and one cell
    a hundred times the rest leaves them all white.

    ``"logabs"`` is ``log10`` of the *magnitude* on a sequential scale, with
    the direction moved out of the colour and into an arrow drawn in the cell:
    ``BETTER`` for a candidate below its baseline, ``WORSE`` for above, ``TIE``
    for exactly equal. Splitting the two makes the whole ramp available for
    magnitude instead of half of it, so a row spanning a factor of ten
    separates into ten distinguishable steps rather than into "large" and
    "white". Exact zeros are drawn at the foot of the ramp; see ``log_floor``.

    ``against_zero`` says the baseline is the exact reference rather than
    another candidate, so the cells are the candidate's own errors. Two things
    follow and both are done here rather than asked of the caller: the arrows
    go, because every error is above zero and an arrow that is always the same
    arrow is furniture; and the linear scale drops its diverging colours for
    the magnitude ramp, because half a diverging scale is the half that would
    never be used.

    Either way the transform is a property of the *colour*. Hovering a cell
    reports what it measured, so nothing the figure claims depends on which
    scale is showing.
    """
    if scale not in _SCALES:
        raise ValueError(f"scale must be one of {_SCALES}, not {scale!r}")
    # Conditions are dropped here and not merely by the caller, so that no
    # route back to the half-empty grid exists: they have no regime, so they
    # cannot be a column of a figure whose columns are regimes. See
    # `condition_bars`, which draws them, and `data.split_conditions`, which
    # is how the panel gets hold of them.
    rows = [row for row in rows if row["bin"] != CONDITIONS]
    if not rows:
        return _empty("no evaluations for this candidate")

    bins = sorted({row["bin"] for row in rows})
    metrics = sorted({row["metric"] for row in rows})
    lookup = {(row["bin"], row["metric"]): row for row in rows}

    # Pass one: the differences, which both scales are drawings of.
    differences, hover = [], []
    for metric in metrics:
        difference_row, hover_row = [], []
        for bin_name in bins:
            row = lookup.get((bin_name, metric))
            if row is None or row["value"] is None or row["baseline"] is None:
                difference_row.append(None)
                hover_row.append("no data")
                continue
            difference, relative = _difference(row)
            difference_row.append(difference)
            if against_zero:
                # Nothing to subtract, so nothing to report but the number and
                # what it is a mean of.
                hover_row.append(
                    f"{candidate_label}<br>mean {metric}: {float(row['value']):.3e}"
                    f"<br>{baseline_label}: 0"
                )
            else:
                hover_row.append(
                    f"{candidate_label}: {float(row['value']):.3e}"
                    f"<br>{baseline_label}: {float(row['baseline']):.3e}"
                    f"<br>{'relative' if relative else 'absolute'} difference: "
                    f"{difference:+.3e}"
                )
        differences.append(difference_row)
        hover.append(hover_row)

    flat = [d for row in differences for d in row if d is not None]
    colorbar = dict(title="error" if against_zero else "vs baseline")

    if scale == "linear" and against_zero:
        z = differences
        colour = dict(colorscale=_MAGNITUDE_SCALE, zmin=0.0,
                      zmax=max(flat, default=0.0))
        arrows = []
    elif scale == "linear":
        z = differences
        colour = dict(colorscale="RdBu", zmid=0.0)
        arrows = []
    else:
        floor = log_floor(flat)
        low = math.log10(floor)
        z = [[None if d is None else math.log10(max(abs(d), floor)) for d in row]
             for row in differences]
        top = max((cell for row in z for cell in row if cell is not None), default=low)
        colorbar.update(_log_ticks(floor, top))
        colour = dict(colorscale=_MAGNITUDE_SCALE, zmin=low, zmax=top)
        arrows = [] if against_zero else _arrows(differences, z, bins, metrics, low, top)

    figure = go.Figure(
        go.Heatmap(
            z=z, x=bins, y=metrics, text=hover,
            hovertemplate="%{y} / %{x}<br>%{text}<extra></extra>",
            colorbar=colorbar, **colour,
        )
    )
    figure.update_layout(**_LAYOUT, xaxis_title="regime", yaxis_title="metric",
                         annotations=arrows)
    return figure


def condition_bars(rows: list[dict], candidate_label="candidate",
                   baseline_label="baseline", against_zero: bool = False):
    """The rung-0 conditions, as raw violations rather than as differences.

    These left the heatmap because they never belonged in a grid: they are
    measured on no system, so they have no regime to be a column of, and the
    metrics they carry appear in no other bin. Half of that figure was empty
    for a structural reason and not for want of data.

    Raw values rather than the difference from a baseline, which is the other
    half of the same argument. Every one of these is a *violation* measure
    with an absolute meaning at zero -- the constraint is satisfied, exactly
    -- and "0.0e+00 against a baseline of 3.2e-02" says something a relative
    difference cannot say at all, since it would be dividing by the number
    that matters. So both bars are drawn and the reader compares them, which
    is also why this figure ignores the colour-scale toggle: there is no
    transform here to choose, only two numbers and their names.

    ``against_zero`` drops the baseline series. The exact reference satisfies
    every condition by construction, so that series is a row of zeros, and a
    bar of height zero is an invisible claim rather than a made one -- the
    sub-label says it instead.
    """
    usable = [row for row in rows if row.get("value") is not None]
    if not usable:
        return _empty("no conditions recorded for this candidate")

    metrics = [row["metric"] for row in sorted(usable, key=lambda r: r["metric"])]
    values = [float(row["value"]) for row in sorted(usable, key=lambda r: r["metric"])]

    figure = go.Figure()
    series = [(candidate_label, values, CANDIDATE)]
    if not against_zero:
        baselines = [None if row["baseline"] is None else float(row["baseline"])
                     for row in sorted(usable, key=lambda r: r["metric"])]
        series.append((baseline_label, baselines, BASELINE))

    for name, heights, colour in series:
        figure.add_bar(
            y=metrics, x=heights, name=name, orientation="h",
            marker_color=colour,
            text=[("" if h is None else f"{h:.3e}") for h in heights],
            textposition="outside", cliponaxis=False,
            hovertemplate=f"{name}<br>%{{y}}: %{{x:.4e}}<extra></extra>",
        )

    figure.update_layout(
        **_LAYOUT, barmode="group",
        xaxis_title="violation (0 = satisfied)", yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=60 + 46 * len(metrics),
    )
    # Room for the value labels drawn outside the longest bar.
    figure.update_xaxes(rangemode="tozero")
    return figure


def _arrows(differences, z, bins, metrics, low: float, top: float) -> list[dict]:
    """One direction glyph per populated cell, in a colour that survives it.

    Annotations rather than the trace's own text because a heatmap draws all
    of its text in one colour, and this ramp runs from pale yellow to dark
    red: a single choice is invisible at one end or the other. An annotation
    carries its own colour, so each glyph is picked against the cell it sits
    on.
    """
    span = top - low
    out = []
    for row_index, metric in enumerate(metrics):
        for column, bin_name in enumerate(bins):
            difference = differences[row_index][column]
            if difference is None:
                continue
            glyph = TIE if difference == 0 else (BETTER if difference < 0 else WORSE)
            shade = (z[row_index][column] - low) / span if span else 0.0
            out.append(dict(
                x=bin_name, y=metric, text=glyph, showarrow=False,
                font=dict(size=11, color="#f4f4f4" if shade > 0.62 else "#333333"),
            ))
    return out


def density_figure(arrays: dict | None, wells: list[list[float]], name: str,
                   live: dict | None = None):
    """n(x) for one system, with its wells marked on the axis.

    ``live`` is the self-consistent density from a live solve (D34), or None
    when nobody has asked for one. It is drawn dashed, in the live colour, and
    labelled *not stored* in the legend -- the amended D13's third option is
    "explicitly marked as a live diagnostic", and on a plot the marking has to
    be the line itself rather than a sentence somewhere else on the page.

    The physics is not computed here. The array arrives through the session,
    which got it from the SCF service's worker thread; `figures` imports
    nothing it could compute a density with, and `test_import_audit` is what
    keeps that true.
    """
    if arrays is None:
        return _empty("the .npz cache for this system is not reachable from here")

    figure = go.Figure()
    figure.add_scatter(
        x=arrays["x"], y=arrays["n_ref"], mode="lines", name="n_ref",
        line=dict(color=REFERENCE, width=2),
    )
    if live is not None and live.get("n") is not None:
        figure.add_scatter(
            x=live["x"], y=live["n"], mode="lines",
            name=_live_label("n [self-consistent]", live),
            line=dict(color=LIVE, width=2, dash="dash"),
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


def potential_figure(arrays: dict | None, name: str, live: dict | None = None):
    """The reference XC potential for one system, and the candidate's beside it.

    The candidate's v_xc used to be absent here, on the grounds that computing
    it would be physics and the dashboard does none. That was a Stage 2
    answer (STAGE2B.md's detail-panel question) and **D34 retired it**: the
    amended D13 keeps the import half -- this module still imports nothing it
    could compute a potential with -- and adds a third option to the other
    half, *explicitly marked as a live diagnostic*. Both curves below are
    injected arrays, marked.

    Two of them, because they answer different questions:

    ``v_xc[n_ref]``
        the candidate's potential on the *reference* density. Directly
        comparable to ``vxc_ref``: same density, same gauge (``_vxc_l2_error``
        compares them absolutely, with no shift), and it is the pointwise
        version of the ``vxc_l2_error`` cell already on the heatmap. Needs no
        solve, so it is the one that appears first.
    ``v_xc[n_scf]``
        the potential the self-consistent solve converged with. It appears
        after a solve, dotted rather than dashed, and it is *not* an
        apples-to-apples comparison with ``vxc_ref`` -- the gap includes the
        density difference. The legend says which density each is on rather
        than leaving that to be inferred.
    """
    if arrays is None or arrays.get("vxc_ref") is None:
        return _empty("no reference potential cached for this system (pre-2.4 build)")

    figure = go.Figure()
    figure.add_scatter(
        x=arrays["x"], y=arrays["vxc_ref"], mode="lines", name="vxc_ref",
        line=dict(color=REFERENCE, width=2),
    )
    live = live or {}
    if live.get("vxc_at_reference") is not None:
        figure.add_scatter(
            x=live["x"], y=live["vxc_at_reference"], mode="lines",
            name=_live_label("v_xc [at n_ref]", live),
            line=dict(color=LIVE, width=2, dash="dash"),
        )
    if live.get("vxc_self_consistent") is not None:
        figure.add_scatter(
            x=live["x"], y=live["vxc_self_consistent"], mode="lines",
            name=_live_label("v_xc [at n_scf]", live),
            line=dict(color=LIVE_FAINT, width=2, dash="dot"),
        )
    if live.get("reference_failure"):
        figure.add_annotation(
            text=f"no candidate v_xc: {live['reference_failure']}",
            showarrow=False, x=0.5, y=1.0, xref="paper", yref="paper",
            font=dict(size=10, color="#b3261e"),
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
