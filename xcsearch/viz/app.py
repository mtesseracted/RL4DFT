"""The Dash app (feature 4, sub-stage 2.7).

    python -m xcsearch.viz.app --store data/exp1d/records.sqlite

Panels: the formula of the selected candidate, a regime x metric heatmap
against a baseline of your choosing, a per-system drill-down, and a history
scrubber over a run with its best-so-far strip. All of it re-reads the store on
an interval, so the panels track a search that is still running -- the store is
in WAL mode and these connections are read-only, so nothing here can block or
corrupt the writer.

The human endpoint is optional: pass an object with ``bounds`` and
``submit(theta)`` (``HumanOptimizer`` is one) and the proposal panel appears.
It is *duck-typed* on purpose -- the dashboard renders records and feeds a
queue, and giving it an import of an optimizer adapter would be the first
crack in "viz imports no physics".
"""

from __future__ import annotations

import argparse
import os

from dash import Dash, Input, Output, State, dcc, html, no_update

from . import data, figures

_POLL_MS = 3000


_HINT_STYLE = {
    "display": "inline-block", "marginLeft": "7px", "width": "15px", "height": "15px",
    "lineHeight": "15px", "textAlign": "center", "borderRadius": "50%",
    "border": "1px solid #b6bcc4", "color": "#6b737c", "fontSize": "10px",
    "fontWeight": 400, "cursor": "help", "verticalAlign": "middle",
}


def _card(title: str, *children, hint: str | None = None) -> html.Div:
    """One panel. ``hint`` becomes a marker beside the heading explaining it.

    A native ``title`` attribute rather than a tooltip component: the text is
    prose, it has to survive with no callback and no CSS asset, and the
    dashboard ships no assets directory. The cost is the browser's own delay
    before it appears, which is the right trade for a panel legend nobody
    needs on every glance.
    """
    heading = [html.Span(title)]
    if hint:
        heading.append(html.Span("i", id=f"hint-{title.split()[0]}", title=hint,
                                 style=_HINT_STYLE))
    return html.Div(
        [html.H3(heading, style={"margin": "0 0 8px", "fontSize": "15px"}), *children],
        style={
            "border": "1px solid #e2e5ea", "borderRadius": "8px",
            "padding": "12px 14px", "marginBottom": "14px", "background": "white",
        },
    )


# The panel legends. Kept as constants rather than inline so that what a panel
# claims to show sits next to what it shows, and so a test can assert every
# panel has one -- an unexplained heatmap is the failure mode here, not a
# missing widget.
_RUN_HINT = (
    "Which search to look at, and which trial of it every panel below "
    "describes.\n\n"
    "Each point is one trial's score: the number the ranking policy returned "
    "and the optimizer was told when the trial was proposed, so lower is "
    "better. The black line is the best score reached up to that trial.\n\n"
    "The slider opens on the best trial of the run and follows the best as a "
    "running search improves on it; drag it and it stays where you put it. "
    "Its value is a trial index, which is not the candidate id -- the line "
    "underneath names both, and the rest of the store keys on the candidate id."
)

_FORMULA_HINT = (
    "The selected trial's functional, written out: LDA exchange scaled by c0, "
    "plus the density raised to 1/2, 1, 3/2 and 2 with coefficients c1..c4, "
    "plus LDA correlation left alone. LDA itself is theta = [1, 0, 0, 0, 0], "
    "which is why c0 near 1 with small tail terms reads as a correction to it "
    "rather than a replacement for it.\n\n"
    "theta is the coefficient vector as proposed. score is the value stored "
    "for that proposal at the time it was made, not a re-derivation of it here."
)

_HEATMAP_HINT = (
    "How the selected candidate does against a reference functional -- LDA "
    "unless you pick another -- everywhere it was measured.\n\n"
    "Rows are metrics. exc_sq_error is the squared error in the XC energy on "
    "the exact density. vxc_l2_error is the density-weighted L2 error in the "
    "XC potential. one_step_dn is how far one Kohn-Sham step moves the "
    "reference density: a functional can get the energy right on the exact "
    "density and still push it somewhere else on the first step.\n\n"
    "Columns are bins. Four are geometry regimes -- equilibrium, stretched, "
    "dissociating, and isolated atoms -- and 'conditions' holds the exact "
    "constraints that belong to no single system: the uniform-gas limit, "
    "one-electron self-interaction, and non-positivity of exchange.\n\n"
    "Each cell is the mean over the systems in that bin, expressed against "
    "the baseline as (candidate - baseline) / |baseline|, or as the plain "
    "difference where the baseline is exactly zero -- a condition it already "
    "satisfies. Hover a cell for both raw numbers and the difference between "
    "them, on either scale.\n\n"
    "linear / log magnitude changes the colour only, never the numbers.\n\n"
    "On linear, colour is the signed difference: every metric here is an "
    "error, so a cell below its baseline is the candidate winning, and on "
    "this scale that is the red end, blue is losing, white is a tie. Colour "
    "is proportional to the difference, which means the largest cell owns the "
    "scale -- a cell a hundred times smaller is white whatever it says.\n\n"
    "On log magnitude, colour is log10 of the size of the difference alone, "
    "pale where the two agree and saturated where they diverge, and the "
    "direction moves out of the colour and into the arrow drawn in each cell: "
    "▼ the candidate is below its baseline (better, these being errors), ▲ "
    "above it, = exactly equal. Splitting the two gives the whole ramp to "
    "magnitude instead of half of it, so cells that differ by a factor of ten "
    "look ten steps apart rather than 'large' and 'white'. The colourbar is "
    "labelled in the differences themselves, not in log units. Exact zeros "
    "sit at the foot of the ramp, one decade below the smallest real "
    "difference on the map -- which is the only thing that can land there, so "
    "that end of the bar reads 0.\n\n"
    "The line above the figure names the population the means were taken "
    "over. Comparing candidates from two different populations uses only the "
    "systems they share and bin alike."
)

_SYSTEM_HINT = (
    "One system of the run's population, and what the selected candidate "
    "scored on it -- the stored per-system records behind one column of the "
    "heatmap above.\n\n"
    "The curves are cached reference data: the exact density n(x) with the "
    "wells marked on the axis, and the reference XC potential where the build "
    "cached one. They are a property of the system, so they do not change as "
    "you move the slider -- only the metric line does.\n\n"
    "The candidate's own density is deliberately not drawn. Producing it "
    "means an SCF solve, and the dashboard does no physics; it renders "
    "records. For reference-vs-LDA-vs-candidate density curves, run "
    "`python -m xcsearch.worlds.exp1d.plots` from the repo root, which solves "
    "each system self-consistently and writes them under data/plots/."
)

_PROPOSAL_HINT = (
    "Hand-place a functional in the same search the optimizer is running "
    "against. The sliders are the coefficients of the formula panel, bounded "
    "by the same box the optimizer proposes inside; submitting queues one "
    "theta, which the search picks up as its next trial and records like any "
    "other, so it lands in the history strip beside the machine's."
)


# What a human endpoint has to provide (B6). Duck-typed, but checked: the
# dashboard renders records and feeds a queue, and importing an optimizer
# adapter to type-check it would be the first crack in "viz imports no
# physics". `bounds` is read by the layout builder, so a missing one already
# fails at create_app; `submit` and `pending_submissions` are read inside the
# submit callback, which is the wrong place to discover a contract violation
# -- a person clicks the button, the callback raises into the browser
# console, and the session looks alive.
_HUMAN_CONTRACT = ("bounds", "submit", "pending_submissions")


def check_human_endpoint(human) -> None:
    """Fail at ``create_app`` if the human endpoint cannot serve the panel.

    All three attributes, not just the one that happens to be read first, and
    at construction rather than on click.
    """
    missing = [name for name in _HUMAN_CONTRACT if not hasattr(human, name)]
    if missing:
        raise TypeError(
            f"{type(human).__name__} is missing {missing}: a human endpoint "
            f"needs {list(_HUMAN_CONTRACT)} -- `bounds` to lay the sliders "
            "out, `submit(theta)` to queue a proposal, and "
            "`pending_submissions` to say how many are waiting. "
            "HumanOptimizer is one."
        )
    if not callable(getattr(human, "submit")):
        raise TypeError(
            f"{type(human).__name__}.submit is not callable: the submit "
            "button calls it with a theta and expects the accepted theta back."
        )


def build_layout(store_path: str, human=None) -> html.Div:
    run_options = [
        {"label": f"run {r['id']} - {r['optimizer']} ({r['candidates']} candidates)",
         "value": r["id"]}
        for r in data.runs(store_path)
    ]
    baseline_options = [
        {"label": c["label"], "value": c["id"]} for c in data.baseline_choices(store_path)
    ]
    # One option per system *name*, not per version. A content-changed rebuild
    # adds a version (D17), and two options reading "atom_Z1" with the same
    # value would be a menu the reader cannot choose between -- the drill-down
    # resolves a name through the selected candidate's own rows anyway, so the
    # version is not the dropdown's to pick. `dict.fromkeys` keeps first sight,
    # and `data.systems` orders newest first, so the name resolves the way
    # `density` resolves it.
    system_options = [
        {"label": name, "value": name}
        for name in dict.fromkeys(s["name"] for s in data.systems(store_path))
    ]

    panels = [
        html.H1("xcsearch", style={"fontSize": "20px", "marginBottom": "2px"}),
        html.Div(os.path.abspath(store_path), style={"color": "#7a828c", "fontSize": "12px"}),
        dcc.Interval(id="tick", interval=_POLL_MS),
        # The watchdog line (B10). Empty while the session is healthy, so it
        # costs nothing to look at and means something when it is not.
        html.Div(id="worker-status",
                 style={"color": "#b3261e", "fontSize": "13px", "margin": "4px 0"}),
        # The resolved candidate id, in one place (B9). The scrubber's value
        # is a *trial index*; every dependent panel needs the *candidate id*
        # that index resolves to, and each one resolving it separately meant
        # three reads of the same history per poll and a race on a run switch
        # -- a callback could fire with the new run and the old index.
        dcc.Store(id="selected"),
        _card(
            "run",
            dcc.Dropdown(
                id="run", options=run_options,
                value=run_options[0]["value"] if run_options else None,
                clearable=False,
            ),
            dcc.Graph(id="history", config={"displayModeBar": False}),
            html.Div(
                # "trial", because that is what the slider's value is. It was
                # labelled "candidate" while carrying the trial index, so the
                # UI was actively wrong rather than merely ambiguous (B9).
                [html.Span("trial", style={"marginRight": "8px"}),
                 dcc.Slider(id="scrubber", min=0, max=0, step=1, value=0,
                            tooltip={"placement": "bottom"})],
                style={"marginTop": "6px"},
            ),
            html.Div(id="selection", style={"fontSize": "12px", "color": "#4a525c",
                                            "marginTop": "2px"}),
            hint=_RUN_HINT,
        ),
        _card("formula", dcc.Markdown(id="formula", mathjax=True), html.Div(id="theta"),
              hint=_FORMULA_HINT),
        _card(
            "regime x metric, against a baseline",
            dcc.Dropdown(
                id="baseline", options=baseline_options,
                value=data.default_baseline(store_path), clearable=False,
            ),
            # What set the two means were taken over (D19). A comparison
            # across populations is over their intersection, and saying so is
            # part of the comparison rather than a footnote to it.
            html.Div(id="heatmap-scope",
                     style={"fontSize": "12px", "color": "#7a828c", "margin": "2px 0"}),
            # Colour scale, not data: both settings draw the same differences.
            dcc.RadioItems(
                id="heatmap-scale",
                options=[{"label": "linear", "value": "linear"},
                         {"label": "log magnitude", "value": "logabs"}],
                value="linear", inline=True,
                style={"fontSize": "12px", "color": "#4a525c"},
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "12px"},
            ),
            dcc.Graph(id="heatmap", config={"displayModeBar": False}),
            hint=_HEATMAP_HINT,
        ),
        _card(
            "system detail",
            dcc.Dropdown(
                id="system", options=system_options,
                value=system_options[0]["value"] if system_options else None,
                clearable=False,
            ),
            html.Div(id="system-metrics", style={"fontSize": "13px", "margin": "6px 0"}),
            dcc.Graph(id="density", config={"displayModeBar": False}),
            dcc.Graph(id="potential", config={"displayModeBar": False}),
            hint=_SYSTEM_HINT,
        ),
    ]

    if human is not None:
        check_human_endpoint(human)
        panels.append(
            _card(
                "propose a functional",
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(f"c{i}", style={"width": "34px",
                                                           "display": "inline-block"}),
                                dcc.Slider(
                                    id={"type": "theta", "index": i},
                                    min=lo, max=hi, step=(hi - lo) / 200.0,
                                    value=1.0 if i == 0 else 0.0,
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ],
                            style={"marginBottom": "4px"},
                        )
                        for i, (lo, hi) in enumerate(human.bounds)
                    ]
                ),
                html.Button("submit", id="submit", n_clicks=0),
                html.Div(id="submitted", style={"fontSize": "13px", "marginTop": "6px"}),
                hint=_PROPOSAL_HINT,
            )
        )

    return html.Div(
        panels,
        style={
            "maxWidth": "900px", "margin": "0 auto", "padding": "18px",
            "fontFamily": "system-ui, sans-serif", "background": "#f6f7f9",
        },
    )


def best_trial(rows: list[dict]) -> int:
    """The trial index of the lowest score in a history, earliest on a tie.

    Earliest rather than latest because that is the trial the best-so-far line
    steps down at: it is where the run *found* the score, and a later trial
    that merely ties it discovered nothing.
    """
    return min(range(len(rows)), key=lambda index: rows[index]["score"])


def scrubber_value(rows: list[dict], current, previous, run_id) -> int:
    """Where the trial slider sits after a poll.

    A run opens on its best trial rather than on trial 0: the first proposal
    of a search is the one nobody wants to look at, and opening on the end is
    no better once the run has finished, since the last trial is only the last
    thing tried. Thereafter the scrubber follows the best while it is parked
    on it -- so a running search's panels track the leader, the way the old
    end-following behaviour tracked the frontier -- and stays put the moment
    somebody drags it.

    ``previous`` is how that intent is read. It is the payload this callback
    published last time, holding the index it *set*; a slider that no longer
    agrees with it has been moved by a person, and a person's position
    outranks any rule here.
    """
    last = len(rows) - 1
    best = best_trial(rows)
    if not previous or previous.get("run_id") != run_id or current is None:
        return best                                  # a run just loaded
    if current != previous.get("trial"):
        return min(current, last)                    # dragged: leave it alone
    if previous.get("trial") == previous.get("best_trial"):
        return best                                  # parked on the best
    return min(current, last)


def scope_label(scope: dict) -> str:
    """One line naming the population a comparison was taken over (D19).

    The dashboard's counterpart to the CLI's ``_scope_line``: same sentence,
    same reason, same count. A cross-population heatmap is computed over the
    systems both candidates were measured on *and binned alike*, and the
    reader is told how many that was.
    """
    n = scope["comparable_systems"]
    if scope["candidate_population"] is None:
        return "no population recorded for this candidate"
    if scope["same_population"]:
        return f"over population {scope['candidate_population']} ({n} systems)"
    line = (
        f"over the {n} shared systems of populations "
        f"{scope['candidate_population']} and {scope['baseline_population']}"
    )
    dropped = scope["shared_systems"] - n
    if dropped:
        line += f" ({dropped} more shared but binned differently)"
    return line


def create_app(store_path: str, human=None, status=None) -> Dash:
    """Build the app. Every callback re-reads the store, so panels live-tail.

    ``status`` is an optional zero-argument callable returning a line to show
    above the panels -- ``serve_human_session`` passes one that reports a dead
    search worker (B10). A callable rather than a queue or an object: the
    dashboard polls it, and everything it knows about the search stays "a
    string someone else computed".
    """
    app = Dash(__name__)
    app.title = "xcsearch"
    app.layout = build_layout(store_path, human)

    @app.callback(
        Output("scrubber", "max"),
        Output("scrubber", "value"),
        Output("history", "figure"),
        Output("selected", "data"),
        Output("selection", "children"),
        Input("run", "value"),
        Input("tick", "n_intervals"),
        State("scrubber", "value"),
        State("selected", "data"),
    )
    def _history(run_id, _tick, current, previous):
        """The one place a trial index becomes a candidate id (B9).

        Reads the history once per poll and publishes both the figure and the
        resolved id, so the dependent panels neither re-read it nor re-resolve
        it -- and cannot resolve it against a different run than this one.

        Where the scrubber lands is ``scrubber_value``'s decision; publishing
        the trial it chose is what lets the next poll tell a dragged slider
        from one this callback placed.
        """
        if run_id is None:
            return 0, 0, figures.history_strip([]), None, ""
        rows = data.history(store_path, run_id)
        if not rows:
            return 0, 0, figures.history_strip([]), None, ""
        last = len(rows) - 1
        best = best_trial(rows)
        value = scrubber_value(rows, current, previous, run_id)

        row = rows[value]
        selected = row["candidate_id"]
        return (
            last,
            value,
            figures.history_strip(rows, selected),
            {"run_id": run_id, "trial": value, "candidate_id": selected,
             "score": row["score"], "best_trial": best},
            # Both identities, together: they are different numbers and the
            # panels below name the candidate one.
            f"trial {value} / candidate {selected}"
            + (f" - best of {last + 1} trials" if value == best else
               f" - best is trial {best}"),
        )

    @app.callback(
        Output("formula", "children"),
        Output("theta", "children"),
        Output("heatmap", "figure"),
        Output("heatmap-scope", "children"),
        Input("selected", "data"),
        Input("baseline", "value"),
        Input("heatmap-scale", "value"),
        Input("tick", "n_intervals"),
    )
    def _candidate(selected, baseline_id, scale, _tick):
        if not selected or baseline_id is None:
            return "*no candidate selected*", "", figures.heatmap([]), ""
        candidate_id = selected["candidate_id"]

        record = data.candidate(store_path, candidate_id) or {}
        baseline = data.candidate(store_path, baseline_id) or {}
        latex = record.get("form_latex", "")
        theta = ", ".join(f"{c:.4f}" for c in record.get("theta", []))
        return (
            f"$${latex}$$",
            html.Div(
                [
                    html.Span(f"candidate {candidate_id}", style={"fontWeight": 600}),
                    html.Span(f"  theta = [{theta}]", style={"color": "#4a525c"}),
                    # The stored score: the number the optimizer was told
                    # (D20), not a re-derivation of it.
                    html.Span(f"  score = {selected['score']:.4e}",
                              style={"color": "#4a525c"}),
                ],
                style={"fontSize": "13px", "fontFamily": "monospace"},
            ),
            figures.heatmap(
                data.heatmap(store_path, candidate_id, baseline_id),
                candidate_label=f"candidate {candidate_id}",
                baseline_label=baseline.get("label") or f"candidate {baseline_id}",
                scale=scale or "linear",
            ),
            scope_label(data.heatmap_scope(store_path, candidate_id, baseline_id)),
        )

    @app.callback(
        Output("system-metrics", "children"),
        Output("density", "figure"),
        Output("potential", "figure"),
        Input("system", "value"),
        Input("selected", "data"),
        Input("tick", "n_intervals"),
    )
    def _system(system, selected, _tick):
        if system is None:
            return "", figures.density_figure(None, [], ""), figures.potential_figure(None, "")

        wells = next(
            (s["wells"] for s in data.systems(store_path) if s["name"] == system), []
        )
        arrays = data.density(store_path, system)

        metrics_text = ""
        if selected:
            metrics = data.system_metrics(
                store_path, selected["candidate_id"], system
            )
            metrics_text = "   ".join(f"{k} = {v:.4e}" for k, v in sorted(metrics.items()))

        return (
            html.Span(metrics_text, style={"fontFamily": "monospace"}),
            figures.density_figure(arrays, wells, system),
            figures.potential_figure(arrays, system),
        )

    @app.callback(
        Output("worker-status", "children"),
        Input("tick", "n_intervals"),
    )
    def _worker_status(_tick):
        """The watchdog line (B10). Silent unless there is something to say."""
        return status() if status is not None else ""

    if human is not None:
        @app.callback(
            Output("submitted", "children"),
            Input("submit", "n_clicks"),
            State({"type": "theta", "index": dash_all()}, "value"),
            prevent_initial_call=True,
        )
        def _submit(n_clicks, theta):
            if not n_clicks:
                return no_update
            accepted = human.submit(theta)
            return (
                f"submitted [{', '.join(f'{c:.4f}' for c in accepted)}]"
                f" - {human.pending_submissions} waiting"
            )

    return app


def dash_all():
    """``ALL`` for pattern-matching callbacks, imported lazily.

    Kept in a function so the import sits next to the one place that needs it
    rather than at module scope, where it reads as a general dependency.
    """
    from dash import ALL

    return ALL


def main() -> None:
    parser = argparse.ArgumentParser(description="xcsearch dashboard")
    parser.add_argument(
        "--store", default=os.path.join("data", "exp1d", "records.sqlite"),
        help="record store to render",
    )
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    app = create_app(args.store)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
