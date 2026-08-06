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


def _card(title: str, *children) -> html.Div:
    return html.Div(
        [html.H3(title, style={"margin": "0 0 8px", "fontSize": "15px"}), *children],
        style={
            "border": "1px solid #e2e5ea", "borderRadius": "8px",
            "padding": "12px 14px", "marginBottom": "14px", "background": "white",
        },
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
        ),
        _card("formula", dcc.Markdown(id="formula", mathjax=True), html.Div(id="theta")),
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
            dcc.Graph(id="heatmap", config={"displayModeBar": False}),
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
            )
        )

    return html.Div(
        panels,
        style={
            "maxWidth": "900px", "margin": "0 auto", "padding": "18px",
            "fontFamily": "system-ui, sans-serif", "background": "#f6f7f9",
        },
    )


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
    )
    def _history(run_id, _tick, current):
        """The one place a trial index becomes a candidate id (B9).

        Reads the history once per poll and publishes both the figure and the
        resolved id, so the dependent panels neither re-read it nor re-resolve
        it -- and cannot resolve it against a different run than this one.
        """
        if run_id is None:
            return 0, 0, figures.history_strip([]), None, ""
        rows = data.history(store_path, run_id)
        if not rows:
            return 0, 0, figures.history_strip([]), None, ""
        last = len(rows) - 1
        # A scrubber sitting at the end follows the run; one a person has
        # dragged back stays where they put it.
        value = last if current is None or current >= last else current
        row = rows[value]
        selected = row["candidate_id"]
        return (
            last,
            value,
            figures.history_strip(rows, selected),
            {"run_id": run_id, "trial": value, "candidate_id": selected,
             "score": row["score"]},
            # Both identities, together: they are different numbers and the
            # panels below name the candidate one.
            f"trial {value} / candidate {selected}",
        )

    @app.callback(
        Output("formula", "children"),
        Output("theta", "children"),
        Output("heatmap", "figure"),
        Output("heatmap-scope", "children"),
        Input("selected", "data"),
        Input("baseline", "value"),
        Input("tick", "n_intervals"),
    )
    def _candidate(selected, baseline_id, _tick):
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
