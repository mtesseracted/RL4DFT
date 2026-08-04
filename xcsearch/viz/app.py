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


def build_layout(store_path: str, human=None) -> html.Div:
    run_options = [
        {"label": f"run {r['id']} - {r['optimizer']} ({r['candidates']} candidates)",
         "value": r["id"]}
        for r in data.runs(store_path)
    ]
    baseline_options = [
        {"label": c["label"], "value": c["id"]} for c in data.baseline_choices(store_path)
    ]
    system_options = [
        {"label": s["name"], "value": s["name"]} for s in data.systems(store_path)
    ]

    panels = [
        html.H1("xcsearch", style={"fontSize": "20px", "marginBottom": "2px"}),
        html.Div(os.path.abspath(store_path), style={"color": "#7a828c", "fontSize": "12px"}),
        dcc.Interval(id="tick", interval=_POLL_MS),
        _card(
            "run",
            dcc.Dropdown(
                id="run", options=run_options,
                value=run_options[0]["value"] if run_options else None,
                clearable=False,
            ),
            dcc.Graph(id="history", config={"displayModeBar": False}),
            html.Div(
                [html.Span("candidate", style={"marginRight": "8px"}),
                 dcc.Slider(id="scrubber", min=0, max=0, step=1, value=0,
                            tooltip={"placement": "bottom"})],
                style={"marginTop": "6px"},
            ),
        ),
        _card("formula", dcc.Markdown(id="formula", mathjax=True), html.Div(id="theta")),
        _card(
            "regime x metric, against a baseline",
            dcc.Dropdown(
                id="baseline", options=baseline_options,
                value=data.default_baseline(store_path), clearable=False,
            ),
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


def create_app(store_path: str, human=None) -> Dash:
    """Build the app. Every callback re-reads the store, so panels live-tail."""
    app = Dash(__name__)
    app.title = "xcsearch"
    app.layout = build_layout(store_path, human)

    @app.callback(
        Output("scrubber", "max"),
        Output("scrubber", "value"),
        Output("history", "figure"),
        Input("run", "value"),
        Input("tick", "n_intervals"),
        State("scrubber", "value"),
    )
    def _history(run_id, _tick, current):
        if run_id is None:
            return 0, 0, figures.history_strip([])
        rows = data.history(store_path, run_id)
        if not rows:
            return 0, 0, figures.history_strip([])
        last = len(rows) - 1
        # A scrubber sitting at the end follows the run; one a person has
        # dragged back stays where they put it.
        value = last if current is None or current >= last else current
        selected = rows[value]["candidate_id"]
        return last, value, figures.history_strip(rows, selected)

    @app.callback(
        Output("formula", "children"),
        Output("theta", "children"),
        Output("heatmap", "figure"),
        Input("run", "value"),
        Input("scrubber", "value"),
        Input("baseline", "value"),
        Input("tick", "n_intervals"),
    )
    def _candidate(run_id, trial, baseline_id, _tick):
        rows = data.history(store_path, run_id) if run_id is not None else []
        if not rows or baseline_id is None:
            return "*no candidate selected*", "", figures.heatmap([])
        trial = min(trial or 0, len(rows) - 1)
        candidate_id = rows[trial]["candidate_id"]

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
                    html.Span(f"  score = {rows[trial]['score']:.4e}",
                              style={"color": "#4a525c"}),
                ],
                style={"fontSize": "13px", "fontFamily": "monospace"},
            ),
            figures.heatmap(
                data.heatmap(store_path, candidate_id, baseline_id),
                candidate_label=f"candidate {candidate_id}",
                baseline_label=baseline.get("label") or f"candidate {baseline_id}",
            ),
        )

    @app.callback(
        Output("system-metrics", "children"),
        Output("density", "figure"),
        Output("potential", "figure"),
        Input("system", "value"),
        Input("run", "value"),
        Input("scrubber", "value"),
        Input("tick", "n_intervals"),
    )
    def _system(system, run_id, trial, _tick):
        if system is None:
            return "", figures.density_figure(None, [], ""), figures.potential_figure(None, "")

        wells = next(
            (s["wells"] for s in data.systems(store_path) if s["name"] == system), []
        )
        arrays = data.density(store_path, system)

        metrics_text = ""
        rows = data.history(store_path, run_id) if run_id is not None else []
        if rows:
            trial = min(trial or 0, len(rows) - 1)
            metrics = data.system_metrics(store_path, rows[trial]["candidate_id"], system)
            metrics_text = "   ".join(f"{k} = {v:.4e}" for k, v in sorted(metrics.items()))

        return (
            html.Span(metrics_text, style={"fontFamily": "monospace"}),
            figures.density_figure(arrays, wells, system),
            figures.potential_figure(arrays, system),
        )

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
