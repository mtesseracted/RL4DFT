"""The Dash app (feature 4, sub-stage 2.7).

    python -m xcsearch.viz.app --store data/exp1d/records.sqlite

Four panels, two across and two down: a run with its history scrubber and
best-so-far strip, the selected candidate's functional split into its exchange
and correlation terms, a regime x metric heatmap against a baseline of your
choosing, and a per-system drill-down. All of it re-reads the store on an
interval, so the panels track a search that is still running -- the store is
in WAL mode and these connections are read-only, so nothing here can block or
corrupt the writer.

The human endpoint is optional: pass an object with ``bounds`` and
``submit(theta)`` (``HumanOptimizer`` is one) and the functional panel grows a
coefficient box per parameter and a submit button.
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


def _row(*panels) -> html.Div:
    """Panels side by side, stacking when the window is too narrow to hold them.

    ``flex-wrap`` rather than a media query: the breakpoint is then a property
    of the panels' own minimum width instead of a number in a stylesheet that
    has to be kept in step with them. ``minWidth: 0`` on each cell because a
    flex item defaults to its content's minimum, and a plotly graph reports a
    wide one -- without it the row refuses to narrow and the page scrolls
    sideways instead of stacking.
    """
    return html.Div(
        [html.Div(panel, style={"flex": "1 1 400px", "minWidth": "0"})
         for panel in panels],
        style={"display": "flex", "gap": "14px", "alignItems": "flex-start",
               "flexWrap": "wrap"},
    )


def _subbox(label: str, *children) -> html.Div:
    """A labelled division inside a card, for a panel with parts."""
    return html.Div(
        [html.Div(label, style={"fontSize": "10px", "letterSpacing": "0.07em",
                                "textTransform": "uppercase", "color": "#8b929a",
                                "marginBottom": "5px"}),
         *children],
        style={"border": "1px solid #eceef1", "borderRadius": "6px",
               "padding": "8px 10px", "marginBottom": "8px", "background": "#fcfcfd"},
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

_FUNCTIONAL_HINT = (
    "The selected trial's functional, split the way the form itself is: an "
    "exchange term carrying every searched coefficient, and a correlation "
    "term carrying none.\n\n"
    "Exchange is LDA exchange scaled by c0, plus the density raised to 1/2, "
    "1, 3/2 and 2 with coefficients c1..c4. Correlation is LDA correlation, "
    "fixed -- the search does not touch it, which is why that box has nothing "
    "to set. LDA itself is theta = [1, 0, 0, 0, 0], which is why c0 near 1 "
    "with small tail terms reads as a correction to it rather than a "
    "replacement for it.\n\n"
    "theta is the coefficient vector as proposed. score is the value stored "
    "for that proposal at the time it was made, not a re-derivation of it here."
)

_HEATMAP_HINT = (
    "How the selected candidate does against a reference functional -- LDA "
    "unless you pick another -- everywhere it was measured.\n\n"
    "The first entry in the baseline list is not a functional. Every metric "
    "here is a distance from exactness, so the exact reference is zero on all "
    "of them, and choosing it shows the candidate's raw errors rather than a "
    "comparison: what it scores, not what it scores relative to something "
    "else. Nothing has a direction on that setting -- every error is above "
    "zero -- so the arrows and the two-sided colours drop away and the whole "
    "ramp goes to size. Raw errors span decades where differences from a "
    "baseline do not, which is why choosing it also brings the log scale "
    "with it; put the scale back to linear if you want it and it will stay "
    "there.\n\n"
    "Rows are metrics. exc_sq_error is the squared error in the XC energy on "
    "the exact density. vxc_l2_error is the density-weighted L2 error in the "
    "XC potential. one_step_dn is how far one Kohn-Sham step moves the "
    "reference density: a functional can get the energy right on the exact "
    "density and still push it somewhere else on the first step.\n\n"
    "Columns are the four geometry regimes: equilibrium, stretched, "
    "dissociating, and isolated atoms.\n\n"
    "The exact conditions are underneath rather than in the grid, and are a "
    "different kind of measurement: the uniform-gas limit, one-electron "
    "self-interaction, and non-positivity of exchange are properties of the "
    "functional itself, measured on no system, so they have no regime to be "
    "a column of and share no metric with the rows above. Kept together they "
    "made a figure that was half empty by construction. They are drawn as "
    "raw violations, not as differences: zero means the constraint holds "
    "exactly, which is a claim relative to nothing, and the colour-scale "
    "toggle does not apply to them.\n\n"
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
    "The boxes under exchange hand-place a functional in the same search the "
    "optimizer is running against, bounded by the same box it proposes "
    "inside; submitting queues one theta, which the search picks up as its "
    "next trial and records like any other, so it lands in the history strip "
    "beside the machine's.\n\n"
    "They are yours to set and do not follow the trial selected on the left: "
    "the formula above is the selected trial's, the numbers in the boxes are "
    "your next proposal. Type into a box, or step it with the buttons either "
    "side -- one click is a two-hundredth of that coefficient's range. A "
    "number outside the box is clipped into it rather than refused, and an "
    "empty box stops the submission rather than guessing at one.\n\n"
    "A session has no trial count. One submission is one trial and it runs "
    "until you stop it -- a budget is what an optimizer needs because "
    "something has to decide when it has had enough, and here you are that "
    "something.\n\n"
    "'start a new run' puts your next submission at the beginning of a fresh "
    "run rather than after the ones already made, for when you want to change "
    "approach and see it separately in the run list. The run appears there "
    "once its first candidate lands, and the panels follow it."
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


# The marker the exchange and correlation halves of a stored formula are
# split on. The form writes `<exchange terms> + \epsilon_c^{...}(n)` and the
# panel shows the two separately, which is a *rendering* convention and not a
# claim about physics -- viz still reads a string out of a record and knows
# nothing about what produced it (D13). If the day comes that forms differ in
# how they separate, the honest fix is for the form to publish its parts
# rather than for this to grow cases; until then one marker and a fallback
# that shows the formula whole are enough.
_CORRELATION_TERM = r"\epsilon_c"


def split_formula(latex: str) -> tuple[str, str]:
    """A stored formula as (exchange, correlation), right-hand sides only.

    The left-hand side is dropped because the two boxes are already labelled;
    printing ``\\epsilon_{xc}(n) =`` above each half would say the wrong thing
    twice. A formula with no correlation term comes back whole in the first
    slot and empty in the second, so an unfamiliar form renders as itself
    rather than as nothing.
    """
    if not latex:
        return "", ""
    _, equals, right = latex.partition("=")
    body = (right if equals else latex).strip()
    cut = body.rfind(_CORRELATION_TERM)
    if cut < 0:
        return body, ""
    return body[:cut].rstrip().rstrip("+").rstrip(), body[cut:].strip()


def step_size(low: float, high: float) -> float:
    """One click of a coefficient's increment buttons.

    A two-hundredth of its range, which is what the sliders these replaced
    stepped by -- and which comes to a round 0.01 for both boxes this form
    uses, so the numbers a person steps through stay ones they would have
    typed.
    """
    return round((high - low) / 200.0, 6)


def stepped(value, direction: int, low: float, high: float) -> float:
    """``value`` moved one step, kept inside the box and off float dust.

    An empty box steps from zero rather than refusing to move: the button
    said it would change the number and doing nothing looks broken. Rounding
    is there because 1.15 + 0.01 is 1.1600000000000001 in binary and a
    coefficient panel that accumulates that is unreadable within a few clicks.
    """
    start = 0.0 if value is None else float(value)
    return round(min(max(start + direction * step_size(low, high), low), high), 6)


_STEP_BUTTON = {
    "width": "24px", "height": "24px", "lineHeight": "1", "padding": "0",
    "border": "1px solid #d4d9e0", "background": "#f6f7f9", "borderRadius": "4px",
    "color": "#4a525c", "cursor": "pointer", "fontSize": "14px",
}


def coefficient_rows(bounds) -> list:
    """A typed box per coefficient, with a step button on either side.

    Text rather than a slider because these are numbers a person has a value
    in mind for -- reproducing a published theta, or nudging one digit of the
    trial on screen -- and a slider is a poor way to say 1.15. The buttons
    keep the one thing a slider was good at, which is moving by a known
    amount without aiming.

    The id shape is unchanged from the sliders, so the submit callback that
    collects them does not know these are different components.
    """
    return [
        html.Div(
            [
                html.Label(f"c{index}", style={"width": "24px", "fontFamily": "monospace",
                                               "fontSize": "12px", "color": "#4a525c"}),
                html.Button("−", id={"type": "theta-down", "index": index},
                            n_clicks=0, style=_STEP_BUTTON),
                dcc.Input(
                    id={"type": "theta", "index": index}, type="number",
                    value=1.0 if index == 0 else 0.0,
                    min=low, max=high, step=step_size(low, high),
                    style={"width": "82px", "margin": "0 4px", "padding": "3px 6px",
                           "border": "1px solid #d4d9e0", "borderRadius": "4px",
                           "fontFamily": "monospace", "fontSize": "12px"},
                ),
                html.Button("+", id={"type": "theta-up", "index": index},
                            n_clicks=0, style=_STEP_BUTTON),
                html.Span(f"[{low:g}, {high:g}]",
                          style={"marginLeft": "8px", "fontSize": "11px",
                                 "color": "#9aa1a9", "fontFamily": "monospace"}),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
        )
        for index, (low, high) in enumerate(bounds)
    ]


def functional_card(human=None, new_run=None) -> html.Div:
    """The functional panel: the formula in parts, and a way to propose one.

    Both modes show the same two boxes. The difference is that in a human
    session the exchange box also carries the coefficients, which is where
    they belong: every searched parameter of this form is an exchange
    parameter, and correlation is fixed LDA with nothing to set. Putting the
    inputs beside the term they are the coefficients *of* is the whole reason
    the panel is split.
    """
    exchange = [dcc.Markdown(id="formula-x", mathjax=True)]
    if human is not None:
        check_human_endpoint(human)
        exchange += [
            html.Div(style={"borderTop": "1px solid #eceef1", "margin": "8px 0 7px"}),
            *coefficient_rows(human.bounds),
        ]

    children = [
        _subbox("exchange", *exchange),
        _subbox(
            "correlation",
            dcc.Markdown(id="formula-c", mathjax=True),
            html.Div("fixed -- the search proposes no correlation coefficients",
                     style={"fontSize": "11px", "color": "#9aa1a9", "marginTop": "2px"}),
        ),
        html.Div(id="theta"),
    ]

    if human is not None:
        children += [
            html.Button("submit", id="submit", n_clicks=0,
                        style={"marginTop": "8px"}),
            # What was last submitted, so the line under the button can be
            # recomputed on the tick instead of frozen at the click.
            dcc.Store(id="last-submission"),
            html.Div(id="submitted", style={"fontSize": "13px", "marginTop": "6px"}),
            # A session runs one trial per submission for as long as you keep
            # submitting, so "start a new run" is the only run-shaped control
            # it needs: there is no length to declare up front.
            *([
                html.Hr(style={"border": "none", "borderTop": "1px solid #eceef1",
                               "margin": "12px 0 10px"}),
                html.Button("start a new run", id="new-run", n_clicks=0),
                html.Div(id="new-run-status",
                         style={"fontSize": "13px", "marginTop": "6px",
                                "color": "#4a525c"}),
            ] if new_run is not None else []),
        ]

    hint = _FUNCTIONAL_HINT if human is None else f"{_FUNCTIONAL_HINT}\n\n{_PROPOSAL_HINT}"
    return _card("functional", *children, hint=hint)


_RELOAD_ADVICE = (
    "the page was loaded from an older version of this app, so it is calling "
    "callbacks with the wrong arguments -- reload it (ctrl-shift-r)"
)


def arguments_expected(app, output: str) -> int | None:
    """How many arguments the registered callback for ``output`` takes.

    Top-level arguments, which is what dash indexes the request against: a
    pattern-matching group counts once however many components it matches.
    ``None`` if no callback is registered for that output at all, which is
    the same illness at a later stage -- a page asking for something this
    server has never heard of.
    """
    callback = app.callback_map.get(output)
    if callback is None:
        return None
    return len(callback.get("inputs", ())) + len(callback.get("state", ()))


def stale_page(app, body: dict) -> str | None:
    """Why this request cannot be served, if it cannot, in one line.

    A dash page fetches the callback signatures once, at load, and then sends
    arguments in that shape forever. Restart the server with a callback that
    has gained an argument and every open page is now wrong -- it sends three
    where four are read, and dash indexes off the end of the list, which
    surfaces as an ``IndexError`` deep in ``_prepare_grouping`` with nothing
    in it naming the real problem. Worse, ``dcc.Interval`` means the page
    keeps asking, so one stale tab prints that traceback every three seconds.

    Checking is cheap and the diagnosis is exact, so the page is told what is
    wrong with it instead of the log being filled with a framework's stack.
    """
    output = body.get("output")
    if not isinstance(output, str):
        return None
    expected = arguments_expected(app, output)
    name = output.split("...")[0].strip(".")
    if expected is None:
        return f"no callback is registered for {name}: {_RELOAD_ADVICE}"
    sent = len(body.get("inputs") or ()) + len(body.get("state") or ())
    if sent < expected:
        return (f"the callback for {name} reads {expected} arguments and the "
                f"page sent {sent}: {_RELOAD_ADVICE}")
    return None


def refuse_stale_pages(app) -> None:
    """Answer a stale page with its diagnosis, and stop caching the contract.

    Two halves of one problem. The guard turns the crash into a 409 naming
    the callback and the argument counts, which is a thing a reader can act
    on. The headers close the door the other way: dash serves the layout and
    the dependency list with no cache directive at all, so nothing stops a
    browser reusing yesterday's contract against today's server, and a page
    that is stale that way looks freshly loaded to the person loading it.
    """
    from flask import request

    server = app.server

    @server.before_request
    def _reject_mismatched_arguments():
        if request.method != "POST":
            return None
        if not request.path.endswith("/_dash-update-component"):
            return None
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return None
        complaint = stale_page(app, body)
        if complaint is None:
            return None
        return server.response_class(complaint, status=409, mimetype="text/plain")

    @server.after_request
    def _do_not_cache_the_contract(response):
        contract = ("/_dash-layout", "/_dash-dependencies")
        if request.path.endswith(contract) or request.path in ("/", ""):
            response.headers["Cache-Control"] = "no-store"
        return response


def serve(app, host: str = "127.0.0.1", port: int = 8050) -> None:
    """Run the app, with an already-open page told to reload on a restart.

    ``dev_tools_hot_reload`` gives dash a per-process reload hash that the
    page polls; a page that comes back with a different hash than the one it
    loaded under reloads itself. That is exactly the stale-page case above,
    healed rather than diagnosed -- restart the session and the tab you left
    open picks up the new callbacks on its own. It is only the hot reloader:
    ``debug`` stays off, so no error overlays and no code-reloading.
    """
    app.run(host=host, port=port, debug=False, dev_tools_hot_reload=True)


def run_options(store_path: str) -> list[dict]:
    return [
        {"label": f"run {r['id']} - {r['optimizer']} ({r['candidates']} candidates)",
         "value": r["id"]}
        for r in data.runs(store_path)
    ]


def baseline_options(store_path: str) -> list[dict]:
    return [
        {"label": c["label"], "value": c["id"]} for c in data.baseline_choices(store_path)
    ]


def run_choice(options: list[dict], current, offered: list[dict], follow_run):
    """Which run the dropdown should be showing after a refresh.

    ``follow_run`` is this session's own run, which a human session passes so
    the panels land on the run the person is creating rather than on whatever
    was newest when the server started. It cannot simply be selected whenever
    it exists: the run row appears only when its first candidate does (B1), so
    "exists" arrives mid-session, and re-selecting it on every poll would drag
    a reader back every three seconds each time they looked at another run.

    So it is selected on the transition, detected against the options the
    client is *already holding*: a run that was not in the last set offered
    and is in this one has just appeared, and that is the only moment this
    reaches for it.
    """
    ids = [option["value"] for option in options]
    if follow_run is not None and follow_run in ids:
        if follow_run not in [option["value"] for option in offered or []]:
            return follow_run
    if current is None or current not in ids:
        return follow_run if follow_run in ids else (ids[0] if ids else None)
    return current


def system_options(store_path: str) -> list[dict]:
    # One option per system *name*, not per version. A content-changed rebuild
    # adds a version (D17), and two options reading "atom_Z1" with the same
    # value would be a menu the reader cannot choose between -- the drill-down
    # resolves a name through the selected candidate's own rows anyway, so the
    # version is not the dropdown's to pick. `dict.fromkeys` keeps first sight,
    # and `data.systems` orders newest first, so the name resolves the way
    # `density` resolves it.
    return [
        {"label": name, "value": name}
        for name in dict.fromkeys(s["name"] for s in data.systems(store_path))
    ]


def build_layout(store_path: str, human=None, new_run=None) -> html.Div:
    """The initial page. Every dropdown here is refreshed by ``_choices``.

    What this builds is a *starting* set of options, not the set: a store
    gains runs, candidates and system versions while the page is open -- a
    human session's own run does not exist until its first submission lands
    (B1) -- and ``app.layout`` is one object served to every visitor, so
    nothing built here is refreshed by reloading the browser either.
    """
    runs = run_options(store_path)
    baselines = baseline_options(store_path)
    systems = system_options(store_path)

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
        # Two rows of two. Reading order is the order the question is asked
        # in: which trial (and how the run got there), what functional that
        # was, where it wins and loses, and then one system in detail. Side
        # by side rather than stacked because the first two are read against
        # each other -- a coefficient means something next to the score it
        # earned -- and so are the last two.
        _row(
            _card(
                "run",
                dcc.Dropdown(
                    id="run", options=runs,
                    value=runs[0]["value"] if runs else None,
                    clearable=False,
                ),
                dcc.Graph(id="history", config={"displayModeBar": False}),
                html.Div(
                    # "trial", because that is what the slider's value is. It
                    # was labelled "candidate" while carrying the trial index,
                    # so the UI was actively wrong rather than ambiguous (B9).
                    [html.Span("trial", style={"marginRight": "8px"}),
                     dcc.Slider(id="scrubber", min=0, max=0, step=1, value=0,
                                tooltip={"placement": "bottom"})],
                    style={"marginTop": "6px"},
                ),
                html.Div(id="selection", style={"fontSize": "12px", "color": "#4a525c",
                                                "marginTop": "2px"}),
                hint=_RUN_HINT,
            ),
            functional_card(human, new_run),
        ),
        _row(
            _card(
                "regime x metric, against a baseline",
                dcc.Dropdown(
                    id="baseline", options=baselines,
                    value=data.default_baseline(store_path), clearable=False,
                ),
                # What set the two means were taken over (D19). A comparison
                # across populations is over their intersection, and saying so
                # is part of the comparison rather than a footnote to it.
                html.Div(id="heatmap-scope",
                         style={"fontSize": "12px", "color": "#7a828c",
                                "margin": "2px 0"}),
                # Colour scale, not data: both draw the same differences.
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
                # Their own figure, because they are their own kind of
                # measurement: no system, so no regime to be a column of, and
                # no metric in common with the grid above.
                _subbox(
                    "exact conditions",
                    dcc.Graph(id="conditions", config={"displayModeBar": False}),
                    html.Div(id="conditions-note",
                             style={"fontSize": "11px", "color": "#9aa1a9"}),
                ),
                hint=_HEATMAP_HINT,
            ),
            _card(
                "system detail",
                dcc.Dropdown(
                    id="system", options=systems,
                    value=systems[0]["value"] if systems else None,
                    clearable=False,
                ),
                html.Div(id="system-metrics",
                         style={"fontSize": "13px", "margin": "6px 0"}),
                dcc.Graph(id="density", config={"displayModeBar": False}),
                dcc.Graph(id="potential", config={"displayModeBar": False}),
                hint=_SYSTEM_HINT,
            ),
        ),
    ]

    return html.Div(
        panels,
        style={
            # Sized for the screen this is read on: 1920 wide, less a
            # scrollbar, this padding, and room left over for a side menu if
            # one ever arrives. Wide matters more than it did now that the
            # panels sit two across -- at the old 900 each pane came out
            # narrower than the heatmap needs.
            "maxWidth": "1600px", "margin": "0 auto", "padding": "18px",
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


def submission_line(human, last: dict) -> str:
    """What became of the last thing submitted, as of now.

    Three states, and the reason to distinguish them is that a person who has
    just clicked wants to know which one they are in: still queued, being
    evaluated, or done and worth this much. The score is the interesting one
    -- it is the number the session was told (D20), read back from the
    endpoint's own history rather than recomputed here.

    ``history`` is not part of the human contract, which is deliberately the
    three things the panel *needs* (``bounds``, ``submit``,
    ``pending_submissions``). An endpoint that keeps one gets a better line;
    one that does not still gets a true one.
    """
    theta = ", ".join(f"{c:.4f}" for c in last["theta"])
    history = getattr(human, "history", None)
    if history is not None and len(history) > last["index"]:
        _, score = history[last["index"]]
        return f"submitted [{theta}] - evaluated, score {score:.4e}"
    pending = human.pending_submissions
    if pending:
        return f"submitted [{theta}] - {pending} waiting"
    return f"submitted [{theta}] - evaluating"


def conditions_note(rows: list[dict], against_zero: bool) -> str:
    """The line under the condition bars: what they are, and how many hold.

    Worth counting rather than leaving to be read off the bars, because
    "satisfied" here means exactly zero and a bar too short to see is not the
    same claim as one that is not there.
    """
    if not rows:
        return ""
    satisfied = [row["metric"] for row in rows if row.get("value") == 0]
    line = (f"{len(satisfied)} of {len(rows)} satisfied exactly"
            if satisfied else f"none of {len(rows)} satisfied exactly")
    if against_zero:
        return f"{line}. The exact reference satisfies all of them, so it is not drawn."
    return f"{line}. Lower is better; these are violations, not errors against data."


def scale_for_baseline(baseline_id, current: str) -> str | None:
    """The colour scale to move to on a change of baseline, or None to stay.

    Raw errors span decades where differences from a baseline do not. On the
    3000-trial run they run from 2.9e-04 to 7.8e-01, so linear colour puts the
    whole ``exc_sq_error`` row at the foot of the ramp and the panel looks
    broken rather than looking like four decades. Moving the toggle along with
    the baseline is a visible move of a control the reader can move back,
    which beats handing them a blank map and letting them go looking.

    That is an observation about what these metrics are -- a squared energy
    error and a density norm are not the same size -- rather than a law, which
    is why the rule is written here as one condition and not derived from the
    data. Only on a change of baseline, and only away from the default: a
    reader who has already chosen a scale keeps it.
    """
    if baseline_id == data.EXACT and current == "linear":
        return "logabs"
    return None


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


def create_app(store_path: str, human=None, status=None, follow_run=None,
               new_run=None) -> Dash:
    """Build the app. Every callback re-reads the store, so panels live-tail.

    ``status`` is an optional zero-argument callable returning a line to show
    above the panels -- ``serve_human_session`` passes one that reports a dead
    search worker (B10). A callable rather than a queue or an object: the
    dashboard polls it, and everything it knows about the search stays "a
    string someone else computed".

    ``new_run`` is the same idea in the other direction: a zero-argument
    callable that asks the session to start a new run and returns a line
    saying what was asked. Its button appears only when one is given. The
    dashboard does not open the run itself and could not -- its connections
    are read-only on purpose -- so this signals the thread that owns the
    writes, exactly as ``submit`` signals the thread that owns the loop.

    ``follow_run`` says which run the panels should land on; it may be an id
    or a callable returning one, since a session's current run changes when
    that button is used.
    """
    app = Dash(__name__)
    app.title = "xcsearch"
    app.layout = build_layout(store_path, human, new_run)

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
        Output("formula-x", "children"),
        Output("formula-c", "children"),
        Output("theta", "children"),
        Output("heatmap", "figure"),
        Output("conditions", "figure"),
        Output("conditions-note", "children"),
        Output("heatmap-scope", "children"),
        Input("selected", "data"),
        Input("baseline", "value"),
        Input("heatmap-scale", "value"),
        Input("tick", "n_intervals"),
    )
    def _candidate(selected, baseline_id, scale, _tick):
        if not selected or baseline_id is None:
            return ("*no candidate selected*", "", "", figures.heatmap([]),
                    figures.condition_bars([]), "", "")
        candidate_id = selected["candidate_id"]

        record = data.candidate(store_path, candidate_id) or {}
        exact = baseline_id == data.EXACT
        baseline = {} if exact else (data.candidate(store_path, baseline_id) or {})
        exchange, correlation = split_formula(record.get("form_latex", ""))
        theta = ", ".join(f"{c:.4f}" for c in record.get("theta", []))
        candidate_label = f"candidate {candidate_id}"
        baseline_label = ("exact reference" if exact else
                          baseline.get("label") or f"candidate {baseline_id}")
        # One query, two figures: the grid was half empty because these two
        # halves share no metric and the conditions have no regime.
        per_system, conditions = data.split_conditions(
            data.heatmap(store_path, candidate_id, baseline_id))
        return (
            f"$${exchange}$$" if exchange else "",
            f"$${correlation}$$" if correlation else "",
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
                per_system,
                candidate_label=candidate_label, baseline_label=baseline_label,
                scale=scale or "linear", against_zero=exact,
            ),
            figures.condition_bars(
                conditions,
                candidate_label=candidate_label, baseline_label=baseline_label,
                against_zero=exact,
            ),
            conditions_note(conditions, exact),
            scope_label(data.heatmap_scope(store_path, candidate_id, baseline_id)),
        )

    @app.callback(
        Output("heatmap-scale", "value"),
        Input("baseline", "value"),
        State("heatmap-scale", "value"),
        prevent_initial_call=True,
    )
    def _scale_for_baseline(baseline_id, current):
        return scale_for_baseline(baseline_id, current) or no_update

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
        Output("run", "options"),
        Output("run", "value"),
        Output("baseline", "options"),
        Output("system", "options"),
        Input("tick", "n_intervals"),
        State("run", "value"),
        State("run", "options"),
    )
    def _choices(_tick, current_run, offered):
        """Keep the dropdowns level with a store that is still being written.

        The panels have always re-read the store on the interval; the menus
        naming what to read did not, and were fixed at whatever existed when
        the server started. That is invisible for a finished run and wrong for
        every live one -- a human session's own run is created by its first
        submission (B1), so it could never be selected at all, and candidates
        submitted during a session never became available as baselines.

        Reloading the browser was no escape either: ``app.layout`` is one
        object built once and served to every visitor.
        """
        options = run_options(store_path)
        following = follow_run() if callable(follow_run) else follow_run
        return (
            options,
            run_choice(options, current_run, offered, following),
            baseline_options(store_path),
            system_options(store_path),
        )

    @app.callback(
        Output("worker-status", "children"),
        Input("tick", "n_intervals"),
    )
    def _worker_status(_tick):
        """The watchdog line (B10). Silent unless there is something to say."""
        return status() if status is not None else ""

    if human is not None:
        match, context = dash_match(), dash_context()

        @app.callback(
            Output({"type": "theta", "index": match}, "value"),
            Input({"type": "theta-down", "index": match}, "n_clicks"),
            Input({"type": "theta-up", "index": match}, "n_clicks"),
            State({"type": "theta", "index": match}, "value"),
            prevent_initial_call=True,
        )
        def _step(_down, _up, value):
            """Move one coefficient by one step, in the direction clicked.

            The one place here that asks dash which input fired, and the one
            place it is the honest answer: the two buttons differ in nothing
            but direction, and unlike the submit line there is no previously
            published state to read the difference out of. The bounds come
            from the endpoint rather than from the component's own min/max,
            so the box a value is clipped into is the form's and not whatever
            the browser sent back.
            """
            fired = context.triggered_id
            if fired is None:
                return no_update
            low, high = human.bounds[fired["index"]]
            return stepped(value, -1 if fired["type"] == "theta-down" else 1,
                           low, high)

        @app.callback(
            Output("submitted", "children"),
            Output("last-submission", "data"),
            Input("submit", "n_clicks"),
            Input("tick", "n_intervals"),
            State({"type": "theta", "index": dash_all()}, "value"),
            State("last-submission", "data"),
            prevent_initial_call=True,
        )
        def _submit(n_clicks, _tick, theta, last):
            """Queue a proposal on a click, and keep its line true after.

            The line used to be written once, at the click, and never again --
            so it reported the queue depth at the only instant it is certainly
            one, and went on saying "1 waiting" long after the trial had been
            evaluated and stored. It is a claim about a live thing, so it has
            to be recomputed like every other live thing here: the tick is an
            input, and which of the two inputs fired is read the way this file
            reads it everywhere else, by comparing against what was last
            published rather than by asking dash.
            """
            submitting = not last or n_clicks != last.get("clicks")
            if submitting:
                if not n_clicks:
                    return no_update, no_update
                empty = [f"c{i}" for i, c in enumerate(theta) if c is None]
                if empty:
                    # The click is still recorded, with no theta on it. Not
                    # doing so would leave this looking like an unsubmitted
                    # click on every tick after, and it would resubmit.
                    return (f"nothing submitted: {', '.join(empty)} empty",
                            {"clicks": n_clicks})
                accepted = human.submit(theta)
                last = {"clicks": n_clicks,
                        "theta": [float(c) for c in accepted],
                        # Where this proposal's result will land, so its
                        # outcome can be recognised rather than guessed at.
                        "index": len(getattr(human, "history", ()))}
            elif not last.get("theta"):
                return no_update, no_update
            return submission_line(human, last), last

        if new_run is not None:
            @app.callback(
                Output("new-run-status", "children"),
                Input("new-run", "n_clicks"),
                prevent_initial_call=True,
            )
            def _new_run(n_clicks):
                """Ask the session for a new run and report what it said.

                The reply is the session's, not this callback's: whether a
                request is worth honouring depends on how many candidates the
                current run has, which is the worker's business to know.
                """
                return new_run() if n_clicks else no_update

    # Registered last, so it sees the whole callback map when it checks one.
    refuse_stale_pages(app)
    return app


def dash_all():
    """``ALL`` for pattern-matching callbacks, imported lazily.

    Kept in a function so the import sits next to the one place that needs it
    rather than at module scope, where it reads as a general dependency.
    """
    from dash import ALL

    return ALL


def dash_match():
    """``MATCH``, for the per-coefficient step callback. See :func:`dash_all`."""
    from dash import MATCH

    return MATCH


def dash_context():
    """``ctx``, the only thing here that asks dash which input fired."""
    from dash import ctx

    return ctx


def main() -> None:
    parser = argparse.ArgumentParser(description="xcsearch dashboard")
    parser.add_argument(
        "--store", default=os.path.join("data", "exp1d", "records.sqlite"),
        help="record store to render",
    )
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    serve(create_app(args.store), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
