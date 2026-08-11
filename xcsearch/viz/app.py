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
    "One form, top to bottom: build it, read it, set its numbers, submit.\n\n"
    "The formula shown is the *working form* -- what the structural editor "
    "holds if you have added a term, and otherwise the selected trial's. "
    "Adding a term changes both the formula and the coefficient boxes below "
    "it, because a coefficient is a coefficient of a particular form; "
    "'clear' empties the editor and hands the panel back to the selection on "
    "the left.\n\n"
    "Each box is one named part of the form document, with the "
    "coefficients that belong to it and the box each one was searched "
    "inside.\n\n"
    "For the seeded form that is exchange -- LDA exchange scaled by one "
    "coefficient, plus the density raised to a menu of rational powers -- and "
    "a correlation part with no free coefficient at all. Which parts exist, "
    "and which coefficients sit under each, is read from the document rather "
    "than assumed here, so a form with three parts renders as one. LDA itself "
    "is the member with the LDA scale at 1 and every power coefficient at 0, "
    "which is why a scale near 1 with small tail terms reads as a correction "
    "to it rather than a replacement for it.\n\n"
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
    "The coefficient boxes hand-place a functional in the same search the "
    "optimizer is running against, bounded by the same box it proposes "
    "inside; submitting queues one theta, which the search picks up as its "
    "next trial and records like any other, so it lands in the history strip "
    "beside the machine's.\n\n"
    "There is one box per slot of the *working form*, so they follow the "
    "structural editor. Submitting a theta for a form the run is not already "
    "on adopts that form: a new run opens under a protocol row naming it, and "
    "your submissions from then on are proposals for it. That is the "
    "difference between the two buttons -- 'submit' keeps you in the seat "
    "with a form of your own, 'optimize' lends the seat to a machine "
    "optimizer for a set number of trials and gives it back.\n\n"
    "Type into a box, or step it with the buttons either side -- one click is "
    "a two-hundredth of that coefficient's range. A number outside the box is "
    "clipped into it rather than refused, and an empty box stops the "
    "submission rather than guessing at one.\n\n"
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


# F1.11's string surgery lived here: a stored one-line formula was cut at the
# last `\epsilon_c` to guess where exchange ended and correlation began. It is
# *deleted* rather than bypassed, because the guess is no longer needed --
# 3.2's forms room stores the latex the representation rendered **per part**,
# so the panel reads the parts a form publishes instead of reverse-engineering
# them from a rendered string. The honest fix the old comment asked for, taken.


def formula_parts(record: dict, form: dict | None) -> list:
    """The functional panel's body: one labelled box per part of the form.

    ``form`` is what ``data.form_for`` returned -- per-part latex and the slot
    boxes, both straight out of the stored document. When it is ``None`` (a
    candidate from an adapter with no document) the whole-form latex the
    candidate row has always carried is shown under one unnamed box, so an
    older store renders as itself rather than as nothing.

    The coefficient list is why a **store-only viewer** can now show bounds:
    they are in the document, the document is in the store, and no candidate
    has to be rebuilt to read them. That was the obstacle R2 recorded, gone as
    a side effect of a machine-path need.
    """
    theta = list(record.get("theta") or [])
    if form is None:
        whole = (record.get("form_latex") or "").strip()
        return [_subbox("form", dcc.Markdown(f"$${whole}$$" if whole else "",
                                             mathjax=True))]

    slots = form.get("slots") or []
    boxes = []
    for index, part in enumerate(sorted(form.get("latex", {}))):
        if part == "__all__":
            continue
        rendered = form["latex"][part]
        rows = [
            _coefficient_line(position, entry, theta)
            for position, entry in enumerate(slots)
            if entry["part"] == part
        ]
        boxes.append(_subbox(
            part,
            dcc.Markdown(f"$${rendered}$$" if rendered else "", mathjax=True),
            *([html.Div(rows, style={"marginTop": "4px"})] if rows else
              [html.Div("fixed -- this part proposes no coefficients",
                        style={"fontSize": "11px", "color": "#9aa1a9",
                               "marginTop": "2px"})]),
        ))
        del index
    return boxes


def working_form_boxes(parts_latex: dict, note: str) -> list:
    """One labelled box per part of the form currently being edited.

    The same shape as :func:`formula_parts`, minus the read-only coefficient
    lines: those numbers are the editable boxes immediately below, and
    printing them twice invites the two to disagree on screen.
    """
    boxes = []
    for part in sorted(parts_latex):
        if part == "__all__":
            continue
        rendered = parts_latex[part]
        boxes.append(_subbox(
            part,
            dcc.Markdown(f"$${rendered}$$" if rendered else "", mathjax=True),
        ))
    if not boxes:
        return [_subbox("form", dcc.Markdown("*no terms yet*"))]
    boxes.append(html.Div(note, style={"fontSize": "11px", "color": "#9aa1a9",
                                       "marginTop": "2px"}))
    return boxes


def identity_line(editing: bool, candidate_id=None, trial=None, score=None,
                  theta=None, form_hash=None, n_slots=0) -> html.Div:
    """Whose form and whose numbers the panel is currently showing.

    This line used to be written by the candidate callback, so it always
    named the selected candidate -- including while the latex directly above
    it was the *editor's* form and the boxes below it were the editor's
    slots. Two of the three panes had moved on and this one had not, which
    made a correct candidate id read as a wrong one.

    It also leads with the **trial**, because that is the number the scrubber
    is on and the one a person is tracking; the candidate id is a store key
    that runs to five digits and means nothing on its own.
    """
    if editing:
        return html.Div(
            [
                html.Span("editing", style={"fontWeight": 600, "color": "#8a5a00"}),
                html.Span(f"  {n_slots} coefficient{'' if n_slots == 1 else 's'}"
                          "  -- not yet proposed",
                          style={"color": "#4a525c"}),
                html.Span(f"  form {(form_hash or '')[:12]}",
                          style={"color": "#9aa1a9"}),
            ],
            style={"fontSize": "13px", "fontFamily": "monospace"},
        )
    numbers = ", ".join(f"{c:.4f}" for c in (theta or []))
    shown = [html.Span(f"trial {trial}" if trial is not None
                       else f"candidate {candidate_id}",
                       style={"fontWeight": 600})]
    if trial is not None:
        shown.append(html.Span(f" (candidate {candidate_id})",
                               style={"color": "#9aa1a9"}))
    shown.append(html.Span(f"  theta = [{numbers}]", style={"color": "#4a525c"}))
    if score is not None:
        # The stored score: the number the optimizer was told (D20), not a
        # re-derivation of it.
        shown.append(html.Span(f"  score = {score:.4e}", style={"color": "#4a525c"}))
    return html.Div(shown, style={"fontSize": "13px", "fontFamily": "monospace"})


def _session_slots(session) -> list:
    """The session's opening form, in the shape a working form is in."""
    return [{"part": entry.get("part"), "bounds": list(entry["bounds"])}
            for entry in session.coefficients]


def _coefficient_line(position: int, entry: dict, theta: list) -> html.Div:
    """``c_k = value  [low, high]`` -- read-only, from the stored document."""
    low, high = entry["bounds"]
    value = theta[position] if position < len(theta) else None
    shown = "--" if value is None else f"{float(value):+.4f}"
    return html.Div(
        [
            html.Span(f"c{position}", style={"width": "26px", "display": "inline-block"}),
            html.Span(shown, style={"width": "78px", "display": "inline-block"}),
            html.Span(f"[{_g(low)}, {_g(high)}]", style={"color": "#9aa1a9"}),
        ],
        style={"fontFamily": "monospace", "fontSize": "11.5px", "color": "#4a525c"},
    )


def _g(value) -> str:
    return "?" if value is None else f"{float(value):g}"


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


def coefficient_rows(bounds, parts=None, disabled: bool = False) -> list:
    """A typed box per coefficient, with a step button on either side.

    ``parts`` names the part each coefficient belongs to, from the form
    document (``DocumentForm.slot_parts``), and the boxes are grouped under
    those names. Grouping is the *form's* statement about itself now, not the
    panel's assumption that every searched parameter is an exchange parameter
    -- which was true of exactly one form.

    Text rather than a slider because these are numbers a person has a value
    in mind for -- reproducing a published theta, or nudging one digit of the
    trial on screen -- and a slider is a poor way to say 1.15. The buttons
    keep the one thing a slider was good at, which is moving by a known
    amount without aiming.

    The id shape is unchanged from the sliders, so the submit callback that
    collects them does not know these are different components.
    """
    bounds = list(bounds)
    parts = list(parts) if parts is not None else [None] * len(bounds)
    rows: list = []
    previous = object()
    for index, ((low, high), part) in enumerate(zip(bounds, parts)):
        if part != previous:
            previous = part
            if part is not None:
                rows.append(html.Div(
                    part,
                    style={"fontSize": "11px", "color": "#7a828c",
                           "margin": "6px 0 2px", "textTransform": "lowercase"},
                ))
        rows.append(_coefficient_input(index, low, high, disabled))
    return rows


def _coefficient_input(index: int, low: float, high: float,
                       disabled: bool = False) -> html.Div:
    return (
        html.Div(
            [
                html.Label(f"c{index}", style={"width": "24px", "fontFamily": "monospace",
                                               "fontSize": "12px", "color": "#4a525c"}),
                html.Button("−", id={"type": "theta-down", "index": index},
                            n_clicks=0, disabled=disabled, style=_STEP_BUTTON),
                dcc.Input(
                    id={"type": "theta", "index": index}, type="number",
                    value=1.0 if index == 0 else 0.0, disabled=disabled,
                    min=low, max=high, step=step_size(low, high),
                    style={"width": "82px", "margin": "0 4px", "padding": "3px 6px",
                           "border": "1px solid #d4d9e0", "borderRadius": "4px",
                           "fontFamily": "monospace", "fontSize": "12px"},
                ),
                html.Button("+", id={"type": "theta-up", "index": index},
                            n_clicks=0, disabled=disabled, style=_STEP_BUTTON),
                html.Span(f"[{low:g}, {high:g}]",
                          style={"marginLeft": "8px", "fontSize": "11px",
                                 "color": "#9aa1a9", "fontFamily": "monospace"}),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "4px"},
        )
    )


# Live numbers are marked, every time they render (D34). Not decoration: the
# amended D13 says a number on screen is either in a record, reachable from
# one, or *explicitly marked as a live diagnostic that is not*, and this is
# what "explicitly" is made of.
_LIVE_STYLE = {"fontFamily": "monospace", "fontSize": "12px", "color": "#8a5a00",
               "background": "#fff8e6", "padding": "2px 6px", "borderRadius": "4px"}

# A failed live solve. Still marked live -- it is the same kind of number,
# and "the diagnostic did not produce one" is a diagnostic.
_FAILED_STYLE = dict(_LIVE_STYLE, color="#b3261e", background="#fdecea")


def _with_term(document, term) -> dict:
    """Append a term to a document's part, as plain JSON.

    The editor's whole document manipulation, and deliberately the whole of
    it: everything that needs to know what a form *means* -- what a term is,
    what the document canonicalizes to -- is a session-injected callable.
    This just puts JSON in a list.
    """
    part = term.get("part", "exchange")
    document = dict(document or {"schema": term["schema"], "variables": {},
                                 "parts": {}})
    document["variables"] = dict(document.get("variables") or {})
    document["variables"].update(term.get("variables") or {})
    parts = dict(document.get("parts") or {})
    existing = parts.get(part) or {"kind": "sum", "terms": []}
    parts[part] = {"kind": "sum", "terms": list(existing["terms"]) + [term["term"]]}
    document["parts"] = parts
    return document


def _disabled_note(session, capability: str) -> html.Div:
    """Why an area cannot act. Rendered in place of nothing (D33)."""
    return html.Div(
        session.why_not(capability),
        style={"fontSize": "11px", "color": "#9aa1a9", "margin": "4px 0"},
    )


def propose_box(session) -> html.Div:
    """The manual-coefficient path: boxes for the *working form*, and submit.

    The boxes used to be the run's form's, laid out once at startup and never
    again -- which was fine while there was one form and wrong the moment the
    structural editor could build another. They are a callback output now
    (``_working_form``), so building a term above changes the coefficients
    below and "edit this form's numbers by hand" means *this* form rather
    than the one the session happened to start with.

    Always rendered, whatever the session can do. Without ``submit`` the
    inputs are disabled and the reason is stated -- which is the whole of the
    session collapse: one layout, and capability is *data* the areas read
    rather than a branch the layout takes.
    """
    can_submit = session.can("submit")
    if can_submit and not session.coefficients:
        raise ValueError(
            "this session can submit but declares no coefficient boxes: the "
            "submit button would post an empty theta. The composition root "
            "builds both from the run's form -- see "
            "`run_stage2.coefficient_layout`."
        )
    children = [html.Div("coefficients", style={"fontSize": "11px",
                                                "color": "#7a828c",
                                                "marginBottom": "2px"})]
    if not can_submit:
        children.append(_disabled_note(session, "submit"))
    children += [
        # Which form these boxes belong to, published so the step buttons clip
        # against *its* bounds rather than the session's opening ones.
        dcc.Store(id="working-form"),
        html.Div(id="coefficient-boxes",
                 children=initial_coefficient_rows(session)),
        html.Button("submit", id="submit", n_clicks=0, disabled=not can_submit,
                    style={"marginTop": "8px"}),
        # What was last submitted, so the line under the button can be
        # recomputed on the tick instead of frozen at the click.
        dcc.Store(id="last-submission"),
        html.Div(id="submitted", style={"fontSize": "13px", "marginTop": "6px"}),
    ]
    return _subbox("edit coefficients by hand", *children)


def initial_coefficient_rows(session) -> list:
    """The boxes the page is served with, before any callback has run.

    The session's opening form: a page that arrived showing nothing and
    filled in on the first poll would flicker, and a store-only viewer -- who
    gets no poll worth the name -- would see an empty panel instead of a
    disabled one with a reason.
    """
    return coefficient_rows(
        [tuple(entry["bounds"]) for entry in session.coefficients],
        [entry.get("part") for entry in session.coefficients],
        disabled=not session.can("submit"),
    )


def new_run_box(session) -> html.Div:
    """Its own pane, at the bottom: it is about the *run*, not the form."""
    return _subbox(
        "run",
        html.Button("start a new run", id="new-run", n_clicks=0,
                    disabled=not session.can("new_run")),
        *([] if session.can("new_run") else [_disabled_note(session, "new_run")]),
        html.Div(id="new-run-status",
                 style={"fontSize": "13px", "marginTop": "6px", "color": "#4a525c"}),
    )


def _dropdown_options(catalogue) -> list[dict]:
    """A session catalogue as Dash dropdown options, whatever shape it is in.

    Three shapes reach here and all three are reasonable things for a
    composition root to hand over: ``{name: {"label": ...}}`` (the palette),
    ``{name: "a sentence"}`` (the objectives -- the prose *is* the label),
    and a bare list of names. Written once because the alternative was
    written twice and only one of them handled a string, which is how a live
    `--human` session died at `build_layout` while every test passed: the
    tests built sessions whose objectives were the empty default.

    **Declaration order, not alphabetical.** The composition root writes
    these catalogues in a deliberate order and the first entry becomes the
    dropdown's initial value, so sorting them silently chose the default.
    It had already gone wrong once: alphabetising the objectives selected
    `both`, while the delegate path's own fallback for an unset objective is
    `dataset_error` -- two defaults for one choice, disagreeing.
    """
    if not catalogue:
        return []
    if not isinstance(catalogue, dict):
        return [{"label": str(name), "value": name} for name in catalogue]
    options = []
    for name, entry in catalogue.items():
        label = entry.get("label", name) if isinstance(entry, dict) else entry
        options.append({"label": str(label), "value": name})
    return options


def editor_box(session) -> html.Div:
    """The structural editor: a palette and a list, never a formula parser (D35).

    The human *is* the form proposer here -- powers are structure (D22), so
    the exponent is something a person sets and the coefficient optimizer
    never sees. Freeform latex input is deliberately out of scope: the
    Dash-era editor manipulates documents structurally and the
    representation's own latex renders the preview, so what you see is what
    the canonical document says rather than what you typed.

    Two submission paths, both in this box:

    * **manual** -- coefficients through the unchanged ``HumanOptimizer`` port,
      in the box above;
    * **the delegate path** -- an optimize button that composes (current
      document x selected objective x protocol) into a *driver-side run*. The
      human proposes the form; the machine fits the coefficients. That is
      D11's two-level search with a person in the outer seat, and because the
      run is a proper recorded run under a protocol row, it is consistent with
      the amended D13 by construction rather than by care.
    """
    can_edit = session.can("edit")
    palette = _dropdown_options(session.palette)
    objectives = _dropdown_options(session.objectives)
    optimizers = [
        dict(option, disabled=not (session.optimizers or {})
             .get(option["value"], {}).get("available", True))
        for option in _dropdown_options(session.optimizers)
    ]
    children = [
        dcc.Store(id="editor-document"),
        html.Div("build a form", style={"fontSize": "11px", "color": "#7a828c",
                                        "marginBottom": "4px"}),
    ]
    if not can_edit:
        children.append(_disabled_note(session, "edit"))
    children += [
        html.Div(
            [
                dcc.Dropdown(id="editor-palette", options=palette,
                             value=palette[0]["value"] if palette else None,
                             disabled=not can_edit, clearable=False,
                             style={"flex": "1 1 160px", "fontSize": "12px"}),
                dcc.Input(id="editor-power-num", type="number", value=1, step=1,
                          disabled=not can_edit,
                          style={"width": "54px", "marginLeft": "6px"}),
                html.Span("/", style={"margin": "0 3px", "color": "#9aa1a9"}),
                dcc.Input(id="editor-power-den", type="number", value=2, step=1,
                          min=1, disabled=not can_edit, style={"width": "54px"}),
                html.Button("add", id="editor-add", n_clicks=0,
                            disabled=not can_edit, style={"marginLeft": "6px"}),
                html.Button("clear", id="editor-clear", n_clicks=0,
                            disabled=not can_edit, style={"marginLeft": "4px"}),
            ],
            style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"},
        ),
        html.Div(id="editor-terms",
                 style={"fontSize": "11.5px", "fontFamily": "monospace",
                        "color": "#4a525c", "margin": "6px 0"}),
        dcc.Markdown(id="editor-preview", mathjax=True),
        html.Div(
            [
                html.Button("check", id="editor-check", n_clicks=0,
                            disabled=not session.can("check")),
                dcc.Dropdown(id="editor-objective", options=objectives,
                             value=objectives[0]["value"] if objectives else None,
                             disabled=not session.can("propose"), clearable=False,
                             style={"flex": "1 1 180px", "marginLeft": "6px",
                                    "fontSize": "12px"}),
            ],
            style={"display": "flex", "alignItems": "center", "marginTop": "6px",
                   "flexWrap": "wrap"},
        ),
        # The delegated run's own two knobs. Both are the person's: a run
        # nobody bounded ran at ~450 ms/trial until the process was killed,
        # and "which optimizer" was a decision buried in the composition root
        # that only the person watching the scores can make well.
        html.Div(
            [
                dcc.Dropdown(id="editor-optimizer", options=optimizers,
                             value=_first_available(optimizers),
                             disabled=not session.can("propose"),
                             clearable=False,
                             style={"flex": "1 1 200px", "fontSize": "12px"}),
                html.Span("trials", style={"margin": "0 4px 0 8px",
                                           "fontSize": "12px",
                                           "color": "#7a828c"}),
                dcc.Input(id="editor-trials", type="number",
                          value=session.default_trials, min=1, step=10,
                          disabled=not session.can("propose"),
                          style={"width": "78px"}),
                html.Button("optimize", id="editor-optimize", n_clicks=0,
                            disabled=not session.can("propose"),
                            style={"marginLeft": "6px"}),
            ],
            style={"display": "flex", "alignItems": "center", "marginTop": "6px",
                   "flexWrap": "wrap"},
        ),
        # Polled, not pushed: the click returns "handed over" and this line
        # is what says how far it got. Without it a bounded run looks
        # identical to a hung one.
        html.Div(id="delegate-progress",
                 style={"fontSize": "12px", "color": "#4a525c",
                        "marginTop": "6px"}),
        html.Div(id="editor-findings",
                 style={"fontSize": "11.5px", "color": "#4a525c",
                        "whiteSpace": "pre-wrap", "marginTop": "6px"}),
    ]
    return _subbox("structural editor", *children)


def _first_available(options: list[dict]) -> str | None:
    """The first option that can actually run, else the first one.

    An unavailable optimizer is still *offered* -- with its reason in the
    label, so the extra is discoverable -- but it must never be what the
    dropdown starts on, or the obvious click fails for a reason the person
    did not choose.
    """
    for option in options:
        if not option.get("disabled"):
            return option["value"]
    return options[0]["value"] if options else None


# A tick-driven spinner. Text rather than CSS because the page ships no
# stylesheet and an animation that needs one would be a new dependency for a
# progress glyph; the interval callback is already running, so the frame is
# free. It advances only while something is *actually* in flight, which is
# what makes a stopped spinner mean "stopped" rather than "not implemented".
_SPINNER = "|/-\\"


def spinner(tick) -> str:
    return _SPINNER[int(tick or 0) % len(_SPINNER)]


def run_activity(session, tick):
    """One line in the run pane: what the worker is doing, right now.

    Three things can be true, and only one of them was visible anywhere: a
    delegated run is grinding through its trials, one has just finished, or
    the session is idle waiting for a submission. The delegate button's reply
    said "handed to cmaes for 6 trials" once, at the click, in the editor --
    which is neither where runs are nor a claim that stays true.
    """
    state = session.delegation() if session.delegation is not None else None
    if state:
        done, total = state.get("done", 0), state.get("trials", 0)
        where = f"run {state.get('run')} ({state.get('optimizer', '?')})"
        status = state.get("state", "")
        if status == "running":
            return html.Span(
                f"{spinner(tick)} {where}: trial {done}/{total}",
                style={"color": "#8a5a00"},
            )
        if status == "finished":
            return html.Span(f"done -- {where} finished {done}/{total} trials; "
                             "the coefficient boxes are yours again",
                             style={"color": "#1e7a34"})
        return html.Span(f"{where}: {status}", style={"color": "#b3261e"})
    if session.submissions is not None:
        pending = (session.submissions() or {}).get("pending", 0)
        if pending:
            return html.Span(f"{spinner(tick)} evaluating {pending} submitted "
                             f"candidate{'s' if pending > 1 else ''}",
                             style={"color": "#8a5a00"})
        return html.Span("idle -- waiting for a submission",
                         style={"color": "#7a828c"})
    return ""


# The checker's vocabulary, ranked (`forms.checker`: REFUSAL/WARNING/NOTE).
# Spelled out because the headline said "error is the worst" against a set
# that contains no such severity -- so `refusal`, the one that means the form
# *cannot be composed*, was reported in the same words as a costing note.
_SEVERITY_RANK = {"note": 1, "warning": 2, "refusal": 3}
_SEVERITY_COLOUR = {"note": "#4a525c", "warning": "#8a5a00", "refusal": "#b3261e"}
_HEADLINE = {
    "note": "PASS with notes",
    "warning": "PASS with warnings -- this form will run",
    "refusal": "REFUSED -- this form cannot be composed against this world",
}


def findings_report(findings: list, run_number: int) -> html.Div:
    """What the check button produced, as something that visibly *happened*.

    A clean check used to render one grey sentence, and clicking again
    rendered the identical grey sentence -- so the honest reading of the
    button was "nothing happens". Two changes, both about that: a green or
    red headline that differs from the surrounding text, and the check's
    ordinal, so a second click on an unchanged form is still visibly a second
    click. Findings themselves are unchanged; they were already the useful
    part.
    """
    ordinal = f"check #{run_number}"
    if not findings:
        return html.Div(
            [
                html.Div(f"PASS -- no findings ({ordinal})",
                         style={"color": "#1e7a34", "fontWeight": 600}),
                html.Div("this form composes against this world: every "
                         "ingredient is supplied, the units balance, and no "
                         "term is undefined on the declared domain.",
                         style={"color": "#4a525c", "marginTop": "2px"}),
            ],
            style={"fontSize": "11.5px"},
        )
    worst = max((f.get("severity") for f in findings),
                key=lambda s: _SEVERITY_RANK.get(s, 0))
    lines = [html.Div(
        f"{_HEADLINE[worst]} -- {len(findings)} "
        f"finding{'' if len(findings) == 1 else 's'} ({ordinal})",
        style={"color": _SEVERITY_COLOUR[worst], "fontWeight": 600},
    )]
    for f in findings:
        lines.append(html.Div(
            [
                html.Div(f"{f.get('severity', '?')}: {f.get('what', '')}",
                         style={"fontWeight": 600}),
                html.Div(f"why: {f.get('why', '')}", style={"color": "#4a525c"}),
                html.Div(f"remedy: {f.get('remedy', '')}", style={"color": "#4a525c"}),
            ],
            style={"marginTop": "4px"},
        ))
    return html.Div(lines, style={"fontSize": "11.5px"})


def solve_line(session, candidate, system):
    """What the live-solve line says about one (candidate, system).

    Four states, and the old line collapsed three of them into "queued...":
    nothing has been asked for, one is running, one finished, one *failed*.
    Rendering "queued" for the first meant the panel announced work it had
    never been asked to do -- on every tick, for any selection -- and
    rendering it for the last meant a solve that raised looked identical to
    one still going, permanently.

    Failures are shown, not logged: the exception is already caught and kept
    by the service (B10's rule -- a worker that dies quietly is the failure
    mode), and keeping it out of the panel wasted the catching.
    """
    state = (session.solve_state(candidate, system)
             if session.solve_state is not None else None)
    result = session.collect(candidate, system) if session.collect else None
    if state == "failed" or (state is None and result is None):
        detail = (session.solve_failure(candidate, system)
                  if session.solve_failure is not None else None)
        if state != "failed":
            return ""      # no reading available: say nothing, never guess
        return html.Span(f"live: solve failed -- {detail or 'no detail recorded'}",
                         style=_FAILED_STYLE)
    if state == "idle":
        return ""
    if result is None:
        return html.Span("live: solving...", style=_LIVE_STYLE)
    return html.Span(
        "live (not stored): "
        + "   ".join(f"{k} = {v}" for k, v in sorted(result.items())),
        style=_LIVE_STYLE,
    )


def delegation_line(state: dict | None) -> str:
    """The delegated run's progress, as one sentence."""
    if not state:
        return ""
    done, trials = state.get("done", 0), state.get("trials", 0)
    where = f"run {state.get('run')}, {state.get('optimizer', '?')}"
    status = state.get("state", "")
    if status == "running":
        return f"{where}: trial {done}/{trials}"
    if status == "finished":
        return (f"{where}: finished {done}/{trials} trials -- the sliders are "
                "yours again")
    return f"{where}: {status} at trial {done}/{trials}"


def functional_card(session) -> html.Div:
    """The functional panel: the form in its own parts, and two ways to propose.

    The boxes are no longer "exchange" and "correlation" by assumption. They
    are whatever parts the selected candidate's *document* declares, filled in
    by the callback, so a two-part form and a five-part form render alike and
    neither is a special case here.
    """
    return _card(
        "functional",
        # Top to bottom is the order the work happens in: build the form,
        # read what it became, set its numbers, then decide about the run.
        # The middle two follow the first -- `_working_form` writes both from
        # whatever the editor holds, falling back to the selected candidate.
        editor_box(session),
        # One container, filled per working form. Which parts a form has is a
        # property of the form, so it cannot be a property of the layout.
        html.Div(id="formula-parts"),
        html.Div(id="theta"),
        propose_box(session),
        new_run_box(session),
        hint=f"{_FUNCTIONAL_HINT}\n\n{_PROPOSAL_HINT}",
    )


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


def build_layout(store_path: str, session=None) -> html.Div:
    """The initial page, and there is only one of it (D33).

    Every dropdown here is refreshed by ``_choices``.

    What this builds is a *starting* set of options, not the set: a store
    gains runs, candidates and system versions while the page is open -- a
    human session's own run does not exist until its first submission lands
    (B1) -- and ``app.layout`` is one object served to every visitor, so
    nothing built here is refreshed by reloading the browser either.
    """
    from .session import Session

    session = session or Session()
    runs = run_options(store_path)
    baselines = baseline_options(store_path)
    systems = system_options(store_path)
    store_world = data.world(store_path)

    panels = [
        html.H1("xcsearch", style={"fontSize": "20px", "marginBottom": "2px"}),
        # Which file, and which world's history is in it. The store is not
        # switchable from here on purpose -- in a live session the search
        # worker owns every write to exactly one file, and a viewer that
        # could point elsewhere would show one store while the submit button
        # fed another. So the page *states* its store instead of choosing it.
        html.Div(
            [
                html.Span(os.path.abspath(store_path)),
                html.Span(
                    f"  ·  world: {store_world or 'empty'}",
                    style={"color": "#9aa1a9"},
                ),
            ],
            style={"color": "#7a828c", "fontSize": "12px"},
        ),
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
                # What the worker is doing *right now*, in the pane about
                # runs. The delegate button's own reply sits at the bottom of
                # the editor, four panes away and written once at the click;
                # a run in progress should be visible where runs are.
                html.Div(id="run-activity",
                         style={"fontSize": "12px", "marginTop": "6px",
                                "fontFamily": "monospace"}),
                hint=_RUN_HINT,
            ),
            functional_card(session),
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
                # The live face of the candidate-SCF service (D34). Per-system
                # click, never a whole-population sweep: that was measured at
                # ~363x the per-trial cost with endemic non-convergence on
                # stretched geometries, so it is a possible future feature and
                # not this one.
                html.Div(
                    [
                        html.Button("solve self-consistently", id="solve",
                                    n_clicks=0,
                                    disabled=not session.can("solve")),
                        *([] if session.can("solve")
                          else [_disabled_note(session, "solve")]),
                        html.Div(id="live-solve",
                                 style={"marginTop": "6px"}),
                    ],
                    style={"margin": "2px 0 8px"},
                ),
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


def submission_line(session, last: dict) -> str:
    """What became of the last thing submitted, as of now.

    Three states, and the reason to distinguish them is that a person who has
    just clicked wants to know which one they are in: still queued, being
    evaluated, or done and worth this much. The score is the interesting one
    -- it is the number the session was told (D20), read back from the
    endpoint's own history rather than recomputed here.

    The session reports both through one reading, ``submissions()``. An
    endpoint that keeps a history gets the better line; one that does not
    still gets a true one, which is why this reads a dict with absent keys
    rather than reaching into an optimizer.
    """
    theta = ", ".join(f"{c:.4f}" for c in last["theta"])
    state = session.submissions() if session.submissions is not None else {}
    scores = state.get("scores") or []
    if len(scores) > last["index"]:
        return f"submitted [{theta}] - evaluated, score {scores[last['index']]:.4e}"
    pending = state.get("pending") or 0
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


def create_app(store_path: str, session=None) -> Dash:
    """Build the app. Every callback re-reads the store, so panels live-tail.

    **One parameter besides the store, and one layout always (D33).** What the
    page can *do* is the injected ``session``: capabilities are data it
    carries, every callback below is registered unconditionally, and one
    without its capability returns a sentence saying so. That retires F1.9 --
    the served page's callback arity no longer moves with configuration, so a
    browser holding an older page cannot post a request this server cannot
    parse.

    Entry points differ only in the session. A live run enables everything; a
    dataset cache with no run enables solves while submission refuses with a
    reason; a store alone disables the physics-dependent areas and says why.
    There are no view modes, deliberately.
    """
    from .session import Session

    session = session or Session()
    app = Dash(__name__)
    app.title = "xcsearch"
    app.layout = build_layout(store_path, session)

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
        Output("formula-parts", "children"),
        Output("coefficient-boxes", "children"),
        Output("working-form", "data"),
        Output("theta", "children"),
        Input("editor-document", "data"),
        Input("selected", "data"),
    )
    def _working_form(document, selected):
        """The form being worked on, rendered and made editable.

        One callback for both panes because they are two views of one fact,
        and two callbacks would race: the boxes could be a form ahead of the
        latex above them, which is exactly the state in which someone submits
        the wrong theta.

        Precedence is the editor, then the selected candidate, then the
        session's opening form. That makes **clear** a way back to the
        candidate view rather than a dead end, and it means a store-only
        viewer -- which has neither editor nor submit -- reads exactly as it
        did before.
        """
        editing = bool(document and document.get("parts"))
        if editing and session.canonicalize is not None:
            try:
                summary = session.canonicalize(document)
            except Exception as failure:  # noqa: BLE001 - shown, never a traceback
                return (html.Div(f"{type(failure).__name__}: {failure}",
                                 style={"color": "#b3261e", "fontSize": "12px"}),
                        no_update, no_update, no_update)
            slots = summary.get("slots") or []
            return (
                working_form_boxes(summary.get("parts") or {}, "from the editor"),
                coefficient_rows(
                    [tuple(s["bounds"]) for s in slots],
                    [s.get("part") for s in slots],
                    disabled=not session.can("submit"),
                ),
                {"document": document, "slots": slots, "source": "editor"},
                identity_line(editing=True, form_hash=summary.get("hash"),
                              n_slots=len(slots)),
            )
        if selected:
            candidate_id = selected["candidate_id"]
            record = data.candidate(store_path, candidate_id) or {}
            form = data.form_for(store_path, candidate_id)
            slots = (form or {}).get("slots") or []
            return (
                formula_parts(record, form),
                coefficient_rows(
                    [tuple(s["bounds"]) for s in slots],
                    [s.get("part") for s in slots],
                    disabled=not session.can("submit"),
                ) if slots else initial_coefficient_rows(session),
                {"document": None,
                 "slots": slots or _session_slots(session),
                 "source": "candidate"},
                identity_line(editing=False, candidate_id=candidate_id,
                              trial=selected.get("trial"),
                              score=selected.get("score"),
                              theta=record.get("theta") or []),
            )
        return (
            dcc.Markdown("*no candidate selected*"),
            initial_coefficient_rows(session),
            {"document": None, "slots": _session_slots(session), "source": "session"},
            "",
        )

    @app.callback(
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
        """The comparison figures. The identity line above them belongs to
        `_working_form`, which is what decides *which form* the panel is
        about -- printing it here left it describing the selected candidate
        while the latex directly above it showed the editor's form."""
        if not selected or baseline_id is None:
            return (figures.heatmap([]), figures.condition_bars([]), "", "")
        candidate_id = selected["candidate_id"]

        exact = baseline_id == data.EXACT
        baseline = {} if exact else (data.candidate(store_path, baseline_id) or {})
        candidate_label = f"candidate {candidate_id}"
        baseline_label = ("exact reference" if exact else
                          baseline.get("label") or f"candidate {baseline_id}")
        # One query, two figures: the grid was half empty because these two
        # halves share no metric and the conditions have no regime.
        per_system, conditions = data.split_conditions(
            data.heatmap(store_path, candidate_id, baseline_id))
        return (
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
        follow = session.follow
        following = follow() if callable(follow) else follow
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
        return session.status() if session.can("status") else ""

    match, context = dash_match(), dash_context()

    @app.callback(
        Output({"type": "theta", "index": match}, "value"),
        Input({"type": "theta-down", "index": match}, "n_clicks"),
        Input({"type": "theta-up", "index": match}, "n_clicks"),
        State({"type": "theta", "index": match}, "value"),
        State("working-form", "data"),
        prevent_initial_call=True,
    )
    def _step(_down, _up, value, working):
        """Move one coefficient by one step, in the direction clicked.

        The one place here that asks dash which input fired, and the one place
        it is the honest answer: the two buttons differ in nothing but
        direction, and unlike the submit line there is no previously published
        state to read the difference out of.

        The bounds come from the *working form* rather than the component's
        own min/max, so the box a value is clipped into is the form's and not
        whatever the browser sent back. It used to read the session's opening
        coefficients, which was the same thing only while there was one form
        -- with the editor able to build another, that would clip a new
        form's coefficient into the old form's box.
        """
        fired = context.triggered_id
        if fired is None:
            return no_update
        boxes = (working or {}).get("slots") or _session_slots(session)
        if fired["index"] >= len(boxes):
            return no_update
        low, high = boxes[fired["index"]]["bounds"]
        return stepped(value, -1 if fired["type"] == "theta-down" else 1, low, high)

    @app.callback(
        Output("submitted", "children"),
        Output("last-submission", "data"),
        Input("submit", "n_clicks"),
        Input("tick", "n_intervals"),
        State({"type": "theta", "index": dash_all()}, "value"),
        State("last-submission", "data"),
        State("working-form", "data"),
        prevent_initial_call=True,
    )
    def _submit(n_clicks, _tick, theta, last, working):
        """Queue a proposal on a click, and keep its line true after.

        The line is recomputed on the tick rather than written once at the
        click: it is a claim about a live thing (is the trial queued, being
        evaluated, or done?), and a claim about a live thing that is written
        once goes stale by construction.

        Registered whatever the session can do. Without ``submit`` it says so
        and returns -- refusing by name, on a normal path, is what replaced
        "the callback does not exist" (F1.9).
        """
        if not session.can("submit"):
            return session.refuse("submit"), no_update
        submitting = not last or n_clicks != last.get("clicks")
        if submitting:
            if not n_clicks:
                return no_update, no_update
            empty = [f"c{i}" for i, c in enumerate(theta) if c is None]
            if empty:
                # The click is still recorded, with no theta on it. Not doing
                # so would leave this looking like an unsubmitted click on
                # every tick after, and it would resubmit.
                return (f"nothing submitted: {', '.join(empty)} empty",
                        {"clicks": n_clicks})
            # The document goes with the theta. A coefficient means nothing
            # without the form it is a coefficient *of*, and the boxes now
            # follow the editor -- so submitting the numbers alone would
            # silently apply a new form's theta to the run's old form.
            accepted = session.submit(theta, (working or {}).get("document"))
            last = {"clicks": n_clicks,
                    "theta": [float(c) for c in accepted],
                    # Where this proposal's result will land, so its outcome
                    # can be recognised rather than guessed at.
                    "index": len(getattr(session.submit, "history", ()))}
        elif not last.get("theta"):
            return no_update, no_update
        return submission_line(session, last), last

    @app.callback(
        Output("new-run-status", "children"),
        Input("new-run", "n_clicks"),
        prevent_initial_call=True,
    )
    def _new_run(n_clicks):
        """Ask the session for a new run and report what it said.

        The reply is the session's, not this callback's: whether a request is
        worth honouring depends on how many candidates the current run has,
        which is the worker's business to know.
        """
        if not n_clicks:
            return no_update
        if not session.can("new_run"):
            return session.refuse("new_run")
        return session.new_run()

    # -- the structural editor (D35) -------------------------------------
    @app.callback(
        Output("editor-document", "data"),
        Output("editor-terms", "children"),
        Output("editor-preview", "children"),
        Input("editor-add", "n_clicks"),
        Input("editor-clear", "n_clicks"),
        State("editor-palette", "value"),
        State("editor-power-num", "value"),
        State("editor-power-den", "value"),
        State("editor-document", "data"),
        prevent_initial_call=True,
    )
    def _edit(add_clicks, clear_clicks, choice, numerator, denominator, document):
        """Add or clear a term. The document transits as plain JSON.

        Both operations that need to know what a form *is* -- building a term
        from the palette, and saying what the document canonicalized to -- are
        session-injected callables. This callback moves JSON around and reads
        strings back, which is why `tests/test_import_audit.py` is unchanged.
        """
        if not session.can("edit"):
            return no_update, session.refuse("edit"), no_update
        fired = context.triggered_id
        if fired == "editor-clear":
            return None, "no terms yet", ""
        if not add_clicks or choice is None:
            return no_update, no_update, no_update
        try:
            term = session.term(choice, (int(numerator or 1), int(denominator or 1)))
            document = _with_term(document, term)
            summary = session.canonicalize(document)
        except Exception as failure:  # noqa: BLE001 - shown, never a traceback
            return no_update, f"{type(failure).__name__}: {failure}", no_update
        lines = [f"{i}. {t}" for i, t in enumerate(summary.get("terms", []))]
        return (
            document,
            html.Div([html.Div(line) for line in lines] or "no terms yet"),
            f"$${summary.get('latex', '')}$$" if summary.get("latex") else "",
        )

    @app.callback(
        Output("editor-findings", "children"),
        Input("editor-check", "n_clicks"),
        Input("editor-optimize", "n_clicks"),
        State("editor-document", "data"),
        State("editor-objective", "value"),
        State("editor-optimizer", "value"),
        State("editor-trials", "value"),
        prevent_initial_call=True,
    )
    def _check_or_optimize(check_clicks, optimize_clicks, document, objective,
                           optimizer, trials):
        """The check button and the delegate path, side by side.

        The check button calls the session's ``check`` -- which is
        ``forms.checker.check``, wired, not reimplemented: one function for
        the machine proposer and for the person, or the two diverge.

        The optimize button composes (document x objective x protocol) into a
        driver-side run through the session's ``propose``. That is the
        recorded face: a proper run under a D18 protocol row, indistinguishable
        in the store from a script-composed one.
        """
        fired = context.triggered_id
        if fired == "editor-optimize":
            if not optimize_clicks:
                return no_update
            if not session.can("propose"):
                return session.refuse("propose")
            if not document:
                return "nothing to optimize: the editor has no terms yet"
            if not trials or int(trials) < 1:
                return "a delegated run needs at least one trial"
            return session.propose(document, objective, optimizer, int(trials))
        if not check_clicks:
            return no_update
        if not session.can("check"):
            return session.refuse("check")
        if not document:
            return "nothing to check: the editor has no terms yet"
        return findings_report(session.check(document), check_clicks)

    @app.callback(
        Output("delegate-progress", "children"),
        Input("tick", "n_intervals"),
    )
    def _delegate_progress(_tick):
        """How far the delegated run has got. Polled, like every other
        reading -- the click that started it returned long before."""
        if session.delegation is None:
            return ""
        return delegation_line(session.delegation())

    @app.callback(
        Output("run-activity", "children"),
        Input("tick", "n_intervals"),
    )
    def _run_activity(tick):
        """Is anything running? Answered in the pane about runs.

        The delegate button's reply is written once, at the click, at the
        bottom of the editor -- neither where a person looks for a run nor a
        claim that stays true. This is polled, and the spinner advances only
        while something is actually in flight.
        """
        return run_activity(session, tick)

    # -- the live face of the SCF service (D34) ---------------------------
    @app.callback(
        Output("live-solve", "children"),
        Input("solve", "n_clicks"),
        Input("tick", "n_intervals"),
        State("system", "value"),
        State("selected", "data"),
        prevent_initial_call=True,
    )
    def _solve(n_clicks, _tick, system, selected):
        """Per-system click: queue a solve, collect it on the tick, mark it live.

        Nothing here is ever stored. The rule (D34) is that any number on
        screen is either in a record, reachable from one, or *explicitly
        marked as a live diagnostic that is not* -- and this is the third
        kind, so it says so every time it renders. Non-convergence renders as
        data, because on these geometries it is signal: near-degenerate
        stretched chains genuinely do not settle.
        """
        if not session.can("solve"):
            return session.refuse("solve") if n_clicks else ""
        if not selected or system is None:
            return ""
        candidate = selected["candidate_id"]
        if context.triggered_id == "solve" and n_clicks:
            session.solve(candidate, system)
        return solve_line(session, candidate, system)

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
