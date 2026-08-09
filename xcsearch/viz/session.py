"""The session: one injected object whose capabilities are data (D33).

The dashboard used to take four constructor parameters -- ``human``,
``status``, ``follow_run``, ``new_run`` -- and grow or shrink around them.
That produced two failure modes worth naming, because the collapse is aimed
at both:

* **F1.9.** A callback's *existence* depended on a constructor argument, so
  the served page's arity moved with configuration. A browser holding
  yesterday's page then posted a request today's server could not parse, and
  the diagnosis was a stack trace about argument counts.
* **View modes.** "Store-only viewer" and "live session" started to read like
  two products. They are not: they are one page injected with different
  capabilities.

So there is **one layout, always**. Every callback is registered
unconditionally. An area whose capability is absent renders disabled *with
the reason stated*, and a callback invoked without its capability refuses by
name rather than doing nothing. Entry points differ only in the session they
inject -- which is what makes "a store-only viewer" a sentence about data
rather than a branch in the code.

Everything on a session is either plain data or a callable the composition
root supplied. The dashboard holds no physics and imports none: ``solve``
runs a Kohn-Sham calculation and ``check`` runs the form checker, but from
here they are functions someone else closed over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The capability names, in the order the panels read them. Named as a tuple
# rather than inferred from the dataclass fields so that adding a field is a
# deliberate act with a UI consequence, not an accident.
CAPABILITIES = (
    "submit", "status", "follow", "new_run", "solve", "check", "propose", "edit",
)

# What a disabled area says. One sentence per capability, written once, so
# that two panels missing the same capability say the same thing -- and so
# that "disabled" is never left to mean "broken".
REASONS = {
    "submit": (
        "this viewer has no search worker to submit to -- open the dashboard "
        "from a `--human` session to propose candidates"
    ),
    "status": "no worker to report on",
    "follow": "not following a live run; pick one from the list",
    "new_run": "only a live session can open a run",
    "solve": (
        "no dataset cache is attached, so a self-consistent solve has nothing "
        "to solve on -- open the dashboard with --data pointing at one"
    ),
    "check": (
        "no form checker is attached; check the document from a session with "
        "one"
    ),
    "propose": (
        "this viewer cannot open a run, so it cannot hand a form to the "
        "optimizer -- open a `--human` session to use the optimize button"
    ),
    "edit": (
        "no form representation is attached, so a document cannot be built or "
        "previewed here"
    ),
}


@dataclass
class Session:
    """What the dashboard is allowed to do, and how.

    Every field is optional and defaults to absent. A session with nothing
    set is exactly a store-only viewer: the page is identical, and every
    interactive area says why it cannot act.
    """

    # Propose a theta to the running search (the feature-4b human optimizer).
    # A plain callable: theta in, the accepted theta back.
    submit: object = None
    # What became of what was submitted: a zero-argument callable returning
    # ``{"pending": int, "scores": [float, ...]}``. Separate from `submit`
    # because it is a *reading*, not an action, and because a session whose
    # endpoint keeps no history still submits -- it just gets a thinner line.
    submissions: object = None
    # Zero-argument callables the page polls.
    status: object = None
    follow: object = None
    new_run: object = None
    # (candidate key, system name) -> a status string; results collected on
    # the tick through `collect`.
    solve: object = None
    collect: object = None
    # A form document (as plain JSON) -> a list of finding dicts.
    check: object = None
    # (document json, objective name) -> a sentence about the run it opened.
    propose: object = None
    # The structural editor's two forms operations (D35). Documents transit
    # the UI as plain JSON and every operation on them is a callable someone
    # else closed over, which is what keeps `tests/test_import_audit.py`
    # unchanged -- the strongest evidence the ports hold.
    #   canonicalize(document json) -> {"hash", "latex", "slots", "terms"}
    #   term(choice, power) -> one term, as json
    canonicalize: object = None
    term: object = None

    # Plain data the panels lay out from.
    coefficients: list = field(default_factory=list)
    objectives: list = field(default_factory=list)
    palette: dict = field(default_factory=dict)

    def can(self, capability: str) -> bool:
        if capability not in CAPABILITIES:
            raise KeyError(
                f"unknown capability {capability!r}: the set is {CAPABILITIES}. "
                "A typo here would silently disable a panel, so it is an error."
            )
        if capability == "edit":
            # Editing is two operations that only make sense together: a
            # palette that can build a term, and a canonicalizer that can say
            # what the document became.
            return self.canonicalize is not None and self.term is not None
        return getattr(self, capability, None) is not None

    def why_not(self, capability: str) -> str:
        """The stated reason an area is disabled. Never a bare 'disabled'."""
        return REASONS.get(capability, f"no {capability} capability")

    def refuse(self, capability: str) -> str:
        """What a callback returns when invoked without its capability.

        Callbacks are registered unconditionally (D33), so this is a normal
        return value on a normal path -- not an exception, and not silence.
        """
        return f"cannot {capability}: {self.why_not(capability)}"

    def capabilities(self) -> dict:
        """The capability map, as data -- what a test asserts over."""
        return {name: self.can(name) for name in CAPABILITIES}


def store_only() -> Session:
    """A viewer over a store and nothing else. The minimum session."""
    return Session()
