"""Content hashes and protocol payloads: the provenance spine (D17, D18).

Stage 2B's organizing finding is that every stored number is a measurement, and
a measurement's meaning is a four-tuple: *what* was measured, *against what*,
*how*, and *among what*. This module owns the two identities that make the
middle coordinates nameable --

* :func:`system_content_hash` -- *against what*. A hash over the physics
  content of one frozen reference system, so that a rebuilt cache whose numbers
  moved produces a different system version rather than overwriting the one
  older evaluations were measured against (D17).
* :func:`protocol_payload` / :func:`protocol_hash` -- *how*. The grid, the
  floors, the anchor window, the evaluator stack and its metric list, the
  library versions and the git commit, as one hashable dict (D18).

World-free and third-party-free (numpy and the standard library). The sparse
Laplacian is handled duck-typed rather than by importing scipy: a record's
operator is whatever carries ``data``/``indices``/``indptr``, and this module
has no opinion about which library made it.

The repair rule this serves: *any field consulted at read time to interpret a
stored measurement must be either immutable or versioned.* A hash is how a
value becomes an identity, and an identity is what makes append-only possible.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

# Bumped only if the *layout* of the hashed byte stream changes. It is part of
# the digest, so an old cache's stored hash can never silently collide with a
# new scheme's -- the two are different strings for the same arrays, which is
# a visible new version rather than a false match.
SYSTEM_HASH_VERSION = "xcsearch.system.v1"
PROTOCOL_HASH_VERSION = "xcsearch.protocol.v1"

# What a system's identity is made of (D17): the arrays that define what a
# candidate was measured *against*. Deliberately not `regime` or `split` --
# those are assignment, not identity, and they belong to the population a run
# was measured over (D19). Deliberately not `name` either: the name is the
# other half of the `(name, content_hash)` key, so renaming a system is a
# different row rather than a different version of the same one.
#
# `e_exact` is absent because it is determined by what is here -- it is
# `e_xc_exact` plus the kinetic, external and Hartree energies of `n_ref` on
# this grid -- so hashing it would add no discrimination, only a second way to
# spell the same difference.
_HASHED_ARRAYS = ("x", "w", "n_ref", "v_ext", "v_hartree", "vxc_ref")
_HASHED_SCALARS = ("e_xc_exact",)


def _feed(digest, label: str, value) -> None:
    """Absorb one labelled field, with its shape and dtype, into the digest.

    The label and the shape go in beside the bytes so that two different
    fields cannot produce the same stream by accident -- a hash over raw
    concatenated bytes would let a longer ``x`` and a shorter ``w`` trade
    length without changing the digest.
    """
    digest.update(label.encode())
    if value is None:
        digest.update(b"none")
        return
    array = np.ascontiguousarray(value)
    if array.dtype.kind not in "fiub":
        raise TypeError(f"{label}: {array.dtype} is not a hashable array dtype")
    digest.update(str(array.dtype.str).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())


def _feed_sparse(digest, label: str, operator) -> None:
    """Absorb a sparse operator by its CSR triple, or its absence.

    Duck-typed on purpose (see the module docstring). Anything that can become
    CSR is hashed as the three arrays a CSR matrix *is*, which is also exactly
    what the ``.npz`` cache stores -- so the digest survives the round trip
    through the file rather than depending on the object that made it.
    """
    if operator is None:
        digest.update(f"{label}none".encode())
        return
    csr = operator.tocsr() if hasattr(operator, "tocsr") else operator
    _feed(digest, f"{label}.data", csr.data)
    _feed(digest, f"{label}.indices", csr.indices)
    _feed(digest, f"{label}.indptr", csr.indptr)
    digest.update(str(csr.shape).encode())


def system_content_hash(
    *,
    x,
    w,
    n_ref,
    e_xc_exact: float,
    v_ext=None,
    v_hartree=None,
    vxc_ref=None,
    laplacian=None,
) -> str:
    """The content hash of one frozen system: physics content only (D17).

    Keyword-only, and every field explicit, because this is the function whose
    *input set* is the decision. Adding a field here changes every system's
    identity, which is a deliberate act with a migration behind it, not
    something to do while passing a record around.

    Computed once at cache-build time and stored in both the ``.npz`` and the
    ``systems`` row; recomputed on load only for a cache built before 2B.1,
    which is the same arithmetic over the same arrays.
    """
    digest = hashlib.sha256(SYSTEM_HASH_VERSION.encode())
    values = {
        "x": x, "w": w, "n_ref": n_ref,
        "v_ext": v_ext, "v_hartree": v_hartree, "vxc_ref": vxc_ref,
    }
    for field in _HASHED_ARRAYS:
        _feed(digest, field, values[field])
    for field in _HASHED_SCALARS:
        _feed(digest, field, np.float64(e_xc_exact))
    _feed_sparse(digest, "laplacian", laplacian)
    return digest.hexdigest()


def record_content_hash(system) -> str:
    """:func:`system_content_hash` for an in-memory ``SystemRecord``."""
    return system_content_hash(
        x=system.x,
        w=system.w,
        n_ref=system.n_ref,
        e_xc_exact=system.e_xc_exact,
        v_ext=system.v_ext,
        v_hartree=system.v_hartree,
        vxc_ref=system.vxc_ref,
        laplacian=system.laplacian,
    )


# --------------------------------------------------------------------------
# The protocol (D18)
# --------------------------------------------------------------------------
def library_versions() -> dict[str, str]:
    """Versions of the libraries whose arithmetic the numbers depend on.

    Best effort by design: a library that is absent is absent from the dict
    rather than an exception at run start. The protocol row is a breadcrumb,
    and a breadcrumb that can abort a search is worse than a missing crumb.
    """
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "optuna", "cma", "dash", "plotly"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - an absent optional extra is not news
            continue
        version = getattr(module, "__version__", None)
        if version is not None:
            versions[name] = str(version)
    return versions


def git_commit() -> str | None:
    """The commit this code is running from, or ``None`` outside a checkout.

    Stored and deliberately unconsumed (D18): nothing reads it back, and
    nothing should start to without saying so. It is here for the day an
    investigation asks "which code produced these numbers", which is a
    question that cannot be answered retroactively if the answer was never
    written down.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except Exception:  # noqa: BLE001 - not a checkout, no git, a submodule, ...
        return None
    return out.stdout.strip() or None


def protocol_payload(
    *,
    world: dict,
    evaluators: list,
    extra: dict | None = None,
) -> dict:
    """The ``how`` of a run, as the dict that becomes a ``protocols`` row.

    ``world`` is the measurement context the dataset owns -- grid parameters,
    margins, floors, the anchor window -- handed over as data so the store
    still never imports a world. ``evaluators`` is the stack itself, each
    member asked to describe itself (``Evaluator.describe``), which is what
    puts the metric list in the row: a protocol that named the evaluators but
    not what they emit would not distinguish a stack that gained a metric.
    """
    payload = {
        "world": dict(world),
        "evaluators": [_describe(ev) for ev in evaluators],
        "libraries": library_versions(),
        "git_commit": git_commit(),
    }
    if extra:
        payload.update(extra)
    return payload


def _describe(evaluator) -> dict:
    describe = getattr(evaluator, "describe", None)
    return describe() if callable(describe) else {"evaluator": type(evaluator).__name__}


def canonical_json(payload: dict) -> str:
    """The one serialization a protocol hash is taken over.

    Sorted keys and no incidental whitespace, so two processes that built the
    same protocol from the same constants agree on the hash rather than on
    dictionary insertion order.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def protocol_hash(payload: dict) -> str:
    digest = hashlib.sha256(PROTOCOL_HASH_VERSION.encode())
    digest.update(canonical_json(payload).encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# The population (D19)
# --------------------------------------------------------------------------
# v2 (3.7, D39): the *role* a system holds in the run -- optimize, evaluate,
# or both -- joins the hash. Same argument as split and regime: it is
# assignment, and a mean over the optimize rows means something different
# from a mean over all of them, so two runs that disagree about it are not
# measuring over the same thing.
POPULATION_HASH_VERSION = "xcsearch.population.v2"


def population_hash(members) -> str:
    """Identity of a population: which system versions, assigned how (D19).

    ``members`` is an iterable of ``(system_version_id, split, regime)`` or,
    since 3.7, ``(system_version_id, split, regime, role)``. A member with no
    role hashes as the empty string, so the *shape* of the tuple is not what
    the digest turns on -- but the version prefix moved to v2 either way,
    because a population that assigns roles is a different population from
    one that does not even when the systems match.
    Order does not matter -- a population is a set, and two runs that loaded
    the same cache in a different order measured over the same thing -- so the
    triples are sorted before hashing.

    The *assignment* is inside the hash, unlike the system content hash which
    excludes it. That is the distinction the whole stage turns on: a rebuilt
    cache that reshuffles the splits changes nothing about what each system
    *is*, and everything about what a mean over them means. Reshuffling
    therefore yields a new population -- with its own baseline -- rather than
    re-scoring the measurements already taken over the old one.

    The ids are store-local, which is correct: a population is a set of rows
    in one store, and two stores' ids are not comparable in the first place.
    """
    rows = sorted(
        (int(member[0]), member[1] or "", member[2] or "",
         (member[3] if len(member) > 3 else None) or "")
        for member in members
    )
    digest = hashlib.sha256(POPULATION_HASH_VERSION.encode())
    digest.update(canonical_json({"members": rows}).encode())
    return digest.hexdigest()
