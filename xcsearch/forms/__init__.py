"""The form representation (Core B): a functional is a document, not a vector.

Four modules, and the split is the ownership rule:

* ``document`` -- node types, canonicalization, serialization, hashing, latex.
  Standard library only (``fractions``, ``json``, ``hashlib``), so the thing
  that defines a form's *identity* can be read, tested and trusted without
  importing physics or a numerical library.
* ``compile`` -- the JAX evaluation path (D23). The only module in the project
  that imports JAX.
* ``checker`` -- ``check(document, supplied, probe)`` (D24). Static checks
  reach no physics; the one dynamic check goes through an injected callable.
* ``library`` -- the seeded documents and the standard-variable fragments.

``forms/`` imports nothing from the rest of the codebase, and nothing outside
the composition layer imports ``forms.compile``: a world publishes its supply
table as plain data, and the composition root is what brings the two together.

This file is deliberately import-free. Re-exporting the document vocabulary
here would give *every* package ``__init__`` a reason to import, and the
ownership table (``tests/test_import_audit.py``) is worth more than the
convenience -- consumers say ``from xcsearch.forms.document import ...`` and a
reader can see which of the four modules they depend on.
"""
