"""Flow360 custom Sphinx autodoc extension — entry point.

Registers the unified Flow360Documenter and connects the signature
post-processing hook.  All logic lives in sibling modules:

- ``flow360_documenters``   — the ClassDocumenter subclass
- ``flow360_doc_utils``     — collectors, formatters, signature hook
- ``flow360_type_resolver`` — Flow360 unit-type resolution
- ``flow360_filters``       — member filtering predicates
"""

from __future__ import annotations

from sphinx.application import Sphinx

from flow360_doc_utils import process_autodoc_signature
from flow360_documenters import Flow360Documenter


def setup(app: Sphinx) -> dict:
    app.setup_extension("sphinx.ext.autodoc")
    app.add_autodocumenter(Flow360Documenter)
    app.connect("autodoc-process-signature", process_autodoc_signature)
    return {"version": "1", "parallel_read_safe": True}
