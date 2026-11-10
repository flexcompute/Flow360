"""Client-side acknowledgment of blocking pre-submission warnings.

Some pre-flight guardrails block a job submission unless the user consciously
acknowledges the issue. Acknowledgment is purely client-side and scoped to a
``with`` block, so it never travels with the serialized params and never affects
pipeline / cloud validation:

    with fl.warning_bypass("potential_length_scale_mismatch"):
        for case in cases:
            project.run_case(...)
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Generator
from typing import Literal

# Ids of blocking pre-submission warnings a caller may consciously acknowledge.
# Declared as a Literal so IDEs auto-suggest the available options at call sites.
BypassableWarning = Literal["potential_length_scale_mismatch"]

LENGTH_SCALE_MISMATCH: BypassableWarning = "potential_length_scale_mismatch"

_warning_bypass_ctx: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "warning_bypass", default=frozenset()
)


@contextlib.contextmanager
def warning_bypass(
    warning_ids: BypassableWarning | list[BypassableWarning],
) -> Generator[None, None, None]:
    """Acknowledge specific blocking pre-submission warnings within a scope.

    Accepts a single warning id or a list. Inside this context, submissions that would
    otherwise be blocked by the listed warnings proceed. Nesting is additive: inner scopes
    union their ids with the outer set.
    """
    ids = [warning_ids] if isinstance(warning_ids, str) else warning_ids
    token = _warning_bypass_ctx.set(_warning_bypass_ctx.get() | frozenset(ids))
    try:
        yield
    finally:
        _warning_bypass_ctx.reset(token)


def is_warning_bypassed(warning_id: BypassableWarning) -> bool:
    """Return whether the given warning id is acknowledged in the active scope."""
    return warning_id in _warning_bypass_ctx.get()
