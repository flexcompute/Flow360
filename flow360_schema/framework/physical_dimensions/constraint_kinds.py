"""Constraint kind definitions for composed primitive types."""

from enum import Enum


class ConstraintKind(Enum):
    """
    Whether a composed type declares a value-range constraint. Used only for dimension
    compatibility checks (which type variants a given physical dimension may host).

    This is a static description of the type definition, orthogonal to two things it is
    easy to confuse it with:

    - Finiteness (is NaN/Inf allowed?). A physical quantity is never legitimately NaN/Inf
      regardless of its constraint kind, so finiteness is enforced independently of this enum.
    - Boundedness (are both endpoints finite?). ``positive``/``non_negative`` are RANGE yet
      unbounded above (``+Inf`` satisfies them), so RANGE does not imply bounded.
    """

    #: The type declares no value-range constraint at all (e.g. an unconstrained component).
    NO_RANGE = "no_range"
    #: The type declares a value-range constraint (e.g. positive, non-negative, a bounded interval).
    RANGE = "range"


__all__ = ["ConstraintKind"]
