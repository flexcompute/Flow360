"""Constraint kind definitions for composed primitive types."""

from enum import Enum


class ConstraintKind(Enum):
    """Constraint categories used in dimension compatibility checks."""

    NO_RANGE = "no_range"
    RANGE = "range"


__all__ = ["ConstraintKind"]
