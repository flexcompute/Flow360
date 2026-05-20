"""
Physical dimension metadata definition

This module defines the PhysicalDimensionMeta dataclass.
It does NOT depend on unyt - only stores string representations.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .constraint_kinds import ConstraintKind

_ALL_CONSTRAINT_KINDS: tuple[ConstraintKind, ...] = tuple(ConstraintKind)


@dataclass(frozen=True)
class PhysicalDimensionMeta:
    """
    Metadata for a physical dimension.

    Attributes:
        name: Physical dimension name (e.g., "length", "velocity")
        si_unit: SI unit string (e.g., "m", "m/s", "kg*m/s^2")
        allow_zero: Whether pure numeric 0 is accepted without unit
        supported_constraint_kinds: Constraint categories supported by this dimension
        validation_value_hook: Optional hook that customizes numeric value validation behavior
    """

    name: str
    si_unit: str
    allow_zero: bool = True  # Most dimensions allow zero except temperature
    unit_system_inference: bool = True  # Whether bare numbers can infer units from the active unit system
    supported_constraint_kinds: tuple[ConstraintKind, ...] = _ALL_CONSTRAINT_KINDS
    validation_value_hook: Callable[[Any, Any, Any], Any] | None = None


__all__ = ["PhysicalDimensionMeta"]
