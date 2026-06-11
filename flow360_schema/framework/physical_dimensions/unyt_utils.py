"""
Unyt utility functions for type checking, unit resolution, and conversions.

This module contains the building blocks used by unyt_adapter.py.
"""

import warnings
from functools import lru_cache
from numbers import Real
from typing import Any

import numpy as np
import unyt as u
from unyt import unyt_array
from unyt.unit_registry import UnitRegistry  # type: ignore[import-untyped]

from .dimension_meta import PhysicalDimensionMeta as DimensionMeta


def _patch_unyt_registry_deepcopy() -> None:
    """Patch UnitRegistry.__deepcopy__ to preserve the _unit_system_id cache.

    unyt's UnitRegistry.__deepcopy__ creates a new instance via __init__,
    which does not copy _unit_system_id (it defaults to the class-level None).
    This forces an expensive recomputation (~40ms) on every subsequent
    Unit.__hash__() call: iterating all ~1000 LUT entries through sympy repr()
    to build an MD5 hash.

    Since deepcopy faithfully copies the LUT, the hash is guaranteed to be
    identical. Preserving the cached value is safe and avoids the recomputation.
    """
    original_deepcopy = UnitRegistry.__deepcopy__

    def _deepcopy_preserve_cache(self: UnitRegistry, memodict: dict[int, object] | None = None) -> UnitRegistry:
        new_registry = original_deepcopy(self, memodict)
        # _unit_system_id is a pure function of lut contents (MD5 hash).
        # deepcopy preserves lut faithfully, so the cached hash stays valid.
        new_registry._unit_system_id = self._unit_system_id
        return new_registry

    UnitRegistry.__deepcopy__ = _deepcopy_preserve_cache


_patch_unyt_registry_deepcopy()


def dsl_to_unyt_unit(dsl_unit: str) -> str:
    """
    Convert common-schema DSL unit string to unyt-compatible format.

    DSL format uses '^' for powers (e.g., "meter^2").
    unyt format uses '**' for powers (e.g., "meter**2").
    """
    return dsl_unit.replace("^", "**")


def unyt_to_dsl_unit(unyt_unit: str) -> str:
    """
    Convert unyt unit string to common-schema DSL format.

    unyt format uses '**' for powers (e.g., "meter**2").
    DSL format uses '^' for powers (e.g., "meter^2").
    """
    return unyt_unit.replace("**", "^")


@lru_cache(maxsize=128)
def get_si_unit(dimension_meta: DimensionMeta) -> Any:
    """Get cached unyt Unit object for a dimension's SI unit."""
    return u.Unit(dsl_to_unyt_unit(dimension_meta.si_unit))


def units_semantically_equivalent(unit_a: Any, unit_b: Any) -> bool:
    """True if two unyt units represent the same physical magnitude.

    Two units are semantically equivalent when they share the same dimensions,
    the same base scaling factor, and the same offset. Example: ``Pa`` and
    ``kg/(m*s**2)`` are equivalent (both pressure, base_value=1, base_offset=0)
    even though their string forms differ.
    """
    return bool(
        unit_a.dimensions == unit_b.dimensions
        and unit_a.base_value == unit_b.base_value
        and unit_a.base_offset == unit_b.base_offset
    )


def ensure_float64(value: Any) -> Any:
    """Enforce float64 dtype on a unyt quantity to avoid platform-dependent precision issues."""
    if value.dtype != np.float64:
        return value.astype(np.float64)
    return value


def is_numeric_scalar(value: Any) -> bool:
    """Check if value is a plain numeric scalar (excluding booleans).

    Also accepts 0-d numpy arrays containing numeric data.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        return True
    return isinstance(value, np.ndarray) and value.ndim == 0 and not is_unyt_quantity(value)


def is_zero_value(value: Any) -> bool:
    """Check if value is a pure numeric zero scalar or iterable of zeros."""
    if is_unyt_quantity(value):
        return False

    try:
        if hasattr(value, "ndim") and value.ndim == 0:
            return float(value) == 0.0
        if isinstance(value, (int, float)):
            return value == 0
        if hasattr(value, "__iter__"):
            return all(float(item) == 0.0 for item in value)
    except (TypeError, ValueError):
        pass

    return False


def coalesce_unyt_quantity_list(value: Any) -> Any:
    """Convert a list of same-unit unyt_quantity into a single unyt_array."""
    if not isinstance(value, list) or not all(is_unyt_quantity(item) for item in value):
        return value
    units = {item.units for item in value}
    if len(units) != 1:
        return value
    return unyt_array([item.value for item in value], units.pop())


def is_bare_numeric_array(value: Any) -> bool:
    """Check if value should be treated as bare numeric array input for unit fallback.

    Only accepts concrete sequences (list, tuple, numpy array) to avoid
    consuming single-use iterables like generators.
    """
    if isinstance(value, (list, tuple)):
        return all(is_numeric_scalar(item) for item in value)

    return (
        isinstance(value, np.ndarray)
        and not is_unyt_quantity(value)
        and value.ndim >= 1
        and np.issubdtype(value.dtype, np.number)
    )


def normalize_unit_for_adapter(unit: Any) -> Any:
    """Normalize a unit-system field value into the unit object expected by adapter converters."""
    # Flow360 custom units must stay as unit objects (not `.units`) to preserve arithmetic behavior.
    if hasattr(unit, "unit_name") and hasattr(unit, "dimension_type"):
        return unit
    # For unyt quantities, use their unit object directly.
    if hasattr(unit, "units"):
        return unit.units
    return unit


def get_unit_from_unit_system(dimension_meta: DimensionMeta) -> tuple[Any, bool]:
    """Get the unit for a bare number from the active unit system.

    Returns (unit, is_fallback) where is_fallback=True means no unit system
    was active and SI was used as default.

    During deserialization the active context is ignored and SI is always returned,
    because serialized data stores SI values only.
    """
    from flow360_schema.framework.validation.context import (
        is_deserializing,
        is_strict_unit_mode,
        unit_system_manager,
    )

    if is_deserializing():
        if unit_system_manager.current is not None:
            warnings.warn(
                "[Internal] Unit system context should not be used during deserialization. "
                "The active unit system context is ignored.",
                stacklevel=2,
            )
        return get_si_unit(dimension_meta), True

    if not dimension_meta.unit_system_inference:
        raise ValueError(
            f"Dimension '{dimension_meta.name}' does not support unit inference from "
            f"the active unit system. Please provide an explicit unit "
            f"(SI unit: {dimension_meta.si_unit})."
        )

    if is_strict_unit_mode():
        raise ValueError(
            f"Value does not have units matching '{dimension_meta.name}' dimension "
            f"({dimension_meta.si_unit}). An explicit unit is required."
        )

    active_unit_system = unit_system_manager.current
    if active_unit_system is not None:
        try:
            raw_unit = active_unit_system[dimension_meta.name]
        except KeyError:
            raise KeyError(
                f"Active unit system does not define dimension '{dimension_meta.name}'. "
                f"Ensure the unit system covers all required dimensions."
            ) from None
        return normalize_unit_for_adapter(raw_unit), False

    return get_si_unit(dimension_meta), True


def is_unyt_quantity(value: Any) -> bool:
    """Check if value is a unyt scalar or array. ``unyt_quantity`` is a subclass of
    ``unyt_array``, so the single isinstance check covers both."""
    return isinstance(value, unyt_array)


def is_unyt_unit(value: Any) -> bool:
    """Check if value is a raw unyt Unit object."""
    return isinstance(value, u.Unit)


def _mole_exponent(unit: Any) -> Any:
    """Net exponent of the amount-of-substance (mole) axis in a unit.

    unyt treats the mole as dimensionless, so a unit's ``.dimensions`` ignores it.
    Summing the powers of every mole symbol (``mol``, ``kmol``, ``mmol``, ...) in
    the unit expression recovers that hidden axis, independent of any SI prefix
    (``kg/mol`` and ``kg/kmol`` both have a mole exponent of ``-1``).
    """
    return sum(
        exponent for base, exponent in unit.expr.as_powers_dict().items() if getattr(base, "name", "").endswith("mol")
    )


def check_dimension(value: Any, dimension_meta: DimensionMeta) -> bool:
    """Check if a unyt quantity has the expected dimension.

    unyt has no amount-of-substance base dimension -- ``mol`` is dimensionless --
    so a plain ``.dimensions`` comparison treats ``kg`` and ``kg/mol`` as the same
    (both ``mass``). Comparing the net mole exponent recovers that hidden axis, so
    a bare mass is rejected on a molar field while equivalent molar forms
    (``g/mol``, ``kg/kmol``) pass. Dimensionally equivalent compound forms (``Pa``
    vs ``kg/(m*s**2)``) carry no mole axis and are unaffected.

    The mole exponent is read directly from the unit expression rather than by
    dividing ``value`` by the SI unit, because unyt refuses to divide offset units
    (degC/degF) and would raise instead of returning a boolean.
    """
    if not is_unyt_quantity(value):
        return False

    si_unit = get_si_unit(dimension_meta)
    if si_unit.dimensions != value.units.dimensions:
        return False

    return bool(_mole_exponent(value.units) == _mole_exponent(si_unit))
