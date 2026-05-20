"""Utility functions for Flow360BaseModel: unit conversion and preprocessing."""

from __future__ import annotations

from typing import Any


def need_conversion(value: Any) -> bool:
    """Check if a value carries physical units and needs unit conversion."""
    return hasattr(value, "units")


def _preprocess_any_model(
    model: Any,
    *,
    params: Any = None,
    exclude: list[str],
    required_by: list[str],
    flow360_unit_system: Any = None,
) -> Any:
    """Preprocess a Flow360BaseModel, converting dimensioned fields to base units.

    Calls model.preprocess() if available (for subclasses that override it),
    otherwise does manual nondimensionalization + recursive descent.
    """
    if hasattr(model, "preprocess"):
        return model.preprocess(
            params=params,
            exclude=exclude,
            required_by=required_by,
            flow360_unit_system=flow360_unit_system,
        )
    # Fallback for models without preprocess (should not normally happen)
    from .base_model import Flow360BaseModel  # noqa: PLC0415 — avoid circular import

    solver_values = {}
    for prop, val in model.__dict__.items():
        if prop in exclude:
            solver_values[prop] = val
        elif isinstance(val, Flow360BaseModel):
            solver_values[prop] = _preprocess_any_model(
                val,
                params=params,
                exclude=exclude,
                required_by=[*required_by, prop],
                flow360_unit_system=flow360_unit_system,
            )
        elif need_conversion(val):
            solver_values[prop] = val.in_base(flow360_unit_system)
        elif isinstance(val, (list, dict)):
            solver_values[prop] = _preprocess_nested(val, [*required_by, prop], params, exclude, flow360_unit_system)
        else:
            solver_values[prop] = val
    return model.__class__(**solver_values)


def _preprocess_nested(
    value: Any,
    required_by: list[str],
    params: Any,
    exclude: list[str],
    flow360_unit_system: Any,
) -> Any:
    """Recursively convert dimensioned values inside lists, dicts, and models."""
    from .base_model import Flow360BaseModel  # noqa: PLC0415 — avoid circular import

    if isinstance(value, list):
        return [
            _preprocess_nested(item, required_by + [f"{i}"], params, exclude, flow360_unit_system)
            for i, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            k: _preprocess_nested(v, required_by + [f"{k}"], params, exclude, flow360_unit_system)
            for k, v in value.items()
        }
    if isinstance(value, Flow360BaseModel):
        return _preprocess_any_model(
            value,
            params=params,
            required_by=required_by,
            exclude=exclude,
            flow360_unit_system=flow360_unit_system,
        )
    if need_conversion(value):
        return value.in_base(flow360_unit_system)
    return value
