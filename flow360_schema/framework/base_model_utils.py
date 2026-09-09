"""Utility functions for Flow360BaseModel: unit conversion and preprocessing."""

from __future__ import annotations

import contextlib
import types
import weakref
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, get_args, get_origin

if TYPE_CHECKING:
    from .base_model import Flow360BaseModel


def need_conversion(value: Any) -> bool:
    """Check if a value carries physical units and needs unit conversion."""
    return hasattr(value, "units")


# Which subtrees can preprocess prove inert? The only mutation preprocess performs is
# converting unit-carrying leaves (``need_conversion``), and ``extra="forbid"`` plus
# validate-on-assignment keep every value within its field's annotation, so a field
# whose annotation cannot reach a unit-carrying type is guaranteed untouched. The one
# bypass in the codebase, ``object.__setattr__`` in EntityList's after-validator,
# writes a value of the annotated type.
#
# The verdict errs toward True: a wrong True only costs speed, a wrong False would
# silently skip a unit conversion. False therefore requires proof — unknown classes,
# ``Any``, bare containers and unresolved annotations all count as True. Classes that
# override ``preprocess`` count as True as a block, since their override may do more
# than convert units.
_INERT_PRIMITIVES = (str, bytes, bool, int, float, complex, type(None))

# True verdicts always cache (they come from a real leaf or a conservative bail-out).
# A False computed while other classes are in progress rests on the assumption that
# those classes are inert, so it is only cached once nothing is pending.
#
# Weak keys: parametrizing EntityList creates a fresh class per subscription, so a
# strong-keyed cache would pin every such throwaway class forever. Static built-in
# types (str, int, ...) cannot be weak-referenced; their verdicts are a single
# issubclass check, so they are simply recomputed.
_class_verdict_cache: weakref.WeakKeyDictionary[type, bool] = weakref.WeakKeyDictionary()
_convertible_fields_cache: weakref.WeakKeyDictionary[type, frozenset[str]] = weakref.WeakKeyDictionary()


def _remember_verdict(cls: type, verdict: bool) -> bool:
    with contextlib.suppress(TypeError):
        _class_verdict_cache[cls] = verdict
    return verdict


def _any_may_carry_units(annotations: tuple[Any, ...], in_progress: set[type]) -> tuple[bool, bool]:
    definite = True
    for annotation in annotations:
        may_carry, is_definite = _annotation_may_carry_units(annotation, in_progress)
        if may_carry:
            return True, True
        definite = definite and is_definite
    return False, definite


def _annotation_may_carry_units(annotation: Any, in_progress: set[type]) -> tuple[bool, bool]:
    """Whether values of this annotation may hold something ``need_conversion`` converts.

    Returns ``(verdict, definite)``; a non-definite False leaned on an in-progress
    class assumed inert for cycle resolution and must not be cached.
    """
    if annotation is Any:
        return True, True
    origin = get_origin(annotation)
    if origin is Annotated:
        return _annotation_may_carry_units(get_args(annotation)[0], in_progress)
    if origin is Literal:
        return False, True
    if origin is Union or origin is types.UnionType:
        return _any_may_carry_units(get_args(annotation), in_progress)
    if origin is not None:
        args = tuple(arg for arg in get_args(annotation) if arg is not Ellipsis)
        if not args:
            return True, True
        return _any_may_carry_units(args, in_progress)
    return _class_may_carry_units(annotation, in_progress)


def _class_may_carry_units(cls: Any, in_progress: set[type]) -> tuple[bool, bool]:
    if not isinstance(cls, type):
        return True, True
    cached = _class_verdict_cache.get(cls)
    if cached is not None:
        return cached, True
    if issubclass(cls, _INERT_PRIMITIVES) or issubclass(cls, Enum):
        return _remember_verdict(cls, False), True

    from .base_model import Flow360BaseModel  # noqa: PLC0415 — avoid circular import

    if issubclass(cls, Flow360BaseModel):
        if cls.preprocess is not Flow360BaseModel.preprocess:
            return _remember_verdict(cls, True), True
        if cls in in_progress:
            return False, False
        in_progress.add(cls)
        try:
            verdict, definite = _any_may_carry_units(
                tuple(field.annotation for field in cls.model_fields.values()), in_progress
            )
        finally:
            in_progress.discard(cls)
        if verdict or definite or not in_progress:
            # A self-referential False resolved at the top of its cycle is the
            # least fixpoint: cycles add no unit-carrying leaves.
            return _remember_verdict(cls, verdict), True
        return verdict, definite
    if issubclass(cls, (list, tuple, set, frozenset, dict)):
        parameterized = tuple(base for base in getattr(cls, "__orig_bases__", ()) if get_args(base))
        if not parameterized:
            return True, True
        verdict, definite = _any_may_carry_units(parameterized, in_progress)
        if definite:
            _remember_verdict(cls, verdict)
        return verdict, definite
    return _remember_verdict(cls, True), True


def convertible_field_names(model_cls: type[Flow360BaseModel]) -> frozenset[str]:
    """Fields of ``model_cls`` whose subtree may hold a unit-carrying value.

    Fields outside this set cannot be changed by ``preprocess``, so it skips them —
    on catalog-heavy models (an entity_info holding 180k surfaces) that is the
    difference between walking every entity and not entering the field at all.
    """
    cached = _convertible_fields_cache.get(model_cls)
    if cached is None:
        cached = frozenset(
            name
            for name, field in model_cls.model_fields.items()
            if _annotation_may_carry_units(field.annotation, set())[0]
        )
        _convertible_fields_cache[model_cls] = cached
    return cached


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
    changed = False
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
        changed = changed or solver_values[prop] is not val
    if not changed:
        return model
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
        items = [
            _preprocess_nested(item, required_by + [f"{i}"], params, exclude, flow360_unit_system)
            for i, item in enumerate(value)
        ]
        if all(new is old for new, old in zip(items, value, strict=True)):
            return value
        return items
    if isinstance(value, dict):
        result = {
            k: _preprocess_nested(v, required_by + [f"{k}"], params, exclude, flow360_unit_system)
            for k, v in value.items()
        }
        if all(new is old for new, old in zip(result.values(), value.values(), strict=True)):
            return value
        return result
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
