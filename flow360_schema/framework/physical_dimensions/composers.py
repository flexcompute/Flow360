"""
Type composition factory

Creates Pydantic-compatible types by composing dimension metadata with data type descriptors.
Uses Annotated with WithJsonSchema for clean Pydantic integration.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Annotated, Any

from pydantic import (
    BeforeValidator,
    PlainSerializer,
    SerializationInfo,
    ValidationInfo,
    WithJsonSchema,
)

from flow360_schema.framework.validation.context import is_deserializing

from .data_types import DataTypeDescriptor
from .dimension_meta import PhysicalDimensionMeta
from .unyt_utils import is_unyt_quantity
from .unyt_utils import is_zero_value as _is_zero_value
from .wire_format_display_unit import is_format_dict, parse_format_dict, serialize_to_dict

ValidationValueHook = Callable[[Any, PhysicalDimensionMeta, DataTypeDescriptor], Any]


def _create_validator(
    physical_dimension_meta: PhysicalDimensionMeta,
    data_type: DataTypeDescriptor,
    validation_value_hook: ValidationValueHook | None = None,
) -> Callable[[Any, ValidationInfo], Any]:
    """Create a validation function for the composed type."""

    def validate(value: Any, info: ValidationInfo) -> Any:
        # Active wire-format dict — convert to a unyt quantity before the rest
        # of the pipeline runs. The shape predicate and parser are imported
        # from the active wire-format module at module top.
        if is_format_dict(value):
            value = parse_format_dict(value, physical_dimension_meta)

        # Handle zero special case
        if _is_zero_value(value) and not physical_dimension_meta.allow_zero:
            raise ValueError(
                f"Zero value requires explicit unit for {physical_dimension_meta.name} "
                f"(e.g., 0K vs 0°C are different). Please provide value with unit."
            )
        # Convert to unyt first (handles unit conversion if already unyt)
        unyt_value, si_fallback = data_type.to_unyt(value, physical_dimension_meta)

        # Warn only when SI fallback was used outside deserialization
        if si_fallback and not is_deserializing():
            warnings.warn(
                f"No unit system context: bare numeric value in '{physical_dimension_meta.name}' "
                f"field is interpreted as SI ({physical_dimension_meta.si_unit}). "
                f"Use an explicit unit or a unit system context to silence this warning.",
                stacklevel=2,
            )

        value_for_validation = unyt_value.value
        if validation_value_hook is not None:
            value_for_validation = validation_value_hook(
                unyt_value,
                physical_dimension_meta,
                data_type,
            )

        # Apply validator chain (on numeric values)
        for validator_func in data_type.validators:
            validator_func(value_for_validation)

        return unyt_value

    return validate


def _create_serializer(
    data_type: DataTypeDescriptor,
    physical_dimension_meta: PhysicalDimensionMeta,
) -> Callable[[Any, SerializationInfo], Any]:
    """Create a serialization function for the composed type."""

    def serialize(value: Any, info: SerializationInfo) -> Any:
        raw = info.context.get("no_unit") if info.context else False
        if raw:
            return data_type.serialize_raw(value)
        # The dimensioned-field validator coerces every valid input to a unyt
        # quantity, so anything that reaches the serializer without units must
        # have matched a sibling union branch (e.g. ``Literal[False]`` or
        # ``Literal["inf"]``). Pass it through verbatim — wrapping it in a
        # wire-format dict would corrupt the wire shape and the value would
        # fail to re-validate against the sibling branch on load.
        if not is_unyt_quantity(value):
            return value
        return serialize_to_dict(value, data_type, physical_dimension_meta)

    return serialize


def _get_hook_cache_key(validation_value_hook: ValidationValueHook | None) -> str:
    """Build stable cache key token for optional validation hook."""
    if validation_value_hook is None:
        return ""

    module = getattr(validation_value_hook, "__module__", "")
    qualname = getattr(validation_value_hook, "__qualname__", "")
    if module and qualname:
        return f"{module}.{qualname}"

    return str(id(validation_value_hook))


def _ensure_constraint_compatibility(
    physical_dimension_meta: PhysicalDimensionMeta,
    data_type: DataTypeDescriptor,
) -> None:
    """Fail loudly when a data type is not allowed by a physical dimension."""
    if data_type.constraint_kind in physical_dimension_meta.supported_constraint_kinds:
        return

    supported = ", ".join(kind.value for kind in physical_dimension_meta.supported_constraint_kinds)
    supported_display = supported if supported else "none"
    raise ValueError(
        "[Internal] Incompatible composed type: "
        f"data type '{data_type.name}' uses constraint kind '{data_type.constraint_kind.value}', "
        f"but physical dimension '{physical_dimension_meta.name}' only supports [{supported_display}]."
    )


# Cache for composed types to ensure type identity
_type_cache: dict[tuple[str, str, str], type] = {}


def _compose_type(
    physical_dimension_meta: PhysicalDimensionMeta,
    data_type: DataTypeDescriptor,
    validation_value_hook: ValidationValueHook | None = None,
) -> type:
    """
    Create a Pydantic-compatible type from dimension + data type using Annotated.

    Uses caching to ensure same inputs return the same type object.

    Args:
        physical_dimension_meta: Physical dimension metadata
        data_type: Data type descriptor
        validation_value_hook: Optional hook to customize value used by shared validators

    Returns:
        An Annotated type that can be used as Pydantic field type
    """
    _ensure_constraint_compatibility(physical_dimension_meta, data_type)

    # Create cache key
    cache_key = (
        physical_dimension_meta.name,
        data_type.name,
        _get_hook_cache_key(validation_value_hook),
    )

    if cache_key in _type_cache:
        return _type_cache[cache_key]

    # Generate JSON schema
    json_schema = data_type.build_schema(data_type.schema_type_name, physical_dimension_meta.si_unit)

    # Create validator and serializer
    validator = _create_validator(
        physical_dimension_meta,
        data_type,
        validation_value_hook,
    )
    serializer = _create_serializer(data_type, physical_dimension_meta)

    # Compose the type using Annotated
    composed_type = Annotated[
        Any,
        BeforeValidator(validator),
        PlainSerializer(serializer),
        WithJsonSchema(json_schema),
    ]

    # Cache the type
    _type_cache[cache_key] = composed_type

    return composed_type  # type: ignore[no-any-return]


__all__ = ["_compose_type"]
