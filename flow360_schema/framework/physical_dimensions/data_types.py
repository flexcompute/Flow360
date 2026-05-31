"""
Data type descriptors

Defines data type behaviors (shape, constraints, schema generation)
independent of physical dimension. Used by composers.py to create composed types.

Each descriptor carries its own compose-logic callbacks (to_unyt, serialize_raw,
serialize_si, build_schema) so that composers.py can dispatch polymorphically
instead of branching on ShapeType.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .constraint_kinds import ConstraintKind
from .serializers import (
    serialize_array,
    serialize_raw_array,
    serialize_raw_scalar,
    serialize_raw_vector2,
    serialize_raw_vector3,
    serialize_scalar,
    serialize_vector2,
    serialize_vector3,
)
from .unyt_adapter import to_unyt_array_with_fallback_info, to_unyt_scalar_with_fallback_info
from .validators import (
    array_shape,
    non_negative,
    non_negative_array,
    non_null_vector3,
    positive,
    positive_array,
    positive_strictly_increasing,
    positive_vector,
    strictly_increasing,
    vector2_shape,
    vector3_shape,
)
from .wire_format_units import generate_array_schema, generate_schema


class ShapeType(Enum):
    """Shape type for determining unyt conversion strategy."""

    SCALAR = "scalar"
    VECTOR3 = "vector3"
    VECTOR2 = "vector2"
    ARRAY = "array"


@dataclass(frozen=True)
class DataTypeDescriptor:
    """
    Describes a data type's behavior independent of physical dimension.

    Attributes:
        name: Type name suffix (e.g., "PositiveFloat64")
        shape: Shape type identifier (scalar, vector3, etc.)
        validators: Tuple of validator functions to apply
        schema_type_name: Common-schema type name for JSON schema $ref
        constraint_kind: Whether this descriptor applies value-range constraints
        to_unyt: Converts raw value + dimension meta to (unyt_quantity, si_fallback_used)
        serialize_raw: Extracts numeric value without unit conversion (for no_unit context)
        serialize_si: Converts unyt value to plain numbers in SI units
        build_schema: Generates JSON Schema dict from (schema_type_name, si_unit)
    """

    name: str
    shape: ShapeType
    validators: tuple[Callable[[Any], Any], ...]  # Use tuple for hashability
    schema_type_name: str  # e.g., "Float64", "Vector3Json"
    constraint_kind: ConstraintKind
    to_unyt: Callable[[Any, Any], tuple[Any, bool]]
    serialize_raw: Callable[[Any], Any]
    serialize_si: Callable[[Any, Any], Any]
    build_schema: Callable[[str, str], dict[str, Any]]


# ============================================================================
# Scalar Types
# ============================================================================

ScalarFloat64 = DataTypeDescriptor(
    name="Float64",
    shape=ShapeType.SCALAR,
    validators=(),
    schema_type_name="Float64",
    constraint_kind=ConstraintKind.NO_RANGE,
    to_unyt=to_unyt_scalar_with_fallback_info,
    serialize_raw=serialize_raw_scalar,
    serialize_si=serialize_scalar,
    build_schema=generate_schema,
)

PositiveScalar = DataTypeDescriptor(
    name="PositiveFloat64",
    shape=ShapeType.SCALAR,
    validators=(positive,),
    schema_type_name="PositiveFloat64",
    constraint_kind=ConstraintKind.RANGE,
    to_unyt=to_unyt_scalar_with_fallback_info,
    serialize_raw=serialize_raw_scalar,
    serialize_si=serialize_scalar,
    build_schema=generate_schema,
)

NonNegativeScalar = DataTypeDescriptor(
    name="NonNegativeFloat64",
    shape=ShapeType.SCALAR,
    validators=(non_negative,),
    schema_type_name="NonNegativeFloat64",
    constraint_kind=ConstraintKind.RANGE,
    to_unyt=to_unyt_scalar_with_fallback_info,
    serialize_raw=serialize_raw_scalar,
    serialize_si=serialize_scalar,
    build_schema=generate_schema,
)


# ============================================================================
# Vector3 Types
# ============================================================================

Vector3Type = DataTypeDescriptor(
    name="Vector3",
    shape=ShapeType.VECTOR3,
    validators=(vector3_shape,),
    schema_type_name="Vector3Json",
    constraint_kind=ConstraintKind.NO_RANGE,
    to_unyt=to_unyt_array_with_fallback_info,
    serialize_raw=serialize_raw_vector3,
    serialize_si=serialize_vector3,
    build_schema=generate_schema,
)

PositiveVector3Type = DataTypeDescriptor(
    name="PositiveVector3",
    shape=ShapeType.VECTOR3,
    validators=(vector3_shape, positive_vector),
    schema_type_name="PositiveVector3Json",
    constraint_kind=ConstraintKind.RANGE,
    to_unyt=to_unyt_array_with_fallback_info,
    serialize_raw=serialize_raw_vector3,
    serialize_si=serialize_vector3,
    build_schema=generate_schema,
)

NonNullVector3Type = DataTypeDescriptor(
    name="NonNullVector3",
    shape=ShapeType.VECTOR3,
    validators=(vector3_shape, non_null_vector3),
    schema_type_name="NonNullVector3Json",
    constraint_kind=ConstraintKind.RANGE,
    to_unyt=to_unyt_array_with_fallback_info,
    serialize_raw=serialize_raw_vector3,
    serialize_si=serialize_vector3,
    build_schema=generate_schema,
)


# ============================================================================
# Vector2 Types
# ============================================================================

Vector2Type = DataTypeDescriptor(
    name="Vector2",
    shape=ShapeType.VECTOR2,
    validators=(vector2_shape,),
    schema_type_name="Vector2Json",
    constraint_kind=ConstraintKind.NO_RANGE,
    to_unyt=to_unyt_array_with_fallback_info,
    serialize_raw=serialize_raw_vector2,
    serialize_si=serialize_vector2,
    build_schema=generate_schema,
)

StrictlyIncreasingVector2Type = DataTypeDescriptor(
    name="StrictlyIncreasingVector2",
    shape=ShapeType.VECTOR2,
    validators=(vector2_shape, strictly_increasing),
    schema_type_name="Vector2Json",
    constraint_kind=ConstraintKind.RANGE,
    to_unyt=to_unyt_array_with_fallback_info,
    serialize_raw=serialize_raw_vector2,
    serialize_si=serialize_vector2,
    build_schema=generate_schema,
)

PositiveStrictlyIncreasingVector2Type = DataTypeDescriptor(
    name="PositiveStrictlyIncreasingVector2",
    shape=ShapeType.VECTOR2,
    validators=(vector2_shape, positive_strictly_increasing),
    schema_type_name="Vector2Json",
    constraint_kind=ConstraintKind.RANGE,
    to_unyt=to_unyt_array_with_fallback_info,
    serialize_raw=serialize_raw_vector2,
    serialize_si=serialize_vector2,
    build_schema=generate_schema,
)


# ============================================================================
# Array Types
# ============================================================================

ArrayType = DataTypeDescriptor(
    name="Array",
    shape=ShapeType.ARRAY,
    validators=(array_shape,),
    schema_type_name="Float64",
    constraint_kind=ConstraintKind.NO_RANGE,
    to_unyt=to_unyt_array_with_fallback_info,
    serialize_raw=serialize_raw_array,
    serialize_si=serialize_array,
    build_schema=generate_array_schema,
)

PositiveArrayType = DataTypeDescriptor(
    name="PositiveArray",
    shape=ShapeType.ARRAY,
    validators=(array_shape, positive_array),
    schema_type_name="PositiveFloat64",
    constraint_kind=ConstraintKind.RANGE,
    to_unyt=to_unyt_array_with_fallback_info,
    serialize_raw=serialize_raw_array,
    serialize_si=serialize_array,
    build_schema=generate_array_schema,
)

NonNegativeArrayType = DataTypeDescriptor(
    name="NonNegativeArray",
    shape=ShapeType.ARRAY,
    validators=(array_shape, non_negative_array),
    schema_type_name="NonNegativeFloat64",
    constraint_kind=ConstraintKind.RANGE,
    to_unyt=to_unyt_array_with_fallback_info,
    serialize_raw=serialize_raw_array,
    serialize_si=serialize_array,
    build_schema=generate_array_schema,
)

# Registry of all descriptors, keyed by composed-type name suffix.
# Auto-collected from all DataTypeDescriptor instances in this module.
DESCRIPTORS_BY_NAME: dict[str, DataTypeDescriptor] = {
    obj.name: obj for obj in vars().values() if isinstance(obj, DataTypeDescriptor)
}


__all__ = [
    "ConstraintKind",
    "DataTypeDescriptor",
    "DESCRIPTORS_BY_NAME",
    "ShapeType",
    # Scalars
    "ScalarFloat64",
    "PositiveScalar",
    "NonNegativeScalar",
    # Vector3
    "Vector3Type",
    "PositiveVector3Type",
    "NonNullVector3Type",
    # Vector2
    "Vector2Type",
    "StrictlyIncreasingVector2Type",
    "PositiveStrictlyIncreasingVector2Type",
    # Array
    "ArrayType",
    "PositiveArrayType",
    "NonNegativeArrayType",
]
