"""
Base class for physical dimension type namespaces

Provides all data type variants via @classproperty.
Concrete physical dimension classes just need to set physical_dimension_meta.
"""

from typing import Any, ClassVar

from .composers import _compose_type
from .data_types import (
    ArrayType,
    NonNegativeArrayType,
    NonNegativeScalar,
    NonNullVector3Type,
    PositiveArrayType,
    PositiveScalar,
    PositiveStrictlyIncreasingVector2Type,
    PositiveVector3Type,
    ScalarFloat64,
    StrictlyIncreasingVector2Type,
    Vector2Type,
    Vector3Type,
)
from .dimension_meta import PhysicalDimensionMeta


class classproperty(property):
    """classproperty decorator - allows defining class-level properties."""

    def __get__(self, _: Any, owner_cls: type) -> Any:  # type: ignore[override]
        return self.fget(owner_cls)  # type: ignore[misc]


class PhysicalDimensionBase:
    """
    Base class for physical dimension type namespaces.

    Subclasses only need to define:
        physical_dimension_meta = PhysicalDimensionMeta(name="length", si_unit="m")

    All type variants (Float64, PositiveFloat64, Vector3, etc.) are
    automatically available via @classproperty.

    Example:
        class Length(PhysicalDimensionBase):
            physical_dimension_meta = PhysicalDimensionMeta(name="length", si_unit="m")

        # Now Length.PositiveFloat64, Length.Vector3, etc. all work
    """

    # Must be set by subclass
    physical_dimension_meta: ClassVar[PhysicalDimensionMeta | None] = None

    @classmethod
    def _meta(cls) -> PhysicalDimensionMeta:
        """Return validated metadata for this physical dimension class."""
        if cls.physical_dimension_meta is None:
            raise TypeError(f"{cls.__name__}.physical_dimension_meta must be set")
        return cls.physical_dimension_meta

    # ========================================================================
    # Scalars
    # ========================================================================

    @classproperty
    def Float64(cls) -> type:
        """Scalar with no constraints."""
        meta = cls._meta()
        return _compose_type(meta, ScalarFloat64, meta.validation_value_hook)

    @classproperty
    def PositiveFloat64(cls) -> type:
        """Scalar, must be positive (> 0)."""
        meta = cls._meta()
        return _compose_type(meta, PositiveScalar, meta.validation_value_hook)

    @classproperty
    def NonNegativeFloat64(cls) -> type:
        """Scalar, must be non-negative (>= 0)."""
        meta = cls._meta()
        return _compose_type(meta, NonNegativeScalar, meta.validation_value_hook)

    # ========================================================================
    # Vectors
    # ========================================================================

    @classproperty
    def Vector3(cls) -> type:
        """3D vector with no constraints (zero vector allowed)."""
        meta = cls._meta()
        return _compose_type(meta, Vector3Type, meta.validation_value_hook)

    @classproperty
    def PositiveVector3(cls) -> type:
        """3D vector, all components must be positive (> 0)."""
        meta = cls._meta()
        return _compose_type(meta, PositiveVector3Type, meta.validation_value_hook)

    @classproperty
    def NonNullVector3(cls) -> type:
        """3D vector with non-zero norm (for axis/direction fields)."""
        meta = cls._meta()
        return _compose_type(meta, NonNullVector3Type, meta.validation_value_hook)

    # ========================================================================
    # Vector2 (Pair, Range)
    # ========================================================================

    @classproperty
    def Vector2(cls) -> type:
        """2D vector with no constraints."""
        meta = cls._meta()
        return _compose_type(meta, Vector2Type, meta.validation_value_hook)

    @classproperty
    def StrictlyIncreasingVector2(cls) -> type:
        """2-element vector, v[0] < v[1] (for range fields)."""
        meta = cls._meta()
        return _compose_type(meta, StrictlyIncreasingVector2Type, meta.validation_value_hook)

    @classproperty
    def PositiveStrictlyIncreasingVector2(cls) -> type:
        """2-element vector, all > 0 and v[0] < v[1] (for positive range fields)."""
        meta = cls._meta()
        return _compose_type(meta, PositiveStrictlyIncreasingVector2Type, meta.validation_value_hook)

    # ========================================================================
    # Arrays (variable-length, runtime: unyt_array)
    # ========================================================================

    @classproperty
    def Array(cls) -> type:
        """Variable-length array, no constraints."""
        meta = cls._meta()
        return _compose_type(meta, ArrayType, meta.validation_value_hook)

    @classproperty
    def PositiveArray(cls) -> type:
        """Variable-length array, all elements > 0."""
        meta = cls._meta()
        return _compose_type(meta, PositiveArrayType, meta.validation_value_hook)

    @classproperty
    def NonNegativeArray(cls) -> type:
        """Variable-length array, all elements >= 0."""
        meta = cls._meta()
        return _compose_type(meta, NonNegativeArrayType, meta.validation_value_hook)


__all__ = ["PhysicalDimensionBase"]
