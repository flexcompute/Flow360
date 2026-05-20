"""Geometric vector and axis types for entity definitions."""

import math
from collections.abc import Generator
from typing import Any

from pydantic import GetJsonSchemaHandler
from pydantic_core import CoreSchema, core_schema

# Fixed-length 3D coordinate tuple
Coordinate = tuple[float, float, float]


def _validate_coordinate(value: object) -> tuple[float, float, float]:
    """Validate that value is a 3-element numeric sequence convertible to Coordinate."""
    if isinstance(value, set):
        raise TypeError(f"set provided {value}, but tuple or array expected.")
    if not hasattr(value, "__len__") or not hasattr(value, "__getitem__"):
        raise TypeError(f"Expected a sequence, got {type(value).__name__}")
    if len(value) != 3:  # type: ignore[arg-type]
        raise ValueError(f"Expected exactly 3 elements, got {len(value)}")  # type: ignore[arg-type]
    try:
        return (float(value[0]), float(value[1]), float(value[2]))  # type: ignore[index]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"All elements must be numeric, got {value}") from exc


class Vector(Coordinate):
    """:class: Vector

    Example
    -------
    >>> v = Vector((2, 1, 1)) # doctest: +SKIP
    """

    @classmethod
    def __get_validators__(cls) -> Generator[Any, None, None]:
        yield cls.validate

    @classmethod
    def __get_pydantic_core_schema__(cls, *args: Any, **kwargs: Any) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: CoreSchema, handler: GetJsonSchemaHandler) -> dict[str, Any]:
        schema = {"properties": {"value": {"type": "array"}}}
        schema["properties"]["value"]["items"] = {"type": "number"}
        schema["properties"]["value"]["strictType"] = {"type": "vector3"}

        return schema

    @classmethod
    def validate(cls, vector: Any) -> "Vector":
        """Validator for vector."""
        _validate_coordinate(vector)
        if not isinstance(vector, cls):
            vector = cls(vector)
        if vector == (0, 0, 0):
            raise ValueError(f"{cls.__name__} cannot be (0, 0, 0)")
        return vector  # type: ignore[no-any-return]

    @classmethod
    def __modify_schema__(cls, field_schema: Any, field: Any) -> None:
        new_schema = {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": [{"type": "number"}, {"type": "number"}, {"type": "number"}],
        }

        field_schema.update(new_schema)


class Axis(Vector):
    """:class: Axis (unit vector)

    Example
    -------
    >>> v = Axis((0, 0, 1)) # doctest: +SKIP
    """

    @classmethod
    def __get_validators__(cls) -> Generator[Any, None, None]:
        yield cls.validate

    @classmethod
    def __get_pydantic_core_schema__(cls, *args: Any, **kwargs: Any) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, vector: Any) -> "Axis | None":  # type: ignore[override]
        """Validator for Axis."""
        if vector is None:
            return None
        vector = super().validate(vector)
        vector_norm = math.sqrt(sum(e * e for e in vector))
        normalized_vector = tuple(e / vector_norm for e in vector)
        return Axis(normalized_vector)
