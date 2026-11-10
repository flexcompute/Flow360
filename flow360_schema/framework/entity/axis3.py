"""Standalone dimensionless three-component axis type."""

import contextlib
import math
from collections.abc import Generator
from typing import Any

import unyt as u
from pydantic import GetJsonSchemaHandler
from pydantic_core import CoreSchema, core_schema

from flow360_schema.framework.physical_dimensions.validators import non_null_vector3, vector3_shape

Axis3Components = tuple[float, float, float]


class Axis3(Axis3Components):
    """A raw, nonzero, dimensionless three-component axis.

    This standalone type lets ``ValueOrExpression`` use its standard number or expression
    envelope instead of the legacy axis wire format, which stores components directly.
    It preserves the submitted magnitude because expressions cannot be normalized before
    evaluation and eager normalization during validation would also discard the original input.
    Consumers must call :meth:`normalized` after evaluating any expression.
    """

    @classmethod
    def __get_validators__(cls) -> Generator[Any, None, None]:
        yield cls.validate

    @classmethod
    def __get_pydantic_core_schema__(cls, *args: Any, **kwargs: Any) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: CoreSchema, handler: GetJsonSchemaHandler) -> dict[str, Any]:
        return {
            "properties": {
                "value": {
                    "type": "array",
                    "items": {"type": "number"},
                    "strictType": {"type": "vector3"},
                }
            }
        }

    @staticmethod
    def _strip_dimensionless_units(value: Any) -> Any:
        if not isinstance(value, (u.unyt_array, u.unyt_quantity)):
            return value
        if value.units.dimensions != u.dimensionless.dimensions:
            raise ValueError("Axis3 must be dimensionless.")
        return value.value

    @classmethod
    def validate(cls, vector: Any) -> "Axis3":
        """Validate a dimensionless three-vector without normalizing it."""
        vector = cls._strip_dimensionless_units(vector)
        with contextlib.suppress(TypeError):
            vector = tuple(cls._strip_dimensionless_units(component) for component in vector)
        components = vector3_shape(vector)
        non_null_vector3(components)
        return cls(components)

    def normalized(self) -> "Axis3":
        """Return the corresponding unit direction."""
        vector_norm = math.hypot(*self)
        return Axis3(tuple(component / vector_norm for component in self))


__all__ = ["Axis3"]
