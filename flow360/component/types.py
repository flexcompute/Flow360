""" Defines 'types' that various fields can be """

from typing import List, Optional, Tuple

import numpy as np
import pydantic.v1 as pd
from pydantic import GetJsonSchemaHandler
from pydantic_core import CoreSchema, core_schema

# type tag default name
TYPE_TAG_STR = "_type"
COMMENTS = "comments"

List2D = List[List[float]]
# we use tuple for fixed length lists, beacause List is a mutable, variable length structure
Coordinate = Tuple[float, float, float]


class Vector(Coordinate):
    """:class: Vector

    Example
    -------
    >>> v = Vector((2, 1, 1)) # doctest: +SKIP
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    # pylint: disable=unused-argument
    @classmethod
    def __get_pydantic_core_schema__(cls, *args, **kwargs) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    # pylint: disable=unused-argument
    @classmethod
    def __get_pydantic_json_schema__(cls, schema: CoreSchema, handler: GetJsonSchemaHandler):
        schema = {"properties": {"value": {"type": "array"}}}
        schema["properties"]["value"]["items"] = {"type": "number"}
        schema["properties"]["value"]["strictType"] = {"type": "vector3"}

        return schema

    @classmethod
    def validate(cls, vector):
        """validator for vector"""

        class _PydanticValidate(pd.BaseModel):
            c: Optional[Coordinate]

        if isinstance(vector, set):
            raise TypeError(f"set provided {vector}, but tuple or array expected.")
        _ = _PydanticValidate(c=vector)
        if not isinstance(vector, cls):
            vector = cls(vector)
        if vector == (0, 0, 0):
            raise ValueError(f"{cls.__name__} cannot be (0, 0, 0)")

        return vector

    # pylint: disable=unused-argument
    @classmethod
    def __modify_schema__(cls, field_schema, field):
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
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __get_pydantic_core_schema__(cls, *args, **kwargs) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, vector):
        """validator for Axis"""
        if vector is None:
            return None
        vector = super().validate(vector)
        vector_norm = np.linalg.norm(vector)
        normalized_vector = tuple(e / vector_norm for e in vector)
        return Axis(normalized_vector)
