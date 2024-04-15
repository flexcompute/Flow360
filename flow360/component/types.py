""" Defines 'types' that various fields can be """

from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import pydantic.v1 as pd
from typing_extensions import Annotated

from ..exceptions import Flow360ValidationError

# type tag default name
TYPE_TAG_STR = "_type"
COMMENTS = "comments"


def annotate_type(UnionType):
    """Annotated union type using TYPE_TAG_STR as discriminator."""
    return Annotated[UnionType, pd.Field(discriminator=TYPE_TAG_STR)]


PositiveFloat = pd.PositiveFloat
NonNegativeFloat = pd.NonNegativeFloat
PositiveInt = pd.PositiveInt
NonNegativeInt = pd.NonNegativeInt
NonNegativeAndNegOneInt = Union[pd.NonNegativeInt, Literal[-1]]
PositiveAndNegOneInt = Union[pd.PositiveInt, Literal[-1]]
Size = Tuple[PositiveFloat, PositiveFloat, PositiveFloat]
MomentLengthType = Tuple[PositiveFloat, PositiveFloat, PositiveFloat]
BoundaryVelocityType = Tuple[Union[float, str], Union[float, str], Union[float, str]]
List2D = List[List[float]]

# we use tuple for fixed length lists, beacause List is a mutable, variable length structure
Coordinate = Tuple[float, float, float]


class _PydanticValidate(pd.BaseModel):
    c: Optional[Coordinate]


class Vector(Coordinate):
    """:class: Vector

    Example
    -------
    >>> v = Vector((2, 1, 1)) # doctest: +SKIP
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, vector):
        """validator for vector"""
        if isinstance(vector, set):
            raise TypeError(
                Flow360ValidationError(f"set provided {vector}, but tuple or array expected.")
            )
        _ = _PydanticValidate(c=vector)
        if not isinstance(vector, cls):
            vector = cls(vector)
        if vector == (0, 0, 0):
            raise ValueError(Flow360ValidationError(f"{cls.__name__} cannot be (0, 0, 0)"), cls)

        return vector

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
    def validate(cls, vector):
        """validator for Axis"""
        vector = super().validate(vector)
        vector_norm = 0.0
        for element in vector:
            vector_norm += element * element
        vector_norm = np.sqrt(vector_norm)
        normalized_vector = tuple(e / vector_norm for e in vector)
        return Axis(normalized_vector)
