""" Defines 'types' that various fields can be """

from typing import List, Literal, Optional, Tuple, Union

import pydantic as pd
from typing_extensions import Annotated

from ..exceptions import Flow360ValidationError

# type tag default name
TYPE_TAG_STR = "_type"
COMMENTS = "comments"


def annotate_type(UnionType):  # pylint:disable=invalid-name
    """Annotated union type using TYPE_TAG_STR as discriminator."""
    return Annotated[UnionType, pd.Field(discriminator=TYPE_TAG_STR)]


PositiveFloat = pd.PositiveFloat
NonNegativeFloat = pd.NonNegativeFloat
PositiveInt = pd.PositiveInt
NonNegativeInt = pd.NonNegativeInt
NonNegativeAndNegOneInt = Union[Literal[-1], pd.NonNegativeInt]
PositiveAndNegOneInt = Union[Literal[-1], pd.PositiveInt]
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
    >>> v = Vector((0, 0, 1)) # doctest: +SKIP
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


Axis = Vector
