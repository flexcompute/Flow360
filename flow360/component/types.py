""" Defines 'types' that various fields can be """

from typing import List, Optional, Tuple, Union

import pydantic as pd
from typing_extensions import Annotated, Literal

from ..exceptions import ValidationError

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
Size = Tuple[PositiveFloat, PositiveFloat, PositiveFloat]
MomentLengthType = Tuple[PositiveFloat, PositiveFloat, PositiveFloat]
BoundaryVelocityType = Tuple[Union[float, str], Union[float, str], Union[float, str]]
List2D = List[List[float]]

# we use tuple for fixed length lists, beacause List is a mutable, variable length structure
Coordinate = Tuple[float, float, float]


class DimensionedValue(pd.BaseModel):
    """DimensionedValue class"""

    v: float
    unit: Literal[None]


class Omega(DimensionedValue):
    """Omega type class"""

    unit: Literal["non-dim", "rad/s", "deg/s"]


class Velocity(DimensionedValue):
    """Velocity type class"""

    unit: Literal["m/s"]


class TimeStep(DimensionedValue):
    """TimeStep type class"""

    v: PositiveFloat
    unit: Literal["s", "sec", "seconds", "deg"]

    def is_second(self) -> bool:
        """is value in seconds

        Returns
        -------
        bool
            returns True if value is in seconds
        """
        return self.unit in ["s", "sec", "seconds"]


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
        _ = _PydanticValidate(c=vector)
        if not isinstance(vector, cls):
            vector = cls(vector)
        if vector == (0, 0, 0):
            raise pd.ValidationError(ValidationError(f"{cls.__name__} cannot be (0, 0, 0)"), cls)
        return vector


class Axis(Vector):
    """alias for class Vector"""
