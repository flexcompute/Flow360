"""Defines 'types' that various fields can be"""

from typing import Annotated, List, Tuple

from flow360_schema.framework.entity.geometric_types import Axis, Coordinate, Vector
from pydantic import Field

# type tag default name
TYPE_TAG_STR = "_type"
COMMENTS = "comments"

List2D = List[List[float]]

Int8 = Annotated[int, Field(ge=0, le=255)]
Color = Tuple[Int8, Int8, Int8]

__all__ = [
    "Axis",
    "Coordinate",
    "Vector",
    "TYPE_TAG_STR",
    "COMMENTS",
    "List2D",
    "Int8",
    "Color",
]
