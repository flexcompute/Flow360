from typing import List, Literal, Tuple, Union

import pydantic as pd

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
