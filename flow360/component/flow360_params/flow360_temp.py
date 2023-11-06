from enum import Enum
from typing import Optional, Union, Literal

from .params_base import Flow360BaseModel

import pydantic as pd

from ..types import Coordinate, Vector, Axis, PositiveFloat, PositiveInt, NonNegativeFloat


class RotationDirectionRule(Enum):
    LeftHand = "leftHand",
    RightHand = "rightHand"

class BETDisk(Flow360BaseModel):
    rotation_direction_rule: Optional[RotationDirectionRule] = pd.Field(alias="rotationDirectionRule")
    center_of_rotation: Coordinate = pd.Field(alias="centerOfRotation")
    axis_of_rotation: Axis = pd.Field(alias="axisOfRotation")
    number_of_blades: PositiveInt = pd.Field(alias="numberOfBlades")
    radius: PositiveFloat = pd.Field()
    omega: float = pd.Field()
    chord_ref: PositiveFloat = pd.Field(alias="chordRef")
    thickness: PositiveFloat = pd.Field(alias="thickness")
    n_loading_nodes: PositiveInt = pd.Field(alias="nLoadingNodes")
    blade_line_chord: PositiveFloat = pd.Field(alias="bladeLineChord")
    initial_blade_direction: Coordinate = pd.Field(alias="initialBladeDirection")
    tip_gap: Union[NonNegativeFloat, Literal["inf"]] = pd.Field(alias="tipGap")
    