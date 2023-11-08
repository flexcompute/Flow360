from enum import Enum
from typing import Optional, Union, Literal, List, Any, Dict

from .params_base import Flow360BaseModel

import pydantic as pd

from ..types import Coordinate, Axis, PositiveFloat, PositiveInt, NonNegativeFloat


class InitialCondition(Flow360BaseModel):
    """:class:`InitialCondition` class"""
    type: str


class FreestreamInitialCondition(InitialCondition):
    """:class:`FreestreamInitialCondition` class"""
    type = pd.Field("Freestream", const=True)


class ExpressionInitialCondition(InitialCondition):
    """:class:`ExpressionInitialCondition` class"""
    type = pd.Field("Expression", const=True)
    rho: str = pd.Field()
    u: str = pd.Field()
    v: str = pd.Field()
    w: str = pd.Field()
    p: str = pd.Field()


class RotationDirectionRule(str, Enum):
    """:class:`RotationDirectionRule` class"""
    LeftHand = "leftHand",
    RightHand = "rightHand"


class BETDiskTwist(Flow360BaseModel):
    """:class:`BETDiskTwist` class"""
    radius: Optional[float] = pd.Field()
    twist: Optional[float] = pd.Field()


class BETDiskChord(Flow360BaseModel):
    """:class:`BETDiskChord` class"""
    radius: Optional[float] = pd.Field()
    chord: Optional[float] = pd.Field()


class BETDiskSectionalPolar(Flow360BaseModel):
    """:class:`BETDiskSectionalPolar` class"""
    lift_coeffs: Optional[List[Coordinate]] = pd.Field(alias="liftCoeffs")
    drag_coeffs: Optional[List[Coordinate]] = pd.Field(alias="dragCoeffs")


class BETDisk(Flow360BaseModel):
    """:class:`BETDisk` class"""
    rotation_direction_rule: Optional[RotationDirectionRule] = pd.Field(alias="rotationDirectionRule")
    center_of_rotation: Coordinate = pd.Field(alias="centerOfRotation")
    axis_of_rotation: Axis = pd.Field(alias="axisOfRotation")
    number_of_blades: PositiveInt = pd.Field(alias="numberOfBlades")
    radius: PositiveFloat = pd.Field()
    omega: float = pd.Field()
    chord_ref: PositiveFloat = pd.Field(alias="chordRef")
    thickness: PositiveFloat = pd.Field(alias="thickness")
    n_loading_nodes: PositiveInt = pd.Field(alias="nLoadingNodes")
    blade_line_chord: Optional[PositiveFloat] = pd.Field(alias="bladeLineChord")
    initial_blade_direction: Optional[Coordinate] = pd.Field(alias="initialBladeDirection")
    tip_gap: Optional[Union[NonNegativeFloat, Literal["inf"]]] = pd.Field(alias="tipGap")
    mach_numbers: List[float] = pd.Field(alias="MachNumbers")
    reynolds_numbers: List[PositiveFloat] = pd.Field(alias="ReynoldsNumbers")
    alphas: List[float] = pd.Field()
    volume_name: Optional[str] = pd.Field(alias="volumeName")
    twists: List[BETDiskTwist] = pd.Field()
    chords: List[BETDiskChord] = pd.Field()
    sectional_polars: List[BETDiskSectionalPolar] = pd.Field(alias="sectionalPolars")
    sectional_radiuses: List[float] = pd.Field(alias="sectionalRadiuses")


class PorousMediumVolumeZone(Flow360BaseModel):
    """:class:`PorousMediumVolumeZone` class"""
    zone_type: str = pd.Field(alias="zoneType")
    center: Coordinate = pd.Field()
    lengths: Coordinate = pd.Field()
    axes: List[Coordinate] = pd.Field(min_items=2, max_items=3)
    windowing_lengths: Optional[List[Coordinate]] = pd.Field()


class PorousMedium(Flow360BaseModel):
    """:class:`PorousMedium` class"""
    darcy_coefficient: Coordinate = pd.Field(alias="DarcyCoefficient")
    forchheimer_coefficient: Coordinate = pd.Field(alias="ForchheimerCoefficient")
    volume_zone: PorousMediumVolumeZone = pd.Field(alias="volumeZone")


class UserDefinedDynamic(Flow360BaseModel):
    """:class:`UserDefinedDynamic` class"""
    dynamics_name: str = pd.Field(alias="dynamicsName")
    input_vars: List[str] = pd.Field(alias="inputVars")
    constants: Optional[Dict] = pd.Field()
    output_vars: Union[List[str], Dict] = pd.Field()
    state_vars_initial_value: List[str] = pd.Field(alias="stateVarsInitialValue")
    update_law: List[str] = pd.Field(alias="updateLaw")
    output_law: List[str] = pd.Field(alias="outputLaw")
    input_boundary_patches: List[str] = pd.Field(alias="inputBoundaryPatches")
    output_target_name: str = pd.Field(alias="outputTargetName")
