"""
Temporary file for new flow360 pydantic models
"""
from typing import Dict, List, Literal, Optional, Union

import pydantic as pd

from ..types import (
    Axis,
    Coordinate,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    Vector,
)
from .params_base import Flow360BaseModel


class InitialCondition(Flow360BaseModel):
    """:class:`InitialCondition` class"""

    type: str


class FreestreamInitialCondition(InitialCondition):
    """:class:`FreestreamInitialCondition` class"""

    type: Literal["freestream"] = pd.Field("freestream", const=True)


class ExpressionInitialCondition(InitialCondition):
    """:class:`ExpressionInitialCondition` class"""

    type: Literal["expression"] = pd.Field("expression", const=True)
    rho: str = pd.Field()
    u: str = pd.Field()
    v: str = pd.Field()
    w: str = pd.Field()
    p: str = pd.Field()


InitialConditions = Union[FreestreamInitialCondition, ExpressionInitialCondition]


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

    lift_coeffs: Optional[List[List[List[float]]]] = pd.Field(
        alias="liftCoeffs", displayed="Lift coefficients"
    )
    drag_coeffs: Optional[List[List[List[float]]]] = pd.Field(
        alias="dragCoeffs", displayed="Drag coefficients"
    )


class BETDisk(Flow360BaseModel):
    """:class:`BETDisk` class"""

    rotation_direction_rule: Optional[Union[Literal["leftHand", "rightHand"]]] = pd.Field(
        alias="rotationDirectionRule", displayed="Rotation direction"
    )
    center_of_rotation: Coordinate = pd.Field(
        alias="centerOfRotation", displayed="Center of rotation"
    )
    axis_of_rotation: Axis = pd.Field(alias="axisOfRotation", displayed="Axis of rotation")
    number_of_blades: PositiveInt = pd.Field(alias="numberOfBlades", displayed="Number of blades")
    radius: PositiveFloat = pd.Field()
    omega: float = pd.Field()
    chord_ref: PositiveFloat = pd.Field(alias="chordRef", displayed="Reference chord")
    thickness: PositiveFloat = pd.Field(alias="thickness")
    n_loading_nodes: PositiveInt = pd.Field(alias="nLoadingNodes", displayed="Loading nodes")
    blade_line_chord: Optional[PositiveFloat] = pd.Field(
        alias="bladeLineChord", displayed="Blade line chord"
    )
    initial_blade_direction: Optional[Coordinate] = pd.Field(
        alias="initialBladeDirection", displayed="Initial blade direction"
    )
    tip_gap: Optional[Union[NonNegativeFloat, Literal["inf"]]] = pd.Field(
        alias="tipGap", displayed="Tip gap"
    )
    mach_numbers: List[float] = pd.Field(alias="MachNumbers", displayed="Mach numbers")
    reynolds_numbers: List[PositiveFloat] = pd.Field(
        alias="ReynoldsNumbers", displayed="Reynolds numbers"
    )
    alphas: List[float] = pd.Field()
    twists: List[BETDiskTwist] = pd.Field()
    chords: List[BETDiskChord] = pd.Field()
    sectional_polars: List[BETDiskSectionalPolar] = pd.Field(
        alias="sectionalPolars", displayed="Sectional polars"
    )
    sectional_radiuses: List[float] = pd.Field(
        alias="sectionalRadiuses", displayed="Sectional radiuses"
    )


class PorousMediumVolumeZone(Flow360BaseModel):
    """:class:`PorousMediumVolumeZone` class"""

    zone_type: Literal["box"] = pd.Field(alias="zoneType")
    center: Coordinate = pd.Field()
    lengths: Coordinate = pd.Field()
    axes: List[Coordinate] = pd.Field(min_items=2, max_items=3)
    windowing_lengths: Optional[Coordinate] = pd.Field(alias="windowingLengths")


class PorousMedium(Flow360BaseModel):
    """:class:`PorousMedium` class"""

    darcy_coefficient: Vector = pd.Field(alias="DarcyCoefficient")
    forchheimer_coefficient: Vector = pd.Field(alias="ForchheimerCoefficient")
    volume_zone: PorousMediumVolumeZone = pd.Field(alias="volumeZone")


class UserDefinedDynamic(Flow360BaseModel):
    """:class:`UserDefinedDynamic` class"""

    name: str = pd.Field(alias="dynamicsName")
    input_vars: List[str] = pd.Field(alias="inputVars")
    constants: Optional[Dict] = pd.Field()
    output_vars: Union[Dict] = pd.Field(alias="outputVars")
    state_vars_initial_value: List[str] = pd.Field(alias="stateVarsInitialValue")
    update_law: List[str] = pd.Field(alias="updateLaw")
    output_law: List[str] = pd.Field(alias="outputLaw")
    input_boundary_patches: List[str] = pd.Field(alias="inputBoundaryPatches")
    output_target_name: str = pd.Field(alias="outputTargetName")
