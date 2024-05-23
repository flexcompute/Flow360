from typing import Optional

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.models.material import Air, MaterialType
from flow360.component.simulation.models.solver_numerics import (
    NavierStokesSolver,
    TransitionModelSolver,
    TurbulenceModelSolverType,
)


class VolumeModelBase(Flow360BaseModel):
    """
    Base class for equation models

    """

    volumes: EntityList[Union[GenericVolume, str]] = pd.Field()
    material: MaterialType = pd.Field()
    initial_conditions: Optional[dict] = pd.Field(None)


class FluidDynamicsAddOnBase(Flow360BaseModel):
    volumes: EntityList[Union[GenericVolume, str]] = pd.Field()


class FluidDynamics(VolumeModelBase):
    """
    General FluidDynamics volume model that contains all the common fields every fluid dynamics zone should have.
    """

    navier_stokes_solver: Optional[NavierStokesSolver] = pd.Field(None)
    turbulence_model_solver: Optional[TurbulenceModelSolverType] = pd.Field(None)
    transition_model_solver: Optional[TransitionModelSolver] = pd.Field(None)

    material: MaterialTypes = pd.Field(Air())


class HeatTransfer(VolumeModelBase):
    """
    General HeatTransfer volume model that contains all the common fields every heat transfer zone should have.
    """

    heat_equation_solver: Optional[components.HeatEquationSolver] = pd.Field(None)
    volumetric_heat_source: Union[NonNegativeFloat, StrictStr] = pd.Field(0)


class ActuatorDisk(FluidDynamicsAddOnBase):
    """Same as Flow360Param ActuatorDisks.
    Note that `center`, `axis_thrust`, `thickness` can be acquired from `entity` so they are not required anymore.
    """

    volumes: Optional[EntityList[Cylinder]] = pd.Field(None)

    center: Optional[Coordinate] = pd.Field(None)
    thickness: Optional[PositiveFloat] = pd.Field(None)
    axis_thrust: Optional[Axis] = pd.Field(None)
    force_per_area: ForcePerArea = pd.Field()


class BETDisk(FluidDynamicsAddOnBase):
    """Same as Flow360Param BETDisk.
    Note that `center_of_rotation`, `axis_of_rotation`, `radius`, `thickness` can be acquired from `entity` so they are not required anymore.
    """

    volumes: Optional[EntityList[Cylinder]] = pd.Field(None)

    rotation_direction_rule: Literal["leftHand", "rightHand"] = pd.Field("rightHand")
    center_of_rotation: Optional[Coordinate] = pd.Field(None)
    axis_of_rotation: Optional[Axis] = pd.Field(None)
    thickness: Optional[LengthType.Positive] = pd.Field(None)
    radius: Optional[LengthType.Positive] = pd.Field(None)
    number_of_blades: pd.conint(strict=True, gt=0, le=10) = pd.Field()
    omega: AngularVelocityType.NonNegative = pd.Field()
    chord_ref: LengthType.Positive = pd.Field()
    n_loading_nodes: pd.conint(strict=True, gt=0, le=1000) = pd.Field()
    blade_line_chord: LengthType.NonNegative = pd.Field(0)
    initial_blade_direction: Optional[Axis] = pd.Field(None)
    tip_gap: Union[LengthType.NonNegative, Literal["inf"]] = pd.Field("inf")
    mach_numbers: List[NonNegativeFloat] = pd.Field()
    reynolds_numbers: List[PositiveFloat] = pd.Field()
    alphas: List[float] = pd.Field()
    twists: List[BETDiskTwist] = pd.Field()
    chords: List[BETDiskChord] = pd.Field()
    sectional_polars: List[BETDiskSectionalPolar] = pd.Field()
    sectional_radiuses: List[float] = pd.Field()


class Rotation(FluidDynamicsAddOnBase):
    """Similar to Flow360Param ReferenceFrame.
    Note that `center`, `axis` can be acquired from `entity` so they are not required anymore.
    Note: Should use the unit system to convert degree or degree per second to radian and radian per second
    """

    # (AKA omega) In conflict with `rotation_per_second`.
    angular_velocity: Optional[Union[float, pd.StrictStr]] = pd.Field(None)
    # (AKA theta) Must be expression otherwise zone become static.
    rotation_angle_radians: Optional[pd.StrictStr] = pd.Field(None)
    parent_volume_name: Optional[VolumeEntityTypes] = pd.Field(None)
    center: Optional[LengthType.Point] = pd.Field(None)
    axis: Optional[Axis] = pd.Field(None)


class PorousMedium(FluidDynamicsAddOnBase):
    """Constains Flow360Param PorousMediumBox and PorousMediumVolumeZone"""

    volumes: EntityList[Volume, Box, str] = pd.Field()

    darcy_coefficient: InverseAreaType.Point = pd.Field()
    forchheimer_coefficient: InverseLengthType.Point = pd.Field()
    volumetric_heat_source: Optional[HeatSourceType] = pd.Field(None)
    # box specific
    axes: Optional[List[Axis]] = pd.Field(None)
    center: Optional[LengthType.Point] = pd.Field(None)
    lengths: Optional[LengthType.Moment] = pd.Field(None)
    windowing_lengths: Optional[Size] = pd.Field(None)


VolumeModelsTypes = Union[
    FluidDynamics,
    ActuatorDisk,
    BETDisk,
    Rotation,
    PorousMedium,
    HeatTransfer,
]
