from typing import Optional

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.models.material import (
    Air,
    FluidMaterialTypes,
    SolidMaterialTypes,
)
from flow360.component.simulation.models.solver_numerics import (
    HeatEquationSolver,
    NavierStokesSolver,
    SpalartAllmaras,
    TransitionModelSolver,
    TurbulenceModelSolverType,
)


class AngularVelocity(SingleAttributeModel):
    value: AngularVelocityType = pd.Field()


class RotationAngleDegrees(SingleAttributeModel):
    value: pd.StrictStr = pd.Field()


class ExpressionInitialConditionBase(Flow360BaseModel):
    """:class:`ExpressionInitialCondition` class"""

    type: Literal["expression"] = pd.Field("expression", frozen=True)
    constants: Optional[Dict[str, str]] = pd.Field()


class NavierStokesInitialCondition(ExpressionInitialConditionBase):
    rho: str = pd.Field(displayed="rho [non-dim]")
    u: str = pd.Field(displayed="u [non-dim]")
    v: str = pd.Field(displayed="v [non-dim]")
    w: str = pd.Field(displayed="w [non-dim]")
    p: str = pd.Field(displayed="p [non-dim]")


class NavierStokesModifiedRestartSolution(NavierStokesInitialCondition):
    type: Literal["restartManipulation"] = pd.Field("restartManipulation", frozen=True)


class HeatEquationInitialCondition(ExpressionInitialConditionBase):
    temperature: str = pd.Field(displayed="T [non-dim]")


class PDEModelBase(Flow360BaseModel):
    """
    Base class for equation models

    """

    material: MaterialType = pd.Field()
    initial_condition: Optional[dict] = pd.Field(None)


class FluidDynamics(PDEModelBase):
    """
    General FluidDynamics volume model that contains all the common fields every fluid dynamics zone should have.
    """

    navier_stokes_solver: NavierStokesSolver = pd.Field(NavierStokesSolver())
    turbulence_model_solver: TurbulenceModelSolverType = pd.Field(SpalartAllmaras())
    transition_model_solver: Optional[TransitionModelSolver] = pd.Field(None)

    material: FluidMaterialTypes = pd.Field(Air())

    initial_condition: Optional[
        Union[NavierStokesModifiedRestartSolution, NavierStokesInitialCondition]
    ] = pd.Field(None)


class HeatTransfer(PDEModelBase):
    """
    General HeatTransfer volume model that contains all the common fields every heat transfer zone should have.
    """

    entities: EntityList[GenericVolume, str] = pd.Field(alias="volumes")

    material: SolidMaterialTypes = pd.Field()

    heat_equation_solver: HeatEquationSolver = pd.Field(HeatEquationSolver())
    volumetric_heat_source: Union[HeatSourceType, pd.StrictStr] = pd.Field(0)

    initial_condition: Optional[HeatEquationInitialCondition] = pd.Field(None)


class ActuatorDisk(Flow360BaseModel):
    """Same as Flow360Param ActuatorDisks.
    Note that `center`, `axis_thrust`, `thickness` can be acquired from `entity` so they are not required anymore.
    """

    entities: Optional[EntityList[Cylinder]] = pd.Field(None, alias="volumes")

    force_per_area: ForcePerArea = pd.Field()


class BETDisk(Flow360BaseModel):
    """Same as Flow360Param BETDisk.
    Note that `center_of_rotation`, `axis_of_rotation`, `radius`, `thickness` can be acquired from `entity` so they are not required anymore.
    """

    entities: Optional[EntityList[Cylinder]] = pd.Field(None, alias="volumes")

    rotation_direction_rule: Literal["leftHand", "rightHand"] = pd.Field("rightHand")
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


class Rotation(Flow360BaseModel):
    """Similar to Flow360Param ReferenceFrame.
    Note that `center`, `axis` can be acquired from `entity` so they are not required anymore.
    Note: Should use the unit system to convert degree or degree per second to radian and radian per second
    """

    entities: EntityList[GenericVolume, Cylinder, str] = pd.Field(alias="volumes")

    rotation: Union[AngularVelocity, RotationAngleDegrees] = pd.Field()
    parent_volume_name: Optional[Union[GenericVolume, str]] = pd.Field(None)


class PorousMedium(Flow360BaseModel):
    """Constains Flow360Param PorousMediumBox and PorousMediumVolumeZone"""

    entities: Optional[EntityList[GenericVolume, Box, str]] = pd.Field(None, alias="volumes")

    darcy_coefficient: InverseAreaType.Point = pd.Field()
    forchheimer_coefficient: InverseLengthType.Point = pd.Field()
    volumetric_heat_source: Optional[Union[HeatSourceType, pd.StrictStr]] = pd.Field(None)
    # needed for GenericVolume, need to check for conflict for Box
    axes: Optional[List[Axis]] = pd.Field(None)


VolumeModelsTypes = Union[
    FluidDynamics,
    ActuatorDisk,
    BETDisk,
    Rotation,
    PorousMedium,
    HeatTransfer,
]
