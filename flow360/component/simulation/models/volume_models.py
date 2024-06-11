from typing import Dict, List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.single_attribute_base import (
    SingleAttributeModel,
)
from flow360.component.simulation.models.material import (
    Air,
    FluidMaterialTypes,
    MaterialBase,
    SolidMaterialTypes,
)
from flow360.component.simulation.models.solver_numerics import (
    HeatEquationSolver,
    NavierStokesSolver,
    SpalartAllmaras,
    TransitionModelSolver,
    TurbulenceModelSolverType,
)
from flow360.component.simulation.primitives import Box, Cylinder, GenericVolume
from flow360.component.simulation.unit_system import (
    AngularVelocityType,
    HeatSourceType,
    InverseAreaType,
    InverseLengthType,
    LengthType,
)

# TODO: Warning: Pydantic V1 import
from flow360.component.types import Axis


class AngularVelocity(SingleAttributeModel):
    value: AngularVelocityType = pd.Field()


class RotationAngleDegrees(SingleAttributeModel):
    # pylint: disable=fixme
    # TODO: We have units for degrees right??
    value: pd.StrictStr = pd.Field()


class ExpressionInitialConditionBase(Flow360BaseModel):
    """:class:`ExpressionInitialCondition` class"""

    type: Literal["expression"] = pd.Field("expression", frozen=True)
    constants: Optional[Dict[str, str]] = pd.Field()


class NavierStokesInitialCondition(ExpressionInitialConditionBase):
    rho: str = pd.Field()
    u: str = pd.Field()
    v: str = pd.Field()
    w: str = pd.Field()
    p: str = pd.Field()


class NavierStokesModifiedRestartSolution(NavierStokesInitialCondition):
    type: Literal["restartManipulation"] = pd.Field("restartManipulation", frozen=True)


class HeatEquationInitialCondition(ExpressionInitialConditionBase):
    temperature: str = pd.Field()


class PDEModelBase(Flow360BaseModel):
    """
    Base class for equation models

    """

    material: MaterialBase = pd.Field()
    initial_condition: Optional[dict] = pd.Field(None)


class Fluid(PDEModelBase):
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


class Solid(PDEModelBase):
    """
    General HeatTransfer volume model that contains all the common fields every heat transfer zone should have.
    """

    entities: EntityList[GenericVolume, str] = pd.Field(alias="volumes")

    material: SolidMaterialTypes = pd.Field()

    heat_equation_solver: HeatEquationSolver = pd.Field(HeatEquationSolver())
    volumetric_heat_source: Union[HeatSourceType, pd.StrictStr] = pd.Field(0)

    initial_condition: Optional[HeatEquationInitialCondition] = pd.Field(None)


class ForcePerArea(Flow360BaseModel):
    """:class:`ForcePerArea` class for setting up force per area for Actuator Disk

    Parameters
    ----------
    radius : List[float]
        Radius of the sampled locations in grid unit

    thrust : List[float]
        Force per area in the axial direction, positive means the axial force follows the same direction as axisThrust.
        It is non-dimensional: trustPerArea[SI=N/m2]/rho_inf/C_inf^2

    circumferential : List[float]
        Force per area in the circumferential direction, positive means the circumferential force follows the same
        direction as axisThrust with the right hand rule. It is non-dimensional:
                                                                circumferentialForcePerArea[SI=N/m2]/rho_inf/C_inf^2

    Returns
    -------
    :class:`ForcePerArea`
        An instance of the component class ForcePerArea.

    Example
    -------
    >>> fpa = ForcePerArea(radius=[0, 1], thrust=[1, 1], circumferential=[1, 1]) # doctest: +SKIP

    TODO: Use dimensioned values
    """

    radius: List[float]
    thrust: List[float]
    circumferential: List[float]

    # pylint: disable=no-self-argument
    @pd.model_validator(mode="before")
    @classmethod
    def check_len(cls, values):
        radius, thrust, circumferential = (
            values.get("radius"),
            values.get("thrust"),
            values.get("circumferential"),
        )
        if len(radius) != len(thrust) or len(radius) != len(circumferential):
            raise ValueError(
                f"length of radius, thrust, circumferential must be the same, \
                but got: len(radius)={len(radius)}, \
                         len(thrust)={len(thrust)}, \
                         len(circumferential)={len(circumferential)}"
            )

        return values


class ActuatorDisk(Flow360BaseModel):
    """Same as Flow360Param ActuatorDisks.
    Note that `center`, `axis_thrust`, `thickness` can be acquired from `entity` so they are not required anymore.
    """

    entities: Optional[EntityList[Cylinder]] = pd.Field(None, alias="volumes")

    force_per_area: ForcePerArea = pd.Field()


class BETDiskTwist(Flow360BaseModel):
    """:class:`BETDiskTwist` class"""

    # TODO: Use dimensioned values, why optional?
    radius: Optional[float] = pd.Field(None)
    twist: Optional[float] = pd.Field(None)


class BETDiskChord(Flow360BaseModel):
    """:class:`BETDiskChord` class"""

    # TODO: Use dimensioned values, why optional?
    radius: Optional[float] = pd.Field(None)
    chord: Optional[float] = pd.Field(None)


class BETDiskSectionalPolar(Flow360BaseModel):
    """:class:`BETDiskSectionalPolar` class"""

    lift_coeffs: Optional[List[List[List[float]]]] = pd.Field()
    drag_coeffs: Optional[List[List[List[float]]]] = pd.Field()


class BETDisk(Flow360BaseModel):
    """Same as Flow360Param BETDisk.
    Note that `center_of_rotation`, `axis_of_rotation`, `radius`, `thickness` can be acquired from `entity` so they are not required anymore.
    """

    entities: Optional[EntityList[Cylinder]] = pd.Field(None, alias="volumes")

    rotation_direction_rule: Literal["leftHand", "rightHand"] = pd.Field("rightHand")
    number_of_blades: pd.StrictInt = pd.Field(gt=0, le=10)
    omega: AngularVelocityType.NonNegative = pd.Field()
    chord_ref: LengthType.Positive = pd.Field()
    n_loading_nodes: pd.StrictInt = pd.Field(gt=0, le=1000)
    blade_line_chord: LengthType.NonNegative = pd.Field(0)
    initial_blade_direction: Optional[Axis] = pd.Field(None)
    tip_gap: Union[LengthType.NonNegative, Literal["inf"]] = pd.Field("inf")
    mach_numbers: List[pd.NonNegativeFloat] = pd.Field()
    reynolds_numbers: List[pd.PositiveFloat] = pd.Field()
    alphas: List[float] = pd.Field()
    twists: List[BETDiskTwist] = pd.Field()
    chords: List[BETDiskChord] = pd.Field()
    sectional_polars: List[BETDiskSectionalPolar] = pd.Field()
    sectional_radiuses: List[float] = pd.Field()


class RotatingReferenceFrame(Flow360BaseModel):
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


VolumeModelTypes = Union[
    Fluid,
    Solid,
    ActuatorDisk,
    BETDisk,
    RotatingReferenceFrame,
    PorousMedium,
]
