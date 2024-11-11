"""Volume models for the simulation framework."""

from typing import Dict, List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.expressions import StringExpression
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
    NoneSolver,
    SpalartAllmaras,
    TransitionModelSolverType,
    TurbulenceModelSolverType,
)
from flow360.component.simulation.models.validation.validation_bet_disk import (
    _check_bet_disk_3d_coefficients_in_polars,
    _check_bet_disk_alphas_in_order,
    _check_bet_disk_duplicate_chords,
    _check_bet_disk_duplicate_twists,
    _check_bet_disk_initial_blade_direction_and_blade_line_chord,
    _check_bet_disk_sectional_radius_and_polars,
)
from flow360.component.simulation.primitives import Box, Cylinder, GenericVolume
from flow360.component.simulation.unit_system import (
    AngularVelocityType,
    HeatSourceType,
    InverseAreaType,
    InverseLengthType,
    LengthType,
    PressureType,
)
from flow360.component.simulation.validation_utils import (
    _validator_append_instance_name,
)

# pylint: disable=fixme
# TODO: Warning: Pydantic V1 import
from flow360.component.types import Axis


class AngleExpression(SingleAttributeModel):
    """Angle expression for Rotation"""

    type_name: Literal["AngleExpression"] = pd.Field("AngleExpression", frozen=True)
    value: StringExpression = pd.Field()


class AngularVelocity(SingleAttributeModel):
    """Angular velocity for Rotation"""

    type_name: Literal["AngularVelocity"] = pd.Field("AngularVelocity", frozen=True)
    value: AngularVelocityType = pd.Field()


class FromUserDefinedDynamics(Flow360BaseModel):
    """Rotation is controlled by user defined dynamics"""

    type_name: Literal["FromUserDefinedDynamics"] = pd.Field("FromUserDefinedDynamics", frozen=True)


class ExpressionInitialConditionBase(Flow360BaseModel):
    """:class:`ExpressionInitialCondition` class"""

    type: Literal["expression"] = pd.Field("expression", frozen=True)
    constants: Optional[Dict[str, StringExpression]] = pd.Field(None)


# pylint: disable=missing-class-docstring
class NavierStokesInitialCondition(ExpressionInitialConditionBase):
    rho: StringExpression = pd.Field()
    u: StringExpression = pd.Field()
    v: StringExpression = pd.Field()
    w: StringExpression = pd.Field()
    p: StringExpression = pd.Field()


class NavierStokesModifiedRestartSolution(NavierStokesInitialCondition):
    type: Literal["restartManipulation"] = pd.Field("restartManipulation", frozen=True)


class HeatEquationInitialCondition(ExpressionInitialConditionBase):
    temperature: StringExpression = pd.Field()


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

    type: Literal["Fluid"] = pd.Field("Fluid", frozen=True)
    navier_stokes_solver: NavierStokesSolver = pd.Field(NavierStokesSolver())
    turbulence_model_solver: TurbulenceModelSolverType = pd.Field(SpalartAllmaras())
    transition_model_solver: TransitionModelSolverType = pd.Field(NoneSolver())

    material: FluidMaterialTypes = pd.Field(Air())

    initial_condition: Optional[
        Union[NavierStokesModifiedRestartSolution, NavierStokesInitialCondition]
    ] = pd.Field(None, discriminator="type")

    # pylint: disable=fixme
    # fixme: Add support for other initial conditions


class Solid(PDEModelBase):
    """
    General HeatTransfer volume model that contains all the common fields every heat transfer zone should have.
    """

    name: Optional[str] = pd.Field(None)
    type: Literal["Solid"] = pd.Field("Solid", frozen=True)
    entities: EntityList[GenericVolume] = pd.Field(alias="volumes")

    material: SolidMaterialTypes = pd.Field()

    heat_equation_solver: HeatEquationSolver = pd.Field(HeatEquationSolver())
    volumetric_heat_source: Union[HeatSourceType, pd.StrictStr] = pd.Field(0)

    initial_condition: Optional[HeatEquationInitialCondition] = pd.Field(None)


# pylint: disable=duplicate-code
class ForcePerArea(Flow360BaseModel):
    """:class:`ForcePerArea` class for setting up force per area for Actuator Disk

    Parameters
    ----------
    radius : LengthType.Array
        Radius of the sampled locations in grid unit

    thrust : PressureType.Array
        Force per area in the axial direction, positive means the axial force follows the same direction as axisThrust.
        It is non-dimensional: trustPerArea[SI=N/m2]/rho_inf/C_inf^2

    circumferential : PressureType.Array
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

    radius: LengthType.Array  # pylint: disable=no-member
    thrust: PressureType.Array  # pylint: disable=no-member
    circumferential: PressureType.Array  # pylint: disable=no-member

    # pylint: disable=no-self-argument, missing-function-docstring
    @pd.model_validator(mode="before")
    @classmethod
    def validate_consistent_array_length(cls, values):
        radius, thrust, circumferential = (
            values.get("radius"),
            values.get("thrust"),
            values.get("circumferential"),
        )
        if len(radius) != len(thrust) or len(radius) != len(circumferential):
            raise ValueError(
                "length of radius, thrust, circumferential must be the same, but got: "
                + f"len(radius)={len(radius)}, len(thrust)={len(thrust)}, len(circumferential)={len(circumferential)}"
            )

        return values


class ActuatorDisk(Flow360BaseModel):
    """Same as Flow360Param ActuatorDisks.
    Note that `center`, `axis_thrust`, `thickness` can be acquired from `entity` so they are not required anymore.
    """

    entities: EntityList[Cylinder] = pd.Field(alias="volumes")
    force_per_area: ForcePerArea = pd.Field()
    name: Optional[str] = pd.Field(None)
    type: Literal["ActuatorDisk"] = pd.Field("ActuatorDisk", frozen=True)


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


# pylint: disable=no-member
class BETDisk(Flow360BaseModel):
    """Same as Flow360Param BETDisk.
    Note that `center_of_rotation`, `axis_of_rotation`, `radius`, `thickness` can be acquired from `entity`
    so they are not required anymore.
    """

    name: Optional[str] = pd.Field(None)
    type: Literal["BETDisk"] = pd.Field("BETDisk", frozen=True)
    entities: EntityList[Cylinder] = pd.Field(alias="volumes")

    rotation_direction_rule: Literal["leftHand", "rightHand"] = pd.Field("rightHand")
    number_of_blades: pd.StrictInt = pd.Field(gt=0, le=10)
    omega: AngularVelocityType.NonNegative = pd.Field()
    chord_ref: LengthType.Positive = pd.Field()
    n_loading_nodes: pd.StrictInt = pd.Field(gt=0, le=1000)
    blade_line_chord: LengthType.NonNegative = pd.Field(0)
    initial_blade_direction: Optional[Axis] = pd.Field(None)
    tip_gap: Union[Literal["inf"], LengthType.NonNegative] = pd.Field("inf")
    mach_numbers: List[pd.NonNegativeFloat] = pd.Field()
    reynolds_numbers: List[pd.PositiveFloat] = pd.Field()
    alphas: List[float] = pd.Field()
    twists: List[BETDiskTwist] = pd.Field()
    chords: List[BETDiskChord] = pd.Field()
    sectional_polars: List[BETDiskSectionalPolar] = pd.Field()
    sectional_radiuses: List[float] = pd.Field()

    @pd.model_validator(mode="after")
    @_validator_append_instance_name
    def check_bet_disk_initial_blade_direction_and_blade_line_chord(self):
        """validate initial blade direction and blade line chord in BET disks"""
        return _check_bet_disk_initial_blade_direction_and_blade_line_chord(self)

    @pd.field_validator("alphas", mode="after")
    @classmethod
    @_validator_append_instance_name
    def check_bet_disk_alphas_in_order(cls, value, info: pd.ValidationInfo):
        """validate order of alphas in BET disks"""
        return _check_bet_disk_alphas_in_order(value, info)

    @pd.field_validator("chords", mode="after")
    @classmethod
    @_validator_append_instance_name
    def check_bet_disk_duplicate_chords(cls, value, info: pd.ValidationInfo):
        """validate duplicates in chords in BET disks"""
        return _check_bet_disk_duplicate_chords(value, info)

    @pd.field_validator("twists", mode="after")
    @classmethod
    @_validator_append_instance_name
    def check_bet_disk_duplicate_twists(cls, value, info: pd.ValidationInfo):
        """validate duplicates in twists in BET disks"""
        return _check_bet_disk_duplicate_twists(value, info)

    @pd.model_validator(mode="after")
    @_validator_append_instance_name
    def check_bet_disk_sectional_radius_and_polars(self):
        """validate duplicates in chords and twists in BET disks"""
        return _check_bet_disk_sectional_radius_and_polars(self)

    @pd.model_validator(mode="after")
    @_validator_append_instance_name
    def check_bet_disk_3d_coefficients_in_polars(self):
        """validate dimension of 3d coefficients in polars"""
        return _check_bet_disk_3d_coefficients_in_polars(self)


class Rotation(Flow360BaseModel):
    """Similar to Flow360Param ReferenceFrame.
    Note that `center`, `axis` can be acquired from `entity` so they are not required anymore.
    Note: Should use the unit system to convert degree or degree per second to radian and radian per second
    """

    name: Optional[str] = pd.Field(None)
    type: Literal["Rotation"] = pd.Field("Rotation", frozen=True)
    entities: EntityList[GenericVolume, Cylinder] = pd.Field(alias="volumes")

    # TODO: Add test for each of the spec specification.
    spec: Union[AngleExpression, FromUserDefinedDynamics, AngularVelocity] = pd.Field(
        discriminator="type_name"
    )
    parent_volume: Optional[Union[GenericVolume, Cylinder]] = pd.Field(None)

    @pd.field_validator("entities", mode="after")
    @classmethod
    def _ensure_entities_have_sufficient_attributes(cls, value: EntityList):
        """Ensure entities have sufficient attributes."""

        for entity in value.stored_entities:
            if entity.axis is None:
                raise ValueError(
                    f"Entity '{entity.name}' must specify `axis` to be used under `Rotation`."
                )
            if entity.center is None:
                raise ValueError(
                    f"Entity '{entity.name}' must specify `center` to be used under `Rotation`"
                )
        return value


class PorousMedium(Flow360BaseModel):
    """Constains Flow360Param PorousMediumBox and PorousMediumVolumeZone"""

    name: Optional[str] = pd.Field(None)
    type: Literal["PorousMedium"] = pd.Field("PorousMedium", frozen=True)
    entities: EntityList[GenericVolume, Box] = pd.Field(alias="volumes")

    darcy_coefficient: InverseAreaType.Point = pd.Field()
    forchheimer_coefficient: InverseLengthType.Point = pd.Field()
    volumetric_heat_source: Optional[Union[HeatSourceType, pd.StrictStr]] = pd.Field(None)
    # Note: Axes will always come from the entity

    @pd.field_validator("entities", mode="after")
    @classmethod
    def _ensure_entities_have_sufficient_attributes(cls, value: EntityList):
        """Ensure entities have sufficient attributes."""

        for entity in value.stored_entities:
            if entity.axes is None:
                raise ValueError(
                    f"Entity '{entity.name}' must specify `axes` to be used under `PorousMedium`."
                )
        return value


VolumeModelTypes = Union[
    Fluid,
    Solid,
    ActuatorDisk,
    BETDisk,
    Rotation,
    PorousMedium,
]
