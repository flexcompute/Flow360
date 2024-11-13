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
    """
    :class:`AngleExpression` class for define the angle expression for :paramref:`Rotation.spec`.
    """

    type_name: Literal["AngleExpression"] = pd.Field("AngleExpression", frozen=True)
    value: StringExpression = pd.Field(
        description="The expression defining the rotation angle as a function of time."
    )


class AngularVelocity(SingleAttributeModel):
    """
    :class:`AngularVelocity` class to define the angular velocity for :paramref:`Rotation.spec`.
    """

    type_name: Literal["AngularVelocity"] = pd.Field("AngularVelocity", frozen=True)
    value: AngularVelocityType = pd.Field(description="The value of the angular velocity.")


class FromUserDefinedDynamics(Flow360BaseModel):
    """
    :class:`FromUserDefinedDynamics` class to define the rotation
    controlled by user defined dynamics for :paramref:`Rotation.spec`.
    """

    type_name: Literal["FromUserDefinedDynamics"] = pd.Field("FromUserDefinedDynamics", frozen=True)


class ExpressionInitialConditionBase(Flow360BaseModel):
    """
    :class:`ExpressionInitialCondition` class for specifying the intial conditions of
    :paramref:`Fluid.initial_condition`.
    """

    type: Literal["expression"] = pd.Field("expression", frozen=True)
    constants: Optional[Dict[str, StringExpression]] = pd.Field(
        None, description="The expression for the initial condition."
    )


# pylint: disable=missing-class-docstring
class NavierStokesInitialCondition(ExpressionInitialConditionBase):
    """
    :class:`NavierStokesInitialCondition` class for specifying the
    :paramref:`Fluid.initial_condition`.

    """

    rho: StringExpression = pd.Field(description="Density")
    u: StringExpression = pd.Field(description="X-direction velocity")
    v: StringExpression = pd.Field(description="Y-direction velocity")
    w: StringExpression = pd.Field(description="Z-direction velocity")
    p: StringExpression = pd.Field(description="Pressure")


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
    :class:`Fluid` class for setting up the volume model that contains
    all the common fields every fluid dynamics zone should have.
    """

    type: Literal["Fluid"] = pd.Field("Fluid", frozen=True)
    navier_stokes_solver: NavierStokesSolver = pd.Field(
        NavierStokesSolver(),
        description="Navier-Stokes solver settings, see "
        + ":class:`NavierStokesSolver` documentation.",
    )
    turbulence_model_solver: TurbulenceModelSolverType = pd.Field(
        SpalartAllmaras(),
        description="Turbulence model solver settings, see "
        + ":class:`~flow360.TurbulenceModelSolver` documentation.",
    )
    transition_model_solver: TransitionModelSolverType = pd.Field(
        NoneSolver(),
        description="Transition solver settings, see "
        + ":class:`~flow360.TransitionModelSolver` documentation.",
    )

    material: FluidMaterialTypes = pd.Field(Air(), description="The material propetry of fluid.")

    initial_condition: Optional[
        Union[NavierStokesModifiedRestartSolution, NavierStokesInitialCondition]
    ] = pd.Field(
        None, discriminator="type", description="The initial condition of the fluid solver."
    )

    # pylint: disable=fixme
    # fixme: Add support for other initial conditions


class Solid(PDEModelBase):
    """
    :class:`Solid` class for setting up the HeatTransfer volume model that
    contains all the common fields every heat transfer zone should have.
    """

    name: Optional[str] = pd.Field(None, description="Name of the `Solid` model.")
    type: Literal["Solid"] = pd.Field("Solid", frozen=True)
    entities: EntityList[GenericVolume] = pd.Field(
        alias="volumes",
        description="The list of :class:`GenericVolume` "
        + "entities on which the heat transfer equation is solved.",
    )

    material: SolidMaterialTypes = pd.Field(description="The material property of solid.")

    heat_equation_solver: HeatEquationSolver = pd.Field(
        HeatEquationSolver(),
        description="Heat equation solver settings, see "
        + ":class:`HeatEquationSolver` documentation.",
    )
    volumetric_heat_source: Union[HeatSourceType, StringExpression] = pd.Field(
        0, description="The volumetric heat source."
    )

    initial_condition: Optional[HeatEquationInitialCondition] = pd.Field(
        None, description="The initial condition of the heat equation solver."
    )


# pylint: disable=duplicate-code
class ForcePerArea(Flow360BaseModel):
    """:class:`ForcePerArea` class for setting up force per area for Actuator Disk.

    Example
    -------
    >>> fpa = ForcePerArea(radius=[0, 1], thrust=[1, 1], circumferential=[1, 1]) # doctest: +SKIP

    TODO: Use dimensioned values
    """

    # pylint: disable=no-member
    radius: LengthType.Array = pd.Field(description="Radius of the sampled locations in grid unit.")
    # pylint: disable=no-member
    thrust: PressureType.Array = pd.Field(
        description="Dimensional force per area in the axial direction, positive means the axial "
        + "force follows the same direction as the thrust axis. "
    )
    # pylint: disable=no-member
    circumferential: PressureType.Array = pd.Field(
        description="Dimensional force per area in the circumferential direction, positive means the "
        + "circumferential force follows the same direction as the thrust axis with the right hand rule. "
    )

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
    """:class:`ActuatorDisk` class for setting up the inputs for an Actuator Disk.
    Please refer to the :ref:`actuator disk knowledge base <knowledge_base_actuatorDisks>` for further information.
    Note that `center`, `axis_thrust`, `thickness` can be acquired from `entity` so they are not required anymore.
    """

    entities: EntityList[Cylinder] = pd.Field(
        alias="volumes",
        description="The list of :class:`Cylinder` entities for the `ActuatorDisk` model",
    )
    force_per_area: ForcePerArea = pd.Field(
        description="The force per area input for the `ActuatorDisk` model. "
        + "See :class:`ForcePerArea` documentation."
    )
    name: Optional[str] = pd.Field(None, description="Name of the `ActuatorDisk` model.")
    type: Literal["ActuatorDisk"] = pd.Field("ActuatorDisk", frozen=True)


class BETDiskTwist(Flow360BaseModel):
    """:class:`BETDiskTwist` class for setting up the :paramref:`BETDisk.twists`."""

    # TODO: Use dimensioned values, why optional?
    radius: Optional[float] = pd.Field(None, description="A list of radial locations.")
    twist: Optional[float] = pd.Field(
        None,
        description="The twist in degrees as a function of radial location. "
        + "Entries in the list must already be sorted by radius.",
    )


class BETDiskChord(Flow360BaseModel):
    """:class:`BETDiskChord` class for setting up the :paramref:`BETDisk.chords`."""

    # TODO: Use dimensioned values, why optional?
    radius: Optional[float] = pd.Field(None, description="A list of radial locations.")
    chord: Optional[float] = pd.Field(
        None,
        description="The blade chord as a function of the radial location. "
        + "Entries in the list must already be sorted by radius.",
    )


class BETDiskSectionalPolar(Flow360BaseModel):
    """:class:`BETDiskSectionalPolar` class for setting up :paramref:`BETDisk.sectional_polars`
    for :class:`BETDisk`. There are two variables, “lift_coeffs” and “drag_coeffs”,
    need to be set up as 3D arrays (implemented as nested lists).
    The first index of the array corresponds to the :paramref:`BETDisk.mach_numbers`
    of the specified polar data.
    The second index of the array corresponds to the :paramref:`BETDisk.reynolds_numbers`
    of the polar data.
    The third index corresponds to the :paramref:`BETDisk.alphas`.
    The value specifies the lift or drag coefficient, respectively.
    """

    lift_coeffs: Optional[List[List[List[float]]]] = pd.Field(
        description="The 3D arrays specifying the list coefficient."
    )
    drag_coeffs: Optional[List[List[List[float]]]] = pd.Field(
        description="The 3D arrays specifying the drag coefficient."
    )


# pylint: disable=no-member
class BETDisk(Flow360BaseModel):
    """:class:`BETDisk` class for defining the Blade Element Theory (BET) model inputs.
    For detailed information on the parameters, please refer to the :ref:`BET knowledge Base <knowledge_base_BETDisks>`.
    To generate the sectional polars the BET translators can be used which are
    outlined :ref:`here <BET_Translators>`
    with best-practices for the sectional polars inputs available :ref:`here <secPolars_bestPractices>`.
    A case study of the XV-15 rotor using the steady BET Disk method is available
    in :ref:`Case Studies <XV15BETDisk_caseStudy>`.
    Because a transient BET Line simulation is simply a time-accurate version of a steady-state
    BET Disk simulation, most of the parameters below are applicable to both methods.
    Note that `center_of_rotation`, `axis_of_rotation`, `radius`, `thickness` can be acquired from `entity`
    so they are not required anymore.
    """

    name: Optional[str] = pd.Field(None, description="Name of the `BETDisk` model.")
    type: Literal["BETDisk"] = pd.Field("BETDisk", frozen=True)
    entities: EntityList[Cylinder] = pd.Field(alias="volumes")

    rotation_direction_rule: Literal["leftHand", "rightHand"] = pd.Field(
        "rightHand",
        description='The rule for rotation direction and thrust direction, "rightHand" or "leftHand".',
    )
    number_of_blades: pd.StrictInt = pd.Field(gt=0, le=10, description="Number of blades to model.")
    omega: AngularVelocityType.NonNegative = pd.Field(description="Rotating speed.")
    chord_ref: LengthType.Positive = pd.Field(
        description="Dimensional reference chord used to compute sectional blade loadings."
    )
    n_loading_nodes: pd.StrictInt = pd.Field(
        gt=0,
        le=1000,
        description="Number of nodes used to compute the sectional thrust and "
        + "torque coefficients :math:`C_t` and :math:`C_q`, defined in :ref:`betDiskLoadingNote`.",
    )
    blade_line_chord: LengthType.NonNegative = pd.Field(
        0,
        description="Dimensional chord to use if performing an unsteady BET Line simulation. "
        + "Default of 0.0 is an indication to run a steady BET Disk simulation.",
    )
    initial_blade_direction: Optional[Axis] = pd.Field(
        None,
        description="Orientation of the first blade in the BET model. "
        + "Must be specified if performing an unsteady BET Line simulation.",
    )
    tip_gap: Union[Literal["inf"], LengthType.NonNegative] = pd.Field(
        "inf",
        description="Dimensional distance between blade tip and solid bodies to "
        + "define a :ref:`tip loss factor <TipGap>`.",
    )
    mach_numbers: List[pd.NonNegativeFloat] = pd.Field(
        description="Mach numbers associated with airfoil polars provided "
        + "in :class:`BETDiskSectionalPolar`."
    )
    reynolds_numbers: List[pd.PositiveFloat] = pd.Field(
        description="Reynolds numbers associated with the airfoil polars "
        + "provided in :class:`BETDiskSectionalPolar`."
    )
    alphas: List[float] = pd.Field(
        description="Alphas associated with airfoil polars provided in "
        + ":class:`BETDiskSectionalPolar` in degrees."
    )
    twists: List[BETDiskTwist] = pd.Field(
        description="A list of :class:`BETDiskTwist` objects specifying the twist in degrees as a "
        + "function of radial location."
    )
    chords: List[BETDiskChord] = pd.Field(
        description="A list of :class:`BETDiskChord` objects specifying the blade chord as a function "
        + "of the radial location. "
    )
    sectional_polars: List[BETDiskSectionalPolar] = pd.Field(
        description="A list of :class:`BETDiskSectionalPolar` objects for every radial location specified in "
        + ":paramref:`sectional_radiuses`."
    )
    sectional_radiuses: List[float] = pd.Field(
        description="A list of the radial locations in grid units at which :math:`C_l` "
        + "and :math:`C_d` are specified in :class:`BETDiskSectionalPolar`."
    )

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
    """
    :class:`Rotation` class for specifying rotation settings.
    """

    name: Optional[str] = pd.Field(None, description="Name of the `Rotation` model.")
    type: Literal["Rotation"] = pd.Field("Rotation", frozen=True)
    entities: EntityList[GenericVolume, Cylinder] = pd.Field(
        alias="volumes",
        description="The entity list for the `Rotation` model. "
        + "The entity should be :class:`Cylinder` or :class:`GenericVolume` type.",
    )

    # TODO: Add test for each of the spec specification.
    spec: Union[AngleExpression, FromUserDefinedDynamics, AngularVelocity] = pd.Field(
        discriminator="type_name",
        description="The angular velocity or rotation angle as a function of time.",
    )
    parent_volume: Optional[Union[GenericVolume, Cylinder]] = pd.Field(
        None,
        description="The parent rotating entity in a nested rotation case."
        + "The entity should be :class:`Cylinder` or :class:`GenericVolume` type.",
    )

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
    """
    :class:`PorousMedium` class for specifying porous media settings.`.
    For further information please refer to the :ref:`porous media knowledge base <knowledge_base_porousMedia>`
    """

    name: Optional[str] = pd.Field(None, description="Name of the `PorousMedium` model.")
    type: Literal["PorousMedium"] = pd.Field("PorousMedium", frozen=True)
    entities: EntityList[GenericVolume, Box] = pd.Field(
        alias="volumes",
        description="The entity list for the `PorousMedium` model. "
        + "The entity should be :class:`Box` type.",
    )

    darcy_coefficient: InverseAreaType.Point = pd.Field(
        description="Darcy coefficient of the porous media model which determines the scaling of the "
        + "viscous loss term. The 3 values define the coefficient for each of the 3 axes defined by "
        + "the reference frame of the volume zone."
    )
    forchheimer_coefficient: InverseLengthType.Point = pd.Field(
        description="Forchheimer coefficient of the porous media model which determines "
        + "the scaling of the inertial loss term."
    )
    volumetric_heat_source: Optional[Union[HeatSourceType, StringExpression]] = pd.Field(
        None, description="The volumetric heat source."
    )
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
