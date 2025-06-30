"""Volume models for the simulation framework."""

# pylint: disable=too-many-lines
import os
import re
from abc import ABCMeta
from typing import Annotated, Dict, List, Literal, Optional, Union

import pydantic as pd

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.expressions import (
    StringExpression,
    validate_angle_expression_of_t_seconds,
)
from flow360.component.simulation.framework.multi_constructor_model_base import (
    MultiConstructorBaseModel,
)
from flow360.component.simulation.framework.single_attribute_base import (
    SingleAttributeModel,
)
from flow360.component.simulation.models.bet.bet_translator_interface import (
    generate_c81_bet_json,
    generate_dfdc_bet_json,
    generate_polar_file_name_list,
    generate_xfoil_bet_json,
    generate_xrotor_bet_json,
    get_file_content,
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
    _check_bet_disk_initial_blade_direction,
    _check_bet_disk_initial_blade_direction_and_blade_line_chord,
    _check_bet_disk_sectional_radius_and_polars,
)
from flow360.component.simulation.primitives import Box, Cylinder, GenericVolume
from flow360.component.simulation.unit_system import (
    AngleType,
    AngularVelocityType,
    HeatSourceType,
    InverseAreaType,
    InverseLengthType,
    LengthType,
    PressureType,
    u,
)
from flow360.component.simulation.validation.validation_context import (
    get_validation_info,
)
from flow360.component.simulation.validation.validation_utils import (
    _validator_append_instance_name,
)

# pylint: disable=fixme
# TODO: Warning: Pydantic V1 import
from flow360.component.types import Axis


class AngleExpression(SingleAttributeModel):
    """
    :class:`AngleExpression` class for define the angle expression for :py:attr:`Rotation.spec`.
    The result of the expression is assumed to be in radians.

    Example
    -------

    >>> fl.AngleExpression("0.1*sin(t)")

    ====
    """

    type_name: Literal["AngleExpression"] = pd.Field("AngleExpression", frozen=True)
    value: StringExpression = pd.Field(
        description="The expression defining the rotation angle as a function of time."
    )

    @pd.field_validator("value", mode="after")
    @classmethod
    def _validate_angle_expression(cls, value):
        errors = validate_angle_expression_of_t_seconds(value)
        if errors:
            raise ValueError(" | ".join(errors))
        return value

    def preprocess(self, **kwargs):
        # locate t_seconds and convert it to (t*flow360_time_to_seconds)
        params = kwargs.get("params")
        one_sec_to_flow360_time = params.convert_unit(
            value=1 * u.s, target_system="flow360_v2"  # pylint:disable=no-member
        )
        flow360_time_to_seconds_expression = f"({1.0/one_sec_to_flow360_time.value} * t)"
        self.value = re.sub(r"\bt_seconds\b", flow360_time_to_seconds_expression, self.value)

        return super().preprocess(**kwargs)


class AngularVelocity(SingleAttributeModel):
    """
    :class:`AngularVelocity` class to define the angular velocity for :py:attr:`Rotation.spec`.

    Example
    -------

    >>> fl.AngularVelocity(812.31 * fl.u.rpm)

    >>> fl.AngularVelocity(85.06 * fl.u.rad / fl.u.s)

    ====
    """

    type_name: Literal["AngularVelocity"] = pd.Field("AngularVelocity", frozen=True)
    value: AngularVelocityType = pd.Field(description="The value of the angular velocity.")


class FromUserDefinedDynamics(Flow360BaseModel):
    """
    :class:`FromUserDefinedDynamics` class to define the rotation
    controlled by user defined dynamics for :py:attr:`Rotation.spec`.

    Example
    -------

    >>> params=fl.SimulationParams(...)
    >>> params.user_defined_dynamics=fl.UserDefinedDynamic(...)
    >>> params.models.append(
    ...     fl.Rotation(
    ...        spec=fl.FromUserDefinedDynamics(),
    ...        entities=[rotation_entity]
    ...     )
    ... )

    ====
    """

    type_name: Literal["FromUserDefinedDynamics"] = pd.Field("FromUserDefinedDynamics", frozen=True)


class ExpressionInitialConditionBase(Flow360BaseModel):
    """
    :class:`ExpressionInitialCondition` class for specifying the initial conditions of
    :py:attr:`Fluid.initial_condition`.
    """

    type_name: Literal["expression"] = pd.Field("expression", frozen=True)
    constants: Optional[Dict[str, StringExpression]] = pd.Field(
        None, description="The expression for the initial condition."
    )


# pylint: disable=missing-class-docstring
class NavierStokesInitialCondition(ExpressionInitialConditionBase):
    """
    :class:`NavierStokesInitialCondition` class for specifying the
    :py:attr:`Fluid.initial_condition`.

    Note
    ----
    The result of the expressions will be treated as non-dimensional values.
    Please refer to the :ref:`Units Introduction<API_units_introduction>` for more details.

    Example
    -------

    >>> fl.NavierStokesInitialCondition(
    ...     rho = "(x <= 0) ? (1.0) : (0.125)",
    ...     u = "0",
    ...     v = "0",
    ...     w = "0",
    ...     p = "(x <= 0) ? (1 / 1.4) : (0.1 / 1.4)"
    ... )

    ====
    """

    type_name: Literal["NavierStokesInitialCondition"] = pd.Field(
        "NavierStokesInitialCondition", frozen=True
    )
    rho: StringExpression = pd.Field("rho", description="Density")
    u: StringExpression = pd.Field("u", description="X-direction velocity")
    v: StringExpression = pd.Field("v", description="Y-direction velocity")
    w: StringExpression = pd.Field("w", description="Z-direction velocity")
    p: StringExpression = pd.Field("p", description="Pressure")

    @pd.field_validator("rho", "u", "v", "w", "p", mode="after")
    @classmethod
    def _disable_expression_for_liquid(cls, value, info: pd.ValidationInfo):
        validation_info = get_validation_info()
        if validation_info is None or validation_info.using_liquid_as_material is False:
            return value

        if value != cls.model_fields[info.field_name].get_default():
            raise ValueError("Expression cannot be used when using liquid as simulation material.")
        return value


class NavierStokesModifiedRestartSolution(NavierStokesInitialCondition):
    type_name: Literal["NavierStokesModifiedRestartSolution"] = pd.Field(
        "NavierStokesModifiedRestartSolution", frozen=True
    )


class HeatEquationInitialCondition(ExpressionInitialConditionBase):
    """
    :class:`HeatEquationInitialCondition` class for specifying the
    :py:attr:`Solid.initial_condition`.

    Note
    ----
    The result of the expressions will be treated as non-dimensional values.
    Please refer to the :ref:`Units Introduction<API_units_introduction>` for more details.

    Example
    -------

    >>> fl.HeatEquationInitialCondition(temperature="1.0")

    ====
    """

    type_name: Literal["HeatEquationInitialCondition"] = pd.Field(
        "HeatEquationInitialCondition", frozen=True
    )
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

    Example
    -------

    >>> fl.Fluid(
    ...     navier_stokes_solver=fl.NavierStokesSolver(
    ...         absolute_tolerance=1e-10,
    ...         linear_solver=fl.LinearSolver(max_iterations=35),
    ...         low_mach_preconditioner=True,
    ...     ),
    ...     turbulence_model_solver=fl.SpalartAllmaras(
    ...         absolute_tolerance=1e-10,
    ...         linear_solver=fl.LinearSolver(max_iterations=25)
    ...     ),
    ...     transition_model_solver=fl.NoneSolver(),
    ... )

    ====
    """

    type: Literal["Fluid"] = pd.Field("Fluid", frozen=True)
    navier_stokes_solver: NavierStokesSolver = pd.Field(
        NavierStokesSolver(),
        description="Navier-Stokes solver settings, see "
        + ":class:`NavierStokesSolver` documentation.",
    )
    turbulence_model_solver: TurbulenceModelSolverType = pd.Field(
        SpalartAllmaras(),
        description="Turbulence model solver settings, see :class:`SpalartAllmaras`, "
        + ":class:`KOmegaSST` and :class:`NoneSolver` documentation.",
    )
    transition_model_solver: TransitionModelSolverType = pd.Field(
        NoneSolver(),
        description="Transition solver settings, see "
        + ":class:`TransitionModelSolver` documentation.",
    )

    material: FluidMaterialTypes = pd.Field(Air(), description="The material property of fluid.")

    initial_condition: Union[NavierStokesModifiedRestartSolution, NavierStokesInitialCondition] = (
        pd.Field(
            NavierStokesInitialCondition(),
            discriminator="type_name",
            description="The initial condition of the fluid solver.",
        )
    )

    # pylint: disable=fixme
    # fixme: Add support for other initial conditions


class Solid(PDEModelBase):
    """
    :class:`Solid` class for setting up the conjugate heat transfer volume model that
    contains all the common fields every heat transfer zone should have.

    Example
    -------

    Define :class:`Solid` model for volumes with the name pattern :code:`"solid-*"`.

    >>> fl.Solid(
    ...     entities=[volume_mesh["solid-*"]],
    ...     heat_equation_solver=fl.HeatEquationSolver(
    ...         equation_evaluation_frequency=2,
    ...         linear_solver=fl.LinearSolver(
    ...             absolute_tolerance=1e-10,
    ...             max_iterations=50
    ...         ),
    ...         relative_tolerance=0.001,
    ...     ),
    ...     initial_condition=fl.HeatEquationInitialCondition(temperature="1.0"),
    ...     material=fl.SolidMaterial(
    ...         name="aluminum",
    ...         thermal_conductivity=235 * fl.u.kg / fl.u.s**3 * fl.u.m / fl.u.K,
    ...         density=2710 * fl.u.kg / fl.u.m**3,
    ...         specific_heat_capacity=903 * fl.u.m**2 / fl.u.s**2 / fl.u.K,
    ...     ),
    ...     volumetric_heat_source=1.0 * fl.u.W / fl.u.m**3,
    ... )

    ====
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
    # pylint: disable=no-member
    volumetric_heat_source: Union[StringExpression, HeatSourceType] = pd.Field(
        0 * u.W / (u.m**3), description="The volumetric heat source."
    )

    initial_condition: Optional[HeatEquationInitialCondition] = pd.Field(
        None, description="The initial condition of the heat equation solver."
    )


# pylint: disable=duplicate-code
class ForcePerArea(Flow360BaseModel):
    """:class:`ForcePerArea` class for setting up force per area for Actuator Disk.

    Example
    -------

    >>> fl.ForcePerArea(
    ...     radius=[0, 1] * fl.u.mm,
    ...     thrust=[4.1, 5.5] * fl.u.Pa,
    ...     circumferential=[4.1, 5.5] * fl.u.Pa,
    ... )

    ====
    """

    # pylint: disable=no-member
    radius: LengthType.NonNegativeArray = pd.Field(
        description="Radius of the sampled locations in grid unit."
    )
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

    Note
    ----
    :py:attr:`Cylinder.center`, :py:attr:`Cylinder.axis` and :py:attr:`Cylinder.height` are taken as the
    center, thrust axis, and thickness of the Actuator Disk, respectively.

    Example
    -------

    >>> fl.ActuatorDisk(
    ...     entities = fl.Cylinder(
    ...         name="actuator_disk",
    ...         center=(0,0,0)*fl.u.mm,
    ...         axis=(-1,0,0),
    ...         height = 30 * fl.u.mm,
    ...         outer_radius=5.0 * fl.u.mm,
    ...     ),
    ...     force_per_area = fl.ForcePerArea(
    ...          radius=[0, 1] * fl.u.mm,
    ...          thrust=[4.1, 5.5] * fl.u.Pa,
    ...          circumferential=[4.1, 5.5] * fl.u.Pa,
    ...     )
    ... )

    ====
    """

    entities: EntityList[Cylinder] = pd.Field(
        alias="volumes",
        description="The list of :class:`Cylinder` entities for the `ActuatorDisk` model",
    )
    force_per_area: ForcePerArea = pd.Field(
        description="The force per area input for the `ActuatorDisk` model. "
        + "See :class:`ForcePerArea` documentation."
    )
    name: Optional[str] = pd.Field("Actuator disk", description="Name of the `ActuatorDisk` model.")
    type: Literal["ActuatorDisk"] = pd.Field("ActuatorDisk", frozen=True)


# pylint: disable=no-member
class BETDiskTwist(Flow360BaseModel):
    """
    :class:`BETDiskTwist` class for setting up the :py:attr:`BETDisk.twists`.

    Example
    -------

    >>> fl.BETDiskTwist(radius=2 * fl.u.inch, twist=26 * fl.u.deg)

    ====
    """

    radius: LengthType.NonNegative = pd.Field(description="The radius of the radial location.")
    twist: AngleType = pd.Field(description="The twist angle at this radial location.")


# pylint: disable=no-member
class BETDiskChord(Flow360BaseModel):
    """
    :class:`BETDiskChord` class for setting up the :py:attr:`BETDisk.chords`.

    Example
    -------

    >>> fl.BETDiskChord(radius=2 * fl.u.inch, chord=18 * fl.u.inch)

    ====
    """

    radius: LengthType.NonNegative = pd.Field(description="The radius of the radial location.")
    chord: LengthType.NonNegative = pd.Field(
        description="The blade chord at this radial location. "
    )


class BETDiskSectionalPolar(Flow360BaseModel):
    """:class:`BETDiskSectionalPolar` class for setting up :py:attr:`BETDisk.sectional_polars`
    for :class:`BETDisk`. There are two variables, “lift_coeffs” and “drag_coeffs”,
    need to be set up as 3D arrays (implemented as nested lists).
    The first index of the array corresponds to the :py:attr:`BETDisk.mach_numbers`
    of the specified polar data.
    The second index of the array corresponds to the :py:attr:`BETDisk.reynolds_numbers`
    of the polar data.
    The third index corresponds to the :py:attr:`BETDisk.alphas`.
    The value specifies the lift or drag coefficient, respectively.

    Example
    -------

    Define :class:`BETDiskSectionalPolar` at one single radial location.
    :code:`lift_coeffs` and :code:`drag_coeffs` are lists with the dimension of 3 x 2 x 2, corresponding to
    3 :py:attr:`BETDisk.mach_numbers` by 2 :py:attr:`BETDisk.reynolds_numbers` by 2 :py:attr:`BETDisk.alphas`.

    >>> lift_coeffs = [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]]
    >>> drag_coeffs = [[[0.01, 0.02], [0.03, 0.04]], [[0.05, 0.06], [0.07, 0.08]], [[0.09, 0.1], [0.11, 0.12]]]
    >>> fl.BETDiskSectionalPolar(
    ...     lift_coeffs=lift_coeffs,
    ...     drag_coeffs=drag_coeffs
    ... )

    ====
    """

    lift_coeffs: List[List[List[float]]] = pd.Field(
        description="The 3D arrays specifying the list coefficient."
    )
    drag_coeffs: List[List[List[float]]] = pd.Field(
        description="The 3D arrays specifying the drag coefficient."
    )


class BETSingleInputFileBaseModel(Flow360BaseModel, metaclass=ABCMeta):
    file_path: str = pd.Field(
        frozen=True,
        description="Path to the BET configuration file. It cannot be changed once initialized.",
    )
    content: str = pd.Field(
        frozen=True,
        description="File content of the BET configuration file. It will be automatically loaded.",
    )

    @pd.model_validator(mode="before")
    @classmethod
    def _extract_content(cls, input_data):
        """
        Read the file content and store it as string.
        """
        if "file_path" not in input_data:
            raise ValueError("file_path is require but is not found in input.")
        if "content" in input_data and input_data.get("content"):
            return input_data

        file_content = get_file_content(input_data["file_path"])

        return {"file_path": os.path.basename(input_data["file_path"]), "content": file_content}


class AuxiliaryPolarFile(BETSingleInputFileBaseModel):
    """Auxiliary polar file for XFoil"""

    type_name: Literal["AuxiliaryPolarFile"] = pd.Field("AuxiliaryPolarFile", frozen=True)


class BETSingleMultiFileBaseModel(Flow360BaseModel, metaclass=ABCMeta):
    file_path: str = pd.Field(
        frozen=True,
        description="Path to the BET configuration file. It cannot be changed once initialized.",
    )
    content: str = pd.Field(
        frozen=True,
        description="File content of the BET configuration file. It will be automatically loaded.",
    )
    polar_files: list[list[AuxiliaryPolarFile]] = pd.Field()

    @pd.model_validator(mode="before")
    @classmethod
    def _extract_content(cls, input_data):
        """
        Read the file content and store it as string.
        """
        if "file_path" not in input_data:
            raise ValueError("file_path is require but is not found in input.")
        if "content" in input_data and input_data.get("content"):
            return input_data
        if "polar_files" in input_data and input_data.get("polar_files"):
            return input_data

        file_path = input_data["file_path"]
        file_content = get_file_content(file_path=file_path)

        # Now read the polar files
        polar_file_obj_list = []
        file_dir = os.path.dirname(file_path)
        for file_name_list in generate_polar_file_name_list(geometry_file_content=file_content):
            polar_file_obj_list.append(
                [{"file_path": os.path.join(file_dir, file_name)} for file_name in file_name_list]
            )
        return {
            "file_path": os.path.basename(file_path),
            "content": file_content,
            "polar_files": polar_file_obj_list,
        }


class XROTORFile(BETSingleInputFileBaseModel):
    type_name: Literal["XRotorFile"] = pd.Field("XRotorFile", frozen=True)


class DFDCFile(BETSingleInputFileBaseModel):
    type_name: Literal["DFDCFile"] = pd.Field("DFDCFile", frozen=True)


class C81File(BETSingleMultiFileBaseModel):
    type_name: Literal["C81File"] = pd.Field("C81File", frozen=True)


class XFOILFile(BETSingleMultiFileBaseModel):
    type_name: Literal["XFoilFile"] = pd.Field("XFoilFile", frozen=True)


BETFileTypes = Annotated[
    Union[XROTORFile, DFDCFile, XFOILFile, C81File],
    pd.Field(discriminator="type_name"),
]


class BETDiskCache(Flow360BaseModel):
    """[INTERNAL] Cache for BETDisk inputs"""

    name: Optional[str] = None
    file: Optional[BETFileTypes] = None
    rotation_direction_rule: Optional[Literal["leftHand", "rightHand"]] = None
    omega: Optional[AngularVelocityType.NonNegative] = None
    chord_ref: Optional[LengthType.Positive] = None
    n_loading_nodes: Optional[pd.StrictInt] = None
    entities: Optional[EntityList[Cylinder]] = None
    angle_unit: Optional[AngleType] = None
    length_unit: Optional[LengthType.NonNegative] = None
    number_of_blades: Optional[pd.StrictInt] = None
    initial_blade_direction: Optional[Axis] = None
    blade_line_chord: Optional[LengthType.NonNegative] = None


class BETDisk(MultiConstructorBaseModel):
    """:class:`BETDisk` class for defining the Blade Element Theory (BET) model inputs.
    For detailed information on the parameters, please refer to the :ref:`BET knowledge Base <knowledge_base_BETDisks>`.
    To generate the sectional polars the BET translators can be used which are
    outlined :ref:`here <BET_Translators>`
    with best-practices for the sectional polars inputs available :ref:`here <secPolars_bestPractices>`.
    A validation study of the XV-15 rotor using the steady BET Disk method is available
    in :ref:`Validation Studies <XV15BETDiskValidationStudy>`.
    Because a transient BET Line simulation is simply a time-accurate version of a steady-state
    BET Disk simulation, most of the parameters below are applicable to both methods.

    Note
    ----
    :py:attr:`Cylinder.center`, :py:attr:`Cylinder.axis`, :py:attr:`Cylinder.outer_radius`,
    and :py:attr:`Cylinder.height` are taken as the rotation center,
    rotation axis, radius, and thickness of the BETDisk, respectively.

    Example
    -------
    >>> fl.BETDisk(
    ...    entities=[fl.Cylinder(...)],
    ...    rotation_direction_rule="leftHand",
    ...    number_of_blades=3,
    ...    omega=rpm * fl.u.rpm,
    ...    chord_ref=14 * fl.u.inch,
    ...    n_loading_nodes=20,
    ...    mach_numbers=[0],
    ...    reynolds_numbers=[1000000],
    ...    twists=[fl.BETDiskTwist(...), ...],
    ...    chords=[fl.BETDiskChord(...), ...],
    ...    alphas=[-2,0,2] * fl.u.deg,
    ...    sectional_radiuses=[13.5, 25.5] * fl.u.inch,
    ...    sectional_polars=[fl.BETDiskSectionalPolar(...), ...]
    ... )

    ====
    """

    name: Optional[str] = pd.Field("BET disk", description="Name of the `BETDisk` model.")
    type: Literal["BETDisk"] = pd.Field("BETDisk", frozen=True)
    type_name: Literal["BETDisk"] = pd.Field("BETDisk", frozen=True)
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
        0 * u.m,
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
        frozen=True,
    )
    mach_numbers: List[pd.NonNegativeFloat] = pd.Field(
        description="Mach numbers associated with airfoil polars provided "
        + "in :class:`BETDiskSectionalPolar`.",
        frozen=True,
    )
    reynolds_numbers: List[pd.PositiveFloat] = pd.Field(
        description="Reynolds numbers associated with the airfoil polars "
        + "provided in :class:`BETDiskSectionalPolar`.",
        frozen=True,
    )
    alphas: AngleType.Array = pd.Field(
        description="Alphas associated with airfoil polars provided in "
        + ":class:`BETDiskSectionalPolar`.",
        frozen=True,
    )
    twists: List[BETDiskTwist] = pd.Field(
        description="A list of :class:`BETDiskTwist` objects specifying the twist in degrees as a "
        + "function of radial location.",
        frozen=True,
    )
    chords: List[BETDiskChord] = pd.Field(
        description="A list of :class:`BETDiskChord` objects specifying the blade chord as a function "
        + "of the radial location. ",
        frozen=True,
    )
    sectional_polars: List[BETDiskSectionalPolar] = pd.Field(
        description="A list of :class:`BETDiskSectionalPolar` objects for every radial location specified in "
        + ":py:attr:`sectional_radiuses`.",
        frozen=True,
    )
    sectional_radiuses: LengthType.NonNegativeArray = pd.Field(
        description="A list of the radial locations in grid units at which :math:`C_l` "
        + "and :math:`C_d` are specified in :class:`BETDiskSectionalPolar`.",
        frozen=True,
    )

    private_attribute_input_cache: BETDiskCache = BETDiskCache()

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

    @pd.field_validator("initial_blade_direction", mode="after")
    @classmethod
    def invalid_growth_rate(cls, value, info: pd.ValidationInfo):
        """Ensure initial_blade_direction is specified in an unsteady simulation"""
        return _check_bet_disk_initial_blade_direction(value, info)

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

    @pd.field_validator(
        "name",
        "rotation_direction_rule",
        "omega",
        "chord_ref",
        "n_loading_nodes",
        "number_of_blades",
        "entities",
        "initial_blade_direction",
        mode="after",
    )
    @classmethod
    def _update_input_cache(cls, value, info: pd.ValidationInfo):
        setattr(
            info.data["private_attribute_input_cache"],
            info.field_name,
            value if info.field_name != "entities" else value.stored_entities,
        )
        return value

    # pylint: disable=too-many-arguments, no-self-argument, not-callable
    @MultiConstructorBaseModel.model_constructor
    @pd.validate_call
    def from_c81(
        cls,
        file: C81File,
        rotation_direction_rule: Literal["leftHand", "rightHand"],
        omega: AngularVelocityType.NonNegative,
        chord_ref: LengthType.Positive,
        n_loading_nodes: pd.StrictInt,
        entities: EntityList[Cylinder],
        number_of_blades: pd.StrictInt,
        length_unit: LengthType.NonNegative,
        angle_unit: AngleType,
        initial_blade_direction: Optional[Axis] = None,
        blade_line_chord: LengthType.NonNegative = 0 * u.m,
        name: str = "BET disk",
    ):
        """Constructs a :class: `BETDisk` instance from a given C81 file and additional inputs.

        Parameters
        ----------
        file: C81File
            C81File class instance containing information about the C81 file.
        rotation_direction_rule: str
            Rule for rotation direction and thrust direction.
        omega: AngularVelocityType.NonNegative
            Rotating speed of the propeller.
        chord_ref: LengthType.Positive
            Dimensional reference cord used to compute sectional blade loadings.
        n_loading_nodes: Int
            Number of nodes used to compute sectional thrust and torque coefficients.
        entities: EntityList[Cylinder]
            List of Cylinder entities used for defining the BET volumes.
        number_of_blades: Int
            Number of blades to model.
        length_unit: LengthType.NonNegative
            Length unit of the geometry/mesh file.
        angle_unit: AngleType
            Angle unit used for AngleType BETDisk parameters.
        initial_blade_direction: Axis, optional
            Orientation of the first blade in BET model.
            Must be specified for unsteady BET simulation.
        blade_line_chord: LengthType.NonNegative
            Dimensional chord used in unsteady BET simulation. Defaults to ``0 * u.m``.


        Returns
        -------
        BETDisk
            An instance of :class:`BETDisk` completed with given inputs.

        Examples
        --------
        Create a BET disk with an C81 file.

        >>> param = fl.BETDisk.from_xrotor(
        ...     file=fl.C81File(file_path="c81_xv15.csv")),
        ...     rotation_direction_rule="leftHand",
        ...     omega=0.0046 * fl.u.deg / fl.u.s,
        ...     chord_ref=14 * fl.u.m,
        ...     n_loading_nodes=20,
        ...     entities=bet_cylinder,
        ...     angle_unit=fl.u.deg,
        ...     number_of_blades=3,
        ...     length_unit=fl.u.m,
        ... )
        """

        params = generate_c81_bet_json(
            geometry_file_content=file.content,
            c81_polar_file_list=file.polar_files,
            rotation_direction_rule=rotation_direction_rule,
            initial_blade_direction=initial_blade_direction,
            blade_line_chord=blade_line_chord,
            omega=omega,
            chord_ref=chord_ref,
            n_loading_nodes=n_loading_nodes,
            entities=entities,
            angle_unit=angle_unit,
            length_unit=length_unit,
            number_of_blades=number_of_blades,
            name=name,
        )

        return cls(**params)

    # pylint: disable=too-many-arguments, no-self-argument, not-callable
    @MultiConstructorBaseModel.model_constructor
    @pd.validate_call
    def from_dfdc(
        cls,
        file: DFDCFile,
        rotation_direction_rule: Literal["leftHand", "rightHand"],
        omega: AngularVelocityType.NonNegative,
        chord_ref: LengthType.Positive,
        n_loading_nodes: pd.StrictInt,
        entities: EntityList[Cylinder],
        length_unit: LengthType.NonNegative,
        angle_unit: AngleType,
        initial_blade_direction: Optional[Axis] = None,
        blade_line_chord: LengthType.NonNegative = 0 * u.m,
        name: str = "BET disk",
    ):
        """Constructs a :class: `BETDisk` instance from a given DFDC file and additional inputs.

        Parameters
        ----------
        file: DFDCFile
            DFDCFile class instance containing information about the DFDC file.
        rotation_direction_rule: str
            Rule for rotation direction and thrust direction.
        omega: AngularVelocityType.NonNegative
            Rotating speed of the propeller.
        chord_ref: LengthType.Positive
            Dimensional reference cord used to compute sectional blade loadings.
        n_loading_nodes: Int
            Number of nodes used to compute sectional thrust and torque coefficients.
        entities: EntityList[Cylinder]
            List of Cylinder entities used for defining the BET volumes.
        length_unit: LengthType.NonNegative
            Length unit used for LengthType BETDisk parameters.
        angle_unit: AngleType
            Angle unit used for AngleType BETDisk parameters.
        initial_blade_direction: Axis, optional
            Orientation of the first blade in BET model.
            Must be specified for unsteady BET simulation.
        blade_line_chord: LengthType.NonNegative
            Dimensional chord used in unsteady BET simulation. Defaults to ``0 * u.m``.


        Returns
        -------
        BETDisk
            An instance of :class:`BETDisk` completed with given inputs.

        Examples
        --------
        Create a BET disk with a DFDC file.

        >>> param = fl.BETDisk.from_dfdc(
        ...     file=fl.DFDCFile(file_path="dfdc_xv15.case")),
        ...     rotation_direction_rule="leftHand",
        ...     omega=0.0046 * fl.u.deg / fl.u.s,
        ...     chord_ref=14 * fl.u.m,
        ...     n_loading_nodes=20,
        ...     entities=bet_cylinder,
        ...     length_unit=fl.u.m,
        ...     angle_unit=fl.u.deg,
        ... )
        """

        params = generate_dfdc_bet_json(
            dfdc_file_content=file.content,
            rotation_direction_rule=rotation_direction_rule,
            initial_blade_direction=initial_blade_direction,
            blade_line_chord=blade_line_chord,
            omega=omega,
            chord_ref=chord_ref,
            n_loading_nodes=n_loading_nodes,
            entities=entities,
            angle_unit=angle_unit,
            length_unit=length_unit,
            name=name,
        )

        return cls(**params)

    # pylint: disable=too-many-arguments, no-self-argument, not-callable
    @MultiConstructorBaseModel.model_constructor
    @pd.validate_call
    def from_xfoil(
        cls,
        file: XFOILFile,
        rotation_direction_rule: Literal["leftHand", "rightHand"],
        omega: AngularVelocityType.NonNegative,
        chord_ref: LengthType.Positive,
        n_loading_nodes: pd.StrictInt,
        entities: EntityList[Cylinder],
        length_unit: LengthType.NonNegative,
        angle_unit: AngleType,
        number_of_blades: pd.StrictInt,
        initial_blade_direction: Optional[Axis],
        blade_line_chord: LengthType.NonNegative = 0 * u.m,
        name: str = "BET disk",
    ):
        """Constructs a :class: `BETDisk` instance from a given XROTOR file and additional inputs.

        Parameters
        ----------
        file: XFOILFile
            XFOILFile class instance containing information about the XFOIL file.
        rotation_direction_rule: str
            Rule for rotation direction and thrust direction.
        omega: AngularVelocityType.NonNegative
            Rotating speed of the propeller.
        chord_ref: LengthType.Positive
            Dimensional reference cord used to compute sectional blade loadings.
        n_loading_nodes: Int
            Number of nodes used to compute sectional thrust and torque coefficients.
        entities: EntityList[Cylinder]
            List of Cylinder entities used for defining the BET volumes.
        length_unit: LengthType.NonNegative
            Length unit used for LengthType BETDisk parameters.
        angle_unit: AngleType
            Angle unit used for AngleType BETDisk parameters.
        number_of_blades: Int
            Number of blades to model.
        initial_blade_direction: Axis, optional
            Orientation of the first blade in BET model.
            Must be specified for unsteady BET simulation.
        blade_line_chord: LengthType.NonNegative
            Dimensional chord used in unsteady BET simulation. Defaults to ``0 * u.m``.


        Returns
        -------
        BETDisk
            An instance of :class:`BETDisk` completed with given inputs.

        Examples
        --------
        Create a BET disk with an XFOIL file.

        >>> param = fl.BETDisk.from_xfoil(
        ...     file=fl.XFOILFile(file_path=("xfoil_xv15.csv")),
        ...     rotation_direction_rule="leftHand",
        ...     initial_blade_direction=[1, 0, 0],
        ...     blade_line_chord=1 * fl.u.m,
        ...     omega=0.0046 * fl.u.deg / fl.u.s,
        ...     chord_ref=14 * fl.u.m,
        ...     n_loading_nodes=20,
        ...     entities=bet_cylinder_imperial,
        ...     length_unit=fl.u.m,
        ...     angle_unit=fl.u.deg,
        ...     number_of_blades=3,
        )
        """

        params = generate_xfoil_bet_json(
            geometry_file_content=file.content,
            xfoil_polar_file_list=file.polar_files,
            rotation_direction_rule=rotation_direction_rule,
            initial_blade_direction=initial_blade_direction,
            blade_line_chord=blade_line_chord,
            omega=omega,
            chord_ref=chord_ref,
            n_loading_nodes=n_loading_nodes,
            entities=entities,
            angle_unit=angle_unit,
            length_unit=length_unit,
            number_of_blades=number_of_blades,
            name=name,
        )

        return cls(**params)

    # pylint: disable=too-many-arguments, no-self-argument, not-callable
    @MultiConstructorBaseModel.model_constructor
    @pd.validate_call
    def from_xrotor(
        cls,
        file: XROTORFile,
        rotation_direction_rule: Literal["leftHand", "rightHand"],
        omega: AngularVelocityType.NonNegative,
        chord_ref: LengthType.Positive,
        n_loading_nodes: pd.StrictInt,
        entities: EntityList[Cylinder],
        length_unit: LengthType.NonNegative,
        angle_unit: AngleType,
        initial_blade_direction: Optional[Axis] = None,
        blade_line_chord: LengthType.NonNegative = 0 * u.m,
        name: str = "BET disk",
    ):
        """Constructs a :class: `BETDisk` instance from a given XROTOR file and additional inputs.

        Parameters
        ----------
        file: XROTORFile
            XROTORFile class instance containing information about the XROTOR file.
        rotation_direction_rule: str
            Rule for rotation direction and thrust direction.
        omega: AngularVelocityType.NonNegative
            Rotating speed of the propeller.
        chord_ref: LengthType.Positive
            Dimensional reference cord used to compute sectional blade loadings.
        n_loading_nodes: Int
            Number of nodes used to compute sectional thrust and torque coefficients.
        entities: EntityList[Cylinder]
            List of Cylinder entities used for defining the BET volumes.
        length_unit: LengthType.NonNegative
            Length unit used for LengthType BETDisk parameters.
        angle_unit: AngleType
            Angle unit used for AngleType BETDisk parameters.
        initial_blade_direction: Axis, optional
            Orientation of the first blade in BET model.
            Must be specified for unsteady BET simulation.
        blade_line_chord: LengthType.NonNegative
            Dimensional chord used in unsteady BET simulation. Defaults to ``0 * u.m``.


        Returns
        -------
        BETDisk
            An instance of :class:`BETDisk` completed with given inputs.

        Examples
        --------
        Create a BET disk with an XROTOR file.

        >>> param = fl.BETDisk.from_xrotor(
        ...     file=fl.XROTORFile(file_path="xrotor_xv15.xrotor")),
        ...     rotation_direction_rule="leftHand",
        ...     omega=0.0046 * fl.u.deg / fl.u.s,
        ...     chord_ref=14 * fl.u.m,
        ...     n_loading_nodes=20,
        ...     entities=bet_cylinder,
        ...     angle_unit=fl.u.deg,
        ...     length_unit=fl.u.m,
        ... )
        """

        params = generate_xrotor_bet_json(
            xrotor_file_content=file.content,
            rotation_direction_rule=rotation_direction_rule,
            initial_blade_direction=initial_blade_direction,
            blade_line_chord=blade_line_chord,
            omega=omega,
            chord_ref=chord_ref,
            n_loading_nodes=n_loading_nodes,
            entities=entities,
            angle_unit=angle_unit,
            length_unit=length_unit,
            name=name,
        )

        return cls(**params)


class Rotation(Flow360BaseModel):
    """
    :class:`Rotation` class for specifying rotation settings.

    Example
    -------

    Define a rotation model :code:`outer_rotation` for the :code:`volume_mesh["outer"]` volume.
    The rotation center and axis are defined via the rotation entity's property:

    >>> outer_rotation_volume = volume_mesh["outer"]
    >>> outer_rotation_volume.center = (-1, 0, 0) * fl.u.m
    >>> outer_rotation_volume.axis = (0, 1, 0)
    >>> outer_rotation = fl.Rotation(
    ...     name="outerRotation",
    ...     volumes=[outer_rotation_volume],
    ...     spec= fl.AngleExpression("sin(t)"),
    ... )

    Define another rotation model :code:`inner_rotation` for the :code:`volume_mesh["inner"]` volume.
    :code:`inner_rotation` is nested in :code:`outer_rotation` by setting :code:`volume_mesh["outer"]`
    as the :py:attr:`Rotation.parent_volume`:

    >>> inner_rotation_volume = volume_mesh["inner"]
    >>> inner_rotation_volume.center = (0, 0, 0) * fl.u.m
    >>> inner_rotation_volume.axis = (0, 1, 0)
    >>> inner_rotation = fl.Rotation(
    ...     name="innerRotation",
    ...     volumes=inner_rotation_volume,
    ...     spec= fl.AngleExpression("-2*sin(t)"),
    ...     parent_volume=outer_rotation_volume  # inner rotation is nested in the outer rotation.
    ... )

    ====
    """

    name: Optional[str] = pd.Field("Rotation", description="Name of the `Rotation` model.")
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
    rotating_reference_frame_model: Optional[bool] = pd.Field(
        None,
        description="Flag to specify whether the non-inertial reference frame model is "
        + "to be used for the rotation model. Steady state simulation requires this flag "
        + "to be True for all rotation models.",
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
    :class:`PorousMedium` class for specifying porous media settings.
    For further information please refer to the :ref:`porous media knowledge base <knowledge_base_porousMedia>`.

    Example
    -------

    Define a porous medium model :code:`porous_zone` with the :py:class:`Box` entity.
    The center and size of the `porous_zone` box are (0, 0, 0) * fl.u.m and (0.2, 0.3, 2) * fl.u.m, respectively.
    The axes of the :code:`porous_zone` are set as (0, 1, 0) and (0, 0, 1).

    >>> fl.PorousMedium(
    ...     entities=[
    ...         fl.Box.from_principal_axes(
    ...             name="porous_zone",
    ...             axes=[(0, 1, 0), (0, 0, 1)],
    ...             center=(0, 0, 0) * fl.u.m,
    ...             size=(0.2, 0.3, 2) * fl.u.m,
    ...         )
    ...    ],
    ...    darcy_coefficient=(1e6, 0, 0) / fl.u.m **2,
    ...    forchheimer_coefficient=(1, 0, 0) / fl.u.m,
    ...    volumetric_heat_source=1.0 * fl.u.W/ fl.u.m **3,
    ... )

    Define a porous medium model :code:`porous_zone` with the :code:`volume_mesh["porous_zone"]` volume.
    The axes of entity must be specified to serve as the the principle axes of the porous medium
    material model, and we set the axes of the :code:`porous_zone` as (1, 0, 0) and (0, 1, 0).

    >>> porous_zone = volume_mesh["porous_zone"]
    >>> porous_zone.axes = [(1, 0, 0), (0, 1, 0)]
    >>> porous_medium_model = fl.PorousMedium(
    ...     entities=[porous_zone],
    ...     darcy_coefficient=(1e6, 0, 0) / fl.u.m **2,
    ...     forchheimer_coefficient=(1, 0, 0) / fl.u.m,
    ...     volumetric_heat_source=1.0 * fl.u.W/ fl.u.m **3,
    ... )

    ====
    """

    name: Optional[str] = pd.Field("Porous medium", description="Name of the `PorousMedium` model.")
    type: Literal["PorousMedium"] = pd.Field("PorousMedium", frozen=True)
    entities: EntityList[GenericVolume, Box] = pd.Field(
        alias="volumes",
        description="The entity list for the `PorousMedium` model. "
        + "The entity should be defined by :class:`Box` or zones from the geometry/volume mesh."
        + "The axes of entity must be specified to serve as the the principle axes of the "
        + "porous medium material model.",
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
    volumetric_heat_source: Optional[Union[StringExpression, HeatSourceType]] = pd.Field(
        None, description="The volumetric heat source."
    )

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

    @pd.field_validator("volumetric_heat_source", mode="after")
    @classmethod
    def _validate_volumetric_heat_source_for_liquid(
        cls, value: Optional[Union[StringExpression, HeatSourceType]]
    ):
        """Disable the volumetric_heat_source when liquid operating condition is used"""
        validation_info = get_validation_info()
        if validation_info is None or validation_info.using_liquid_as_material is False:
            return value
        if value is not None:
            raise ValueError(
                "`volumetric_heat_source` cannot be setup under `PorousMedium` when using "
                "liquid as simulation material."
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
