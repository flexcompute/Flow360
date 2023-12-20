"""
Flow360 solver parameters
"""
# pylint: disable=too-many-lines
# pylint: disable=unused-import
from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from typing import (
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    Union,
    get_args,
    get_type_hints,
)

import pydantic as pd
from pydantic import StrictStr
from typing_extensions import Literal

from ...error_messages import unit_system_inconsistent_msg, use_unit_system_msg
from ...exceptions import (
    Flow360ConfigError,
    Flow360NotImplementedError,
    Flow360RuntimeError,
    Flow360ValidationError,
)
from ...log import log
from ...user_config import UserConfig
from ...version import __version__
from ..constants import constants
from ..types import (
    Axis,
    Coordinate,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    Vector,
)
from ..utils import _get_value_or_none, beta_feature
from .conversions import ExtraDimensionedProperty
from .flow360_legacy import (
    LegacyModel,
    get_output_fields,
    try_add_unit,
    try_set,
    try_update,
)
from .flow360_output import (
    AeroacousticOutput,
    AnimationSettings,
    AnimationSettingsExtended,
    IsoSurfaceOutput,
    IsoSurfaceOutputLegacy,
    IsoSurfaces,
    MonitorOutput,
    Monitors,
    ProbeMonitor,
    SliceOutput,
    SliceOutputLegacy,
    Slices,
    SurfaceIntegralMonitor,
    SurfaceOutput,
    SurfaceOutputLegacy,
    Surfaces,
    VolumeOutput,
    VolumeOutputLegacy,
)
from .params_base import (
    DeprecatedAlias,
    Flow360BaseModel,
    Flow360SortableBaseModel,
    _self_named_property_validator,
    flow360_json_encoder,
)
from .physical_properties import _AirModel
from .solvers import (
    HeatEquationSolver,
    HeatEquationSolverLegacy,
    IncompressibleNavierStokesSolver,
    KOmegaSST,
    LinearSolver,
    NavierStokesSolver,
    NavierStokesSolverLegacy,
    NoneSolver,
    SpalartAllmaras,
    TransitionModelSolver,
    TransitionModelSolverLegacy,
    TurbulenceModelSolverLegacy,
    TurbulenceModelSolverTypes,
)
from .unit_system import (
    AngularVelocityType,
    AreaType,
    CGS_unit_system,
    CGSUnitSystem,
    DensityType,
    DimensionedType,
    Flow360UnitSystem,
    ImperialUnitSystem,
    LengthType,
    PressureType,
    SI_unit_system,
    SIUnitSystem,
    TemperatureType,
    TimeType,
    UnitSystem,
    UnitSystemTypes,
    VelocityType,
    ViscosityType,
    flow360_unit_system,
    imperial_unit_system,
    u,
    unit_system_manager,
)
from .validations import _check_duplicate_boundary_name, _check_tri_quad_boundaries

BoundaryVelocityType = Union[VelocityType.Vector, Tuple[StrictStr, StrictStr, StrictStr]]
BoundaryAxisType = Union[Axis, Tuple[StrictStr, StrictStr, StrictStr]]


# pylint: disable=invalid-name
def get_time_non_dim_unit(mesh_unit_length, C_inf, extra_msg=""):
    """
    returns time non-dimensionalisation
    """

    if mesh_unit_length is None or C_inf is None:
        required = ["mesh_unit", "mesh_unit_length"]
        raise Flow360ConfigError(f"You need to provide one of {required} AND C_inf {extra_msg}")
    return mesh_unit_length / C_inf


def get_length_non_dim_unit(mesh_unit_length, extra_msg=""):
    """
    returns length non-dimensionalisation
    """
    if mesh_unit_length is None:
        required = ["mesh_unit", "mesh_unit_length"]
        raise Flow360ConfigError(f"You need to provide one of {required} {extra_msg}")
    return mesh_unit_length


class MeshBoundary(Flow360BaseModel):
    """Mesh boundary"""

    no_slip_walls: Union[List[str], List[int]] = pd.Field(alias="noSlipWalls")


class Boundary(Flow360BaseModel, metaclass=ABCMeta):
    """Basic Boundary class"""

    type: str
    name: Optional[str] = pd.Field(
        None, title="Name", description="Optional unique name for boundary."
    )


class NoSlipWall(Boundary):
    """No slip wall boundary"""

    type: Literal["NoSlipWall"] = pd.Field("NoSlipWall", const=True)
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")


class SlipWall(Boundary):
    """Slip wall boundary"""

    type: Literal["SlipWall"] = pd.Field("SlipWall", const=True)


class FreestreamBoundary(Boundary):
    """Freestream boundary"""

    type: Literal["Freestream"] = pd.Field("Freestream", const=True)
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")


class IsothermalWall(Boundary):
    """IsothermalWall boundary"""

    type: Literal["IsothermalWall"] = pd.Field("IsothermalWall", const=True)
    temperature: Union[PositiveFloat, StrictStr] = pd.Field(alias="Temperature")
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")


class HeatFluxWall(Boundary):
    """:class:`HeatFluxWall` class for specifying heat flux wall boundaries

    Parameters
    ----------
    heatFlux : float
        Heat flux at the wall.

    velocity: BoundaryVelocityType
        (Optional) Velocity of the wall. If not specified, the boundary is stationary.

    Returns
    -------
    :class:`HeatFluxWall`
        An instance of the component class HeatFluxWall.

    Example
    -------
    >>> heatFluxWall = HeatFluxWall(heatFlux=-0.01, velocity=(0, 0, 0))
    """

    type: Literal["HeatFluxWall"] = pd.Field("HeatFluxWall", const=True)
    heat_flux: Union[float, StrictStr] = pd.Field(alias="heatFlux")
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="velocity")


class SubsonicOutflowPressure(Boundary):
    """SubsonicOutflowPressure boundary"""

    type: Literal["SubsonicOutflowPressure"] = pd.Field("SubsonicOutflowPressure", const=True)
    static_pressure_ratio: PositiveFloat = pd.Field(alias="staticPressureRatio")


class SubsonicOutflowMach(Boundary):
    """SubsonicOutflowMach boundary"""

    type: Literal["SubsonicOutflowMach"] = pd.Field("SubsonicOutflowMach", const=True)
    Mach: PositiveFloat = pd.Field(alias="MachNumber")


class SubsonicInflow(Boundary):
    """SubsonicInflow boundary"""

    type: Literal["SubsonicInflow"] = pd.Field("SubsonicInflow", const=True)
    total_pressure_ratio: PositiveFloat = pd.Field(alias="totalPressureRatio")
    total_temperature_ratio: PositiveFloat = pd.Field(alias="totalTemperatureRatio")
    ramp_steps: Optional[PositiveInt] = pd.Field(alias="rampSteps")
    velocity_direction: Optional[BoundaryVelocityType] = pd.Field(alias="velocityDirection")


class SupersonicInflow(Boundary):
    """:class:`SupersonicInflow` class for specifying the full fluid state at supersonic inflow boundaries

    Parameters
    ----------
    total_temperature_ratio : PositiveFloat
        Ratio of total temperature to static temperature at the inlet.

    total_pressure_ratio: PositiveFloat
        Ratio of the total pressure to static pressure at the inlet.

    static_pressure_ratio: PositiveFloat
        Ratio of the inlet static pressure to the freestream static pressure. Default freestream static pressure in
        Flow360 = 1.0/gamma.

    velocity_direction: BoundaryAxisType
        (Optional) 3-array of either float values or string expressions. Unit vector which specifies the direction
        of the incoming flow. If not specified, the boundary patch normal is used to specify direction.

    Returns
    -------
    :class:`SupersonicInflow`
        An instance of the component class SupersonicInflow.

    Example
    -------
    >>> supersonicInflow = SupersonicInflow(totalTemperatureRatio=2.1, totalPressureRatio=3.0, staticPressureRatio=1.2)
    """

    type: Literal["SupersonicInflow"] = pd.Field("SupersonicInflow", const=True)
    total_temperature_ratio: PositiveFloat = pd.Field(
        alias="totalTemperatureRatio", supported_solver_version="release-23.3.2.0gt"
    )
    total_pressure_ratio: PositiveFloat = pd.Field(
        alias="totalPressureRatio", supported_solver_version="release-23.3.2.0gt"
    )
    static_pressure_ratio: PositiveFloat = pd.Field(
        alias="staticPressureRatio", supported_solver_version="release-23.3.2.0gt"
    )
    velocity_direction: Optional[BoundaryAxisType] = pd.Field(
        alias="velocityDirection", supported_solver_version="release-23.3.2.0gt"
    )


class SlidingInterfaceBoundary(Boundary):
    """:class: `SlidingInterface` boundary"""

    type: Literal["SlidingInterface"] = pd.Field("SlidingInterface", const=True)


class WallFunction(Boundary):
    """:class: `WallFunction` boundary"""

    type: Literal["WallFunction"] = pd.Field("WallFunction", const=True)


class MassInflow(Boundary):
    """:class: `MassInflow` boundary"""

    type: Literal["MassInflow"] = pd.Field("MassInflow", const=True)
    mass_flow_rate: PositiveFloat = pd.Field(alias="massFlowRate")
    ramp_steps: Optional[PositiveInt] = pd.Field(alias="rampSteps")


class MassOutflow(Boundary):
    """:class: `MassOutflow` boundary"""

    type: Literal["MassOutflow"] = pd.Field("MassOutflow", const=True)
    mass_flow_rate: PositiveFloat = pd.Field(alias="massFlowRate")
    ramp_steps: Optional[PositiveInt] = pd.Field(alias="rampSteps")


class SolidIsothermalWall(Boundary):
    """:class: `SolidIsothermalWall` boundary"""

    type: Literal["SolidIsothermalWall"] = pd.Field("SolidIsothermalWall", const=True)
    temperature: Union[PositiveFloat, StrictStr] = pd.Field(alias="Temperature")


class SolidAdiabaticWall(Boundary):
    """:class: `SolidAdiabaticWall` boundary"""

    type: Literal["SolidAdiabaticWall"] = pd.Field("SolidAdiabaticWall", const=True)


class TranslationallyPeriodic(Boundary):
    """:class: `TranslationallyPeriodic` boundary"""

    type: Literal["TranslationallyPeriodic"] = pd.Field("TranslationallyPeriodic", const=True)
    paired_patch_name: Optional[str] = pd.Field(alias="pairedPatchName")
    translation_vector: Optional[Vector] = pd.Field(alias="translationVector")


class RotationallyPeriodic(Boundary):
    """:class: `RotationallyPeriodic` boundary"""

    type: Literal["RotationallyPeriodic"] = pd.Field("RotationallyPeriodic", const=True)
    paired_patch_name: Optional[str] = pd.Field(alias="pairedPatchName")
    axis_of_rotation: Optional[Vector] = pd.Field(alias="axisOfRotation")
    theta_radians: Optional[float] = pd.Field(alias="thetaRadians")


BoundaryType = Union[
    NoSlipWall,
    SlipWall,
    FreestreamBoundary,
    IsothermalWall,
    HeatFluxWall,
    SubsonicOutflowPressure,
    SubsonicOutflowMach,
    SubsonicInflow,
    SupersonicInflow,
    SlidingInterfaceBoundary,
    WallFunction,
    MassInflow,
    MassOutflow,
    SolidIsothermalWall,
    SolidAdiabaticWall,
    TranslationallyPeriodic,
    RotationallyPeriodic,
]


class ForcePerArea(Flow360BaseModel):
    """:class:`ForcePerArea` class for setting up force per area for Actuator Disk

    Parameters
    ----------
    radius : Coordinate
        Radius of the sampled locations in grid unit

    thrust : Axis
        Force per area in the axial direction, positive means the axial force follows the same direction as axisThrust.
        It is non-dimensional: trustPerArea[SI=N/m2]/rho_inf/C_inf^2

    circumferential : PositiveFloat
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
    """

    radius: List[float]
    thrust: List[float]
    circumferential: List[float]

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def check_len(cls, values):
        """
        root validator
        """
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
    """:class:`ActuatorDisk` class for setting up an Actuator Disk

    Parameters
    ----------
    center : Coordinate
        Coordinate of center of ActuatorDisk, eg (0, 0, 0)

    axis_thrust : Axis
        direction of thrust, it is a unit vector

    thickness : PositiveFloat
        Thickness of Actuator Disk in mesh units

    force_per_area : :class:`ForcePerArea`
        Force per Area data for actuator disk. See ActuatorDisk.ForcePerArea for details

    Returns
    -------
    :class:`ActuatorDisk`
        An instance of the component class ActuatorDisk.

    Example
    -------
    >>> ad = ActuatorDisk(center=(0, 0, 0), axis_thrust=(0, 0, 1), thickness=20,
    ... force_per_area=ForcePerArea(...))
    """

    center: Coordinate
    axis_thrust: Axis = pd.Field(alias="axisThrust", displayed="Axis thrust")
    thickness: PositiveFloat
    force_per_area: ForcePerArea = pd.Field(alias="forcePerArea", displayed="Force per area")

    @classmethod
    def _schema_get_widgets(cls) -> dict[str, str]:
        # Return widget paths for the UI schema
        return {"center": "vector3", "axisThrust": "vector3"}


class SlidingInterface(Flow360BaseModel):
    """:class:`SlidingInterface` class for setting up sliding interface

    Parameters
    ----------
    center : Coordinate
        Coordinate representing the origin of rotation, eg. (0, 0, 0)

    axis : Axis
        Axis of rotation, eg. (0, 0, 1)

    stationary_patches : List[str]
        A list of static patch names of an interface

    rotating_patches : List[str]
        A list of dynamic patch names of an interface

    volume_name : Union[str, int, List[str], List[int]]
        A list of dynamic volume zones related to the above {omega, centerOfRotation, axisOfRotation}

    name: str, optional
        Name of slidingInterface

    parent_volume_name : str, optional
        Name of the volume zone that the rotating reference frame is contained in, used to compute the acceleration in
        the nested rotating reference frame

    theta_radians : str, optional
        Expression for rotation angle (in radians) as a function of time

    theta_degrees : str, optional
        Expression for rotation angle (in degrees) as a function of time

    omega : Union[float, Omega], optional
        Nondimensional rotating speed, radians/nondim-unit-time

    omega_radians
        Nondimensional rotating speed, radians/nondim-unit-time

    omega_degrees
        Nondimensional rotating speed, degrees/nondim-unit-time

    is_dynamic
        Whether rotation of this interface is dictated by userDefinedDynamics


    Returns
    -------
    :class:`SlidingInterface`
        An instance of the component class SlidingInterface.

    Example
    -------
    >>> si = SlidingInterface(
            center=(0, 0, 0),
            axis=(0, 0, 1),
            stationary_patches=['patch1'],
            rotating_patches=['patch2'],
            volume_name='volume1',
            omega=1
        )
    """

    center: Coordinate = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")
    stationary_patches: List[str] = pd.Field(alias="stationaryPatches")
    rotating_patches: List[str] = pd.Field(alias="rotatingPatches")
    volume_name: Union[str, int, List[str], List[int]] = pd.Field(alias="volumeName")
    parent_volume_name: Optional[str] = pd.Field(alias="parentVolumeName")
    name: Optional[str] = pd.Field(alias="interfaceName")
    theta_radians: Optional[str] = pd.Field(alias="thetaRadians")
    theta_degrees: Optional[str] = pd.Field(alias="thetaDegrees")
    omega_radians: Optional[float] = pd.Field(alias="omegaRadians")
    omega_degrees: Optional[float] = pd.Field(alias="omegaDegrees")
    is_dynamic: Optional[bool] = pd.Field(alias="isDynamic")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_one_of = [
            "omega_radians",
            "omega_degrees",
            "theta_radians",
            "theta_degrees",
            "is_dynamic",
        ]

    @classmethod
    def _schema_get_widgets(cls) -> dict[str, str]:
        # Return widget paths for the UI schema
        return {"centerOfRotation": "vector3"}


class MeshSlidingInterface(Flow360BaseModel):
    """
    Sliding interface component
    """

    stationary_patches: List[str] = pd.Field(alias="stationaryPatches")
    rotating_patches: List[str] = pd.Field(alias="rotatingPatches")
    axis: Axis = pd.Field(alias="axisOfRotation")
    center: Coordinate = pd.Field(alias="centerOfRotation")

    @classmethod
    def from_case_sliding_interface(cls, si: SlidingInterface):
        """
        create mesh sliding interface (for Flow360Mesh.json) from case params SlidingInterface
        """
        return cls(
            stationary_patches=si.stationary_patches,
            rotating_patches=si.rotating_patches,
            axis=si.axis,
            center=si.center,
        )


class RampCFL(Flow360BaseModel):
    """
    Ramp CFL for time stepping component
    """

    type: Literal["Ramp"] = pd.Field("Ramp", const=True)
    initial: Optional[PositiveFloat] = pd.Field()
    final: Optional[PositiveFloat] = pd.Field()
    ramp_steps: Optional[int] = pd.Field(alias="rampSteps")
    randomizer: Optional[Dict] = pd.Field()

    @classmethod
    def default_steady(cls):
        """
        returns default steady CFL settings
        """
        return cls(initial=5, final=200, ramp_steps=40)

    @classmethod
    def default_unsteady(cls):
        """
        returns default unsteady CFL settings
        """
        return cls(initial=1, final=1e6, ramp_steps=30)


class AdaptiveCFL(Flow360BaseModel):
    """
    Adaptive CFL for time stepping component
    """

    type: Literal["Adaptive"] = pd.Field("Adaptive", const=True)
    min: Optional[PositiveFloat] = pd.Field(default=0.1)
    max: Optional[PositiveFloat] = pd.Field(default=10000)
    max_relative_change: Optional[PositiveFloat] = pd.Field(alias="maxRelativeChange", default=1)
    convergence_limiting_factor: Optional[PositiveFloat] = pd.Field(
        alias="convergenceLimitingFactor", default=0.25
    )


# pylint: disable=E0213
class BaseTimeStepping(Flow360BaseModel, metaclass=ABCMeta):
    """
    Base class for time stepping component
    """

    max_pseudo_steps: Optional[pd.conint(gt=0, le=100000)] = pd.Field(2000, alias="maxPseudoSteps")
    CFL: Optional[Union[RampCFL, AdaptiveCFL]] = pd.Field()

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> BaseTimeStepping:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [DeprecatedAlias(name="physical_steps", deprecated="maxPhysicalSteps")]


class SteadyTimeStepping(BaseTimeStepping):
    """
    Steady time stepping component
    """

    model_type: Literal["Steady"] = pd.Field("Steady", alias="modelType", const=True)
    physical_steps: Literal[1] = pd.Field(1, alias="physicalSteps", const=True)
    time_step_size: Literal["inf"] = pd.Field("inf", alias="timeStepSize", const=True)


class UnsteadyTimeStepping(BaseTimeStepping):
    """
    Unsteady time stepping component
    """

    model_type: Literal["Unsteady"] = pd.Field("Unsteady", alias="modelType", const=True)
    physical_steps: Optional[PositiveInt] = pd.Field(alias="physicalSteps")
    time_step_size: Optional[TimeType.Positive] = pd.Field(alias="timeStepSize")


TimeStepping = Union[SteadyTimeStepping, UnsteadyTimeStepping]


class _GenericBoundaryWrapper(Flow360BaseModel):
    v: BoundaryType


class Boundaries(Flow360SortableBaseModel):
    """:class:`Boundaries` class for setting up Boundaries

    Parameters
    ----------
    <boundary_name> : BoundaryType
        Supported boundary types: Union[NoSlipWall, SlipWall, FreestreamBoundary, IsothermalWall, HeatFluxWall,
                                        SubsonicOutflowPressure, SubsonicOutflowMach, SubsonicInflow,
                                        SupersonicInflow, SlidingInterfaceBoundary, WallFunction,
                                        MassInflow, MassOutflow, SolidIsothermalWall, SolidAdiabaticWall]

    Returns
    -------
    :class:`Boundaries`
        An instance of the component class Boundaries.

    Example
    -------
    >>> boundaries = Boundaries(
            wing=NoSlipWall(),
            symmetry=SlipWall(),
            freestream=FreestreamBoundary()
        )
    """

    @classmethod
    def get_subtypes(cls) -> list:
        return list(get_args(_GenericBoundaryWrapper.__fields__["v"].type_))

    @pd.root_validator(pre=True)
    def validate_boundary(cls, values):
        """Validator for boundary list section

        Raises
        ------
        ValidationError
            When boundary is incorrect
        """
        return _self_named_property_validator(
            values, _GenericBoundaryWrapper, msg="is not any of supported boundary types."
        )

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> Boundaries:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)

    @classmethod
    def _schema_get_widgets(cls) -> dict[str, str]:
        # Return widget paths for the UI schema
        return {
            "additionalProperties/velocity/value": "vector3",
            "additionalProperties/Velocity/value": "vector3",
            "additionalProperties/velocityDirection/value": "vector3",
            "additionalProperties/translationVector": "vector3",
            "additionalProperties/axisOfRotation": "vector3",
        }


class VolumeZoneBase(Flow360BaseModel, metaclass=ABCMeta):
    """Basic Boundary class"""

    model_type: str = pd.Field(alias="modelType")


class InitialConditionHeatTransfer(Flow360BaseModel):
    """InitialConditionHeatTransfer"""

    T_solid: Union[PositiveFloat, StrictStr] = pd.Field()


class HeatTransferVolumeZone(VolumeZoneBase):
    """HeatTransferVolumeZone type"""

    model_type = pd.Field("HeatTransfer", alias="modelType", const=True)
    thermal_conductivity: PositiveFloat = pd.Field(alias="thermalConductivity")
    volumetric_heat_source: Optional[Union[NonNegativeFloat, StrictStr]] = pd.Field(
        alias="volumetricHeatSource"
    )
    heat_capacity: Optional[PositiveFloat] = pd.Field(alias="heatCapacity")
    initial_condition: Optional[InitialConditionHeatTransfer] = pd.Field(alias="initialCondition")


class ReferenceFrameDynamic(Flow360BaseModel):
    """:class:`ReferenceFrameDynamic` class for setting up dynamic reference frame

    Parameters
    ----------
    center : LengthType.Point
        Coordinate representing the origin of rotation, eg. (0, 0, 0)

    axis : Axis
        Axis of rotation, eg. (0, 0, 1)

    Returns
    -------
    :class:`ReferenceFrameDynamic`
        An instance of the component class ReferenceFrameDynamic.

    Example
    -------
    >>> rf = ReferenceFrameDynamic(
            center=(0, 0, 0),
            axis=(0, 0, 1),
        )
    """

    center: LengthType.Point = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")
    is_dynamic: bool = pd.Field(True, alias="isDynamic", const=True)


class ReferenceFrameExpression(Flow360BaseModel):
    """:class:`ReferenceFrameExpression` class for setting up reference frame using expression

    Parameters
    ----------
    center : Coordinate
        Coordinate representing the origin of rotation, eg. (0, 0, 0)

    axis : Axis
        Axis of rotation, eg. (0, 0, 1)

    parent_volume_name : str, optional
        Name of the volume zone that the rotating reference frame is contained in, used to compute the acceleration in
        the nested rotating reference frame

    theta_radians : str, optional
        Expression for rotation angle (in radians) as a function of time

    theta_degrees : str, optional
        Expression for rotation angle (in degrees) as a function of time


    Returns
    -------
    :class:`ReferenceFrameExpression`
        An instance of the component class ReferenceFrame.

    Example
    -------
    >>> rf = ReferenceFrameExpression(
            center=(0, 0, 0),
            axis=(0, 0, 1),
            theta_radians="1 * t"
        )
    """

    theta_radians: Optional[str] = pd.Field(alias="thetaRadians")
    theta_degrees: Optional[str] = pd.Field(alias="thetaDegrees")
    center: LengthType.Point = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_one_of = [
            "theta_radians",
            "theta_degrees",
        ]


class ReferenceFrameOmegaRadians(Flow360BaseModel):
    """:class:`ReferenceFrameOmegaRadians` class for setting up reference frame

    Parameters
    ----------
    center : Coordinate
        Coordinate representing the origin of rotation, eg. (0, 0, 0)

    axis : Axis
        Axis of rotation, eg. (0, 0, 1)

    omega_radians: float
        Nondimensional rotating speed, radians/nondim-unit-time


    Returns
    -------
    :class:`ReferenceFrameOmegaRadians`
        An instance of the component class ReferenceFrameOmegaRadians.

    """

    omega_radians: float = pd.Field(alias="omegaRadians")
    center: LengthType.Point = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> ReferenceFrameOmegaRadians:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)


class ReferenceFrameOmegaDegrees(Flow360BaseModel):
    """:class:`ReferenceFrameOmegaDegrees` class for setting up reference frame

    Parameters
    ----------
    center : Coordinate
        Coordinate representing the origin of rotation, eg. (0, 0, 0)

    axis : Axis
        Axis of rotation, eg. (0, 0, 1)

    omega_degrees: AngularVelocityType
        Nondimensional rotating speed, radians/nondim-unit-time


    Returns
    -------
    :class:`ReferenceFrameOmegaDegrees`
        An instance of the component class ReferenceFrameOmegaDegrees.

    """

    omega_degrees: float = pd.Field(alias="omegaDegrees")
    center: LengthType.Point = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> ReferenceFrameOmegaDegrees:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)


class ReferenceFrame(Flow360BaseModel):
    """:class:`ReferenceFrame` class for setting up reference frame

    Parameters
    ----------
    center : Coordinate
        Coordinate representing the origin of rotation, eg. (0, 0, 0)

    axis : Axis
        Axis of rotation, eg. (0, 0, 1)

    omega: AngularVelocityType
        Rotating speed, for example radians / s


    Returns
    -------
    :class:`ReferenceFrame`
        An instance of the component class ReferenceFrame.

    Example
    -------
    >>> rf = ReferenceFrame(
            center=(0, 0, 0),
            axis=(0, 0, 1),
            omega=1 * u.rad / u.s
        )
    """

    omega: AngularVelocityType = pd.Field()
    center: LengthType.Point = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> ReferenceFrameOmegaRadians:
        """
        returns configuration object in flow360 units system
        """

        solver_values = self._convert_dimensions_to_solver(params, **kwargs)
        omega_radians = solver_values.pop("omega").value
        return ReferenceFrameOmegaRadians(omega_radians=omega_radians, **solver_values)


class FluidDynamicsVolumeZone(VolumeZoneBase):
    """FluidDynamicsVolumeZone type"""

    model_type = pd.Field("FluidDynamics", alias="modelType", const=True)
    reference_frame: Optional[
        Union[
            ReferenceFrame,
            ReferenceFrameOmegaRadians,
            ReferenceFrameExpression,
            ReferenceFrameDynamic,
        ]
    ] = pd.Field(alias="referenceFrame")

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> FluidDynamicsVolumeZone:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)

    @classmethod
    def _schema_get_optional_objects(cls) -> List[str]:
        return ["referenceFrame"]


VolumeZoneType = Union[FluidDynamicsVolumeZone, HeatTransferVolumeZone]


class _GenericVolumeZonesWrapper(Flow360BaseModel):
    v: VolumeZoneType


class VolumeZones(Flow360SortableBaseModel):
    """:class:`VolumeZones` class for setting up volume zones

    Parameters
    ----------
    <zone_name> : Union[FluidDynamicsVolumeZone, HeatTransferVolumeZone]

    Returns
    -------
    :class:`VolumeZones`
        An instance of the component class VolumeZones.

    Example
    -------
    >>> zones = VolumeZones(
            zone1=FluidDynamicsVolumeZone(),
            zone2=HeatTransferVolumeZone(thermal_conductivity=1)
        )
    """

    @classmethod
    def get_subtypes(cls) -> list:
        return list(get_args(_GenericVolumeZonesWrapper.__fields__["v"].type_))

    @pd.root_validator(pre=True)
    def validate_zone(cls, values):
        """Validator for zone list section

        Raises
        ------
        ValidationError
            When zone is incorrect
        """
        return _self_named_property_validator(
            values, _GenericVolumeZonesWrapper, msg="is not any of supported volume zone types."
        )

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> VolumeZones:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)

    @classmethod
    def _schema_get_widgets(cls) -> dict[str, str]:
        # Return widget paths for the UI schema
        return {
            "additionalProperties/referenceFrame/centerOfRotation/value": "vector3",
            "additionalProperties/referenceFrame/axisOfRotation": "vector3",
        }


class Geometry(Flow360BaseModel):
    """
    :class: Geometry component
    """

    ref_area: Optional[AreaType.Positive] = pd.Field(alias="refArea")
    moment_center: Optional[LengthType.Point] = pd.Field(alias="momentCenter")
    moment_length: Optional[LengthType.Moment] = pd.Field(alias="momentLength")
    mesh_unit: Optional[LengthType] = pd.Field(alias="meshUnit")

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> Geometry:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, exclude=["mesh_unit"], **kwargs)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        allow_but_remove = ["meshName", "endianness"]


class FreestreamBase(Flow360BaseModel, metaclass=ABCMeta):
    """
    :class: Freestream component
    """

    model_type: str
    alpha: Optional[float] = pd.Field(alias="alphaAngle", default=0)
    beta: Optional[float] = pd.Field(alias="betaAngle", default=0)
    turbulent_viscosity_ratio: Optional[NonNegativeFloat] = pd.Field(
        alias="turbulentViscosityRatio"
    )


class FreestreamFromMach(FreestreamBase):
    """
    :class: Freestream component using Mach numbers
    """

    model_type: Literal["FromMach"] = pd.Field("FromMach", alias="modelType", const=True)
    Mach: PositiveFloat = pd.Field()
    Mach_ref: Optional[PositiveFloat] = pd.Field(alias="MachRef")
    mu_ref: PositiveFloat = pd.Field(alias="muRef")
    temperature: PositiveFloat = pd.Field(alias="Temperature")

    # pylint: disable=arguments-differ, unused-argument
    def to_solver(self, params: Flow360Params, **kwargs) -> FreestreamFromMach:
        """
        returns configuration object in flow360 units system
        """
        return self


class FreestreamFromMachReynolds(FreestreamBase):
    """
    :class: Freestream component using Mach and Reynolds numbers
    """

    model_type: Literal["FromMachReynolds"] = pd.Field(
        "FromMachReynolds", alias="modelType", const=True
    )
    Mach: PositiveFloat = pd.Field()
    Mach_ref: Optional[PositiveFloat] = pd.Field(alias="MachRef")
    Reynolds: PositiveFloat = pd.Field()
    temperature: PositiveFloat = pd.Field(alias="Temperature")

    # pylint: disable=arguments-differ, unused-argument
    def to_solver(self, params: Flow360Params, **kwargs) -> FreestreamFromMach:
        """
        returns configuration object in flow360 units system
        """
        return self


class ZeroFreestream(FreestreamBase):
    """
    :class: Zero velocity freestream component
    """

    model_type: Literal["ZeroMach"] = pd.Field("ZeroMach", alias="modelType", const=True)
    Mach: Literal[0] = pd.Field(0, const=True)
    Mach_ref: PositiveFloat = pd.Field(alias="MachRef")
    mu_ref: PositiveFloat = pd.Field(alias="muRef")
    temperature: PositiveFloat = pd.Field(alias="Temperature")

    # pylint: disable=arguments-differ, unused-argument
    def to_solver(self, params: Flow360Params, **kwargs) -> ZeroFreestream:
        """
        returns configuration object in flow360 units system
        """
        return self


class FreestreamFromVelocity(FreestreamBase):
    """
    :class: Freestream component using dimensioned velocity
    """

    model_type: Literal["FromVelocity"] = pd.Field("FromVelocity", alias="modelType", const=True)
    velocity: VelocityType.Positive = pd.Field()
    velocity_ref: Optional[VelocityType.Positive] = pd.Field(alias="velocityRef")

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> FreestreamFromMach:
        """
        returns configuration object in flow360 units system
        """

        extra = [
            ExtraDimensionedProperty(
                name="viscosity",
                dependency_list=["fluid_properties"],
                value_factory=lambda: params.fluid_properties.to_fluid_properties().viscosity,
            ),
            ExtraDimensionedProperty(
                name="temperature",
                dependency_list=["fluid_properties"],
                value_factory=lambda: params.fluid_properties.to_fluid_properties().temperature,
            ),
        ]

        solver_values = self._convert_dimensions_to_solver(params, extra=extra, **kwargs)
        mach = solver_values.pop("velocity")
        mach_ref = solver_values.pop("velocity_ref", None)
        mu_ref = solver_values.pop("viscosity")
        temperature = solver_values.pop("temperature").to("K")

        if solver_values.get("model_type") is not None:
            solver_values.pop("model_type")

        if mach_ref is not None:
            mach_ref = mach_ref.v.item()

        return FreestreamFromMach(
            Mach=mach, Mach_ref=mach_ref, temperature=temperature, mu_ref=mu_ref, **solver_values
        )


class ZeroFreestreamFromVelocity(FreestreamBase):
    """
    :class: Zero velocity freestream component using dimensioned velocity
    """

    model_type: Literal["ZeroVelocity"] = pd.Field("ZeroVelocity", alias="modelType", const=True)
    velocity: Literal[0] = pd.Field(0, const=True)
    velocity_ref: VelocityType.Positive = pd.Field(alias="velocityRef")

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> ZeroFreestream:
        """
        returns configuration object in flow360 units system
        """

        extra = [
            ExtraDimensionedProperty(
                name="viscosity",
                dependency_list=["fluid_properties"],
                value_factory=lambda: params.fluid_properties.to_fluid_properties().viscosity,
            ),
            ExtraDimensionedProperty(
                name="temperature",
                dependency_list=["fluid_properties"],
                value_factory=lambda: params.fluid_properties.to_fluid_properties().temperature,
            ),
        ]
        solver_values = self._convert_dimensions_to_solver(params, extra=extra, **kwargs)
        mach = solver_values.pop("velocity")
        mach_ref = solver_values.pop("velocity_ref", None)
        mu_ref = solver_values.pop("viscosity")
        temperature = solver_values.pop("temperature").to("K")

        if solver_values.get("model_type") is not None:
            solver_values.pop("model_type")

        return ZeroFreestream(
            Mach=mach, Mach_ref=mach_ref, temperature=temperature, mu_ref=mu_ref, **solver_values
        )


FreestreamTypes = Union[
    FreestreamFromMach,
    FreestreamFromMachReynolds,
    FreestreamFromVelocity,
    ZeroFreestream,
    ZeroFreestreamFromVelocity,
]


class _FluidProperties(Flow360BaseModel):
    """
    Model representing fluid properties.

    Parameters
    ----------
    temperature : TemperatureType
        Temperature of the fluid.
    pressure : PressureType
        Pressure of the fluid.
    density : DensityType
        Density of the fluid.
    viscosity : ViscosityType
        Viscosity of the fluid.

    """

    temperature: TemperatureType = pd.Field()
    pressure: PressureType = pd.Field()
    density: DensityType = pd.Field()
    viscosity: ViscosityType = pd.Field()

    def to_fluid_properties(self) -> _FluidProperties:
        """returns an instance of _FluidProperties"""

        return self


class AirPressureTemperature(Flow360BaseModel):
    """
    Model representing air properties based on pressure and temperature.

    Parameters
    ----------
    pressure : PressureType
        Pressure of the air.
    temperature : TemperatureType
        Temperature of the air.

    """

    model_type: str = pd.Field("AirPressure", alias="modelType", const=True)
    pressure: PressureType = pd.Field()
    temperature: TemperatureType = pd.Field()

    def to_fluid_properties(self) -> _FluidProperties:
        """Converts the instance to _FluidProperties, incorporating temperature, pressure, density, and viscosity."""

        fluid_properties = _FluidProperties(
            temperature=self.temperature,
            pressure=self.pressure,
            density=_AirModel.density_from_pressure_temperature(self.pressure, self.temperature),
            viscosity=_AirModel.viscosity_from_temperature(self.temperature),
        )
        return fluid_properties

    def speed_of_sound(self) -> VelocityType:
        """Calculates the speed of sound in the air based on the temperature."""
        return _AirModel.speed_of_sound(self.temperature)


class AirDensityTemperature(Flow360BaseModel):
    """
    Model representing air properties based on density and temperature.

    Parameters
    ----------
    temperature : TemperatureType
        Temperature of the air.
    density : DensityType
        Density of the air.

    """

    model_type: str = pd.Field("AirDensity", alias="modelType", const=True)
    temperature: TemperatureType = pd.Field()
    density: DensityType = pd.Field()

    def to_fluid_properties(self) -> _FluidProperties:
        """Converts the instance to _FluidProperties, incorporating temperature, pressure, density, and viscosity."""

        fluid_properties = _FluidProperties(
            temperature=self.temperature,
            pressure=_AirModel.pressure_from_density_temperature(self.density, self.temperature),
            density=self.density,
            viscosity=_AirModel.viscosity_from_temperature(self.temperature),
        )
        return fluid_properties

    def speed_of_sound(self) -> VelocityType:
        """Calculates the speed of sound in the air based on the temperature."""
        return _AirModel.speed_of_sound(self.temperature)


class USstandardAtmosphere(Flow360BaseModel):
    """
    Model representing the U.S. Standard Atmosphere.

    Parameters
    ----------
    altitude : LengthType
        Altitude above sea level.
    temperature_offset : TemperatureType, default: 0
        Offset to the standard temperature.

    """

    altitude: LengthType = pd.Field()
    temperature_offset: TemperatureType = pd.Field(default=0)

    def __init__(self):
        super().__init__()
        raise Flow360NotImplementedError("USstandardAtmosphere not implemented yet.")

    def to_fluid_properties(self) -> _FluidProperties:
        """Converts the instance to _FluidProperties, incorporating temperature, pressure, density, and viscosity."""


# pylint: disable=no-member
air = AirDensityTemperature(temperature=288.15 * u.K, density=1.225 * u.kg / u.m**3)

FluidPropertyTypes = Union[AirDensityTemperature, AirPressureTemperature]


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

    rotation_direction_rule: Optional[Literal["leftHand", "rightHand"]] = pd.Field(
        alias="rotationDirectionRule", displayed="Rotation direction"
    )
    center_of_rotation: Coordinate = pd.Field(
        alias="centerOfRotation", displayed="Center of rotation"
    )
    axis_of_rotation: Axis = pd.Field(alias="axisOfRotation", displayed="Axis of rotation")
    number_of_blades: pd.conint(strict=True, gt=0, le=10) = pd.Field(
        alias="numberOfBlades", displayed="Number of blades"
    )
    radius: LengthType.Positive = pd.Field(alias="radius", displayed="Radius")
    omega: AngularVelocityType.NonNegative = pd.Field(
        alias="omega", displayed="Angular velocity (omega)"
    )
    chord_ref: LengthType.Positive = pd.Field(alias="chordRef", displayed="Reference chord")
    thickness: LengthType.Positive = pd.Field(alias="thickness")
    n_loading_nodes: pd.conint(strict=True, gt=0, le=1000) = pd.Field(
        alias="nLoadingNodes", displayed="Loading nodes"
    )
    blade_line_chord: Optional[LengthType.NonNegative] = pd.Field(
        alias="bladeLineChord", displayed="Blade line chord"
    )
    initial_blade_direction: Optional[Axis] = pd.Field(
        alias="initialBladeDirection", displayed="Initial blade direction"
    )
    tip_gap: Optional[Union[LengthType.NonNegative, Literal["inf"]]] = pd.Field(
        alias="tipGap", displayed="Tip gap"
    )
    mach_numbers: List[NonNegativeFloat] = pd.Field(alias="MachNumbers", displayed="Mach numbers")
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

    @pd.validator("alphas")
    def check_alphas_in_order(cls, alpha):
        """
        check alpha angles are listed in order
        """
        if alpha != sorted(alpha):
            raise ValueError("BET Disk: alphas are not in increasing order")
        return alpha

    @pd.root_validator()
    def check_number_of_sections(cls, values):
        """
        check lengths of sectional radiuses and polars are equal
        """
        sectionalRadiuses = values.get("sectional_radiuses")
        sectionalPolars = values.get("sectional_polars")
        assert len(sectionalRadiuses) == len(sectionalPolars)
        return values

    @classmethod
    def _schema_get_widgets(cls) -> dict[str, str]:
        # Return widget paths for the UI schema
        return {
            "centerOfRotation": "vector3",
            "axisOfRotation": "vector3",
            "initialBladeDirection": "vector3",
        }


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

    @classmethod
    def _schema_get_widgets(cls) -> dict[str, str]:
        # Return widget paths for the UI schema
        return {
            "DarcyCoefficient": "vector3",
            "ForchheimerCoefficient": "vector3",
            "volumeZone/center": "vector3",
            "volumeZone/lengths": "vector3",
            "volumeZone/axes/items": "vector3",
            "volumeZone/windowingLengths": "vector3",
        }


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


# pylint: disable=too-many-instance-attributes
class Flow360Params(Flow360BaseModel):
    """
    Flow360 solver parameters
    """

    unit_system: UnitSystemTypes = pd.Field(alias="unitSystem", mutable=False, discriminator="name")
    version: str = pd.Field(__version__, mutable=False)

    geometry: Optional[Geometry] = pd.Field()
    fluid_properties: Optional[FluidPropertyTypes] = pd.Field(alias="fluidProperties")
    boundaries: Optional[Boundaries] = pd.Field()
    initial_condition: Optional[InitialConditions] = pd.Field(
        alias="initialCondition", discriminator="type"
    )
    time_stepping: Optional[TimeStepping] = pd.Field(
        alias="timeStepping", default=SteadyTimeStepping(), discriminator="model_type"
    )
    navier_stokes_solver: Optional[NavierStokesSolver] = pd.Field(alias="navierStokesSolver")
    turbulence_model_solver: Optional[TurbulenceModelSolverTypes] = pd.Field(
        alias="turbulenceModelSolver", discriminator="model_type"
    )
    transition_model_solver: Optional[TransitionModelSolver] = pd.Field(
        alias="transitionModelSolver"
    )
    heat_equation_solver: Optional[HeatEquationSolver] = pd.Field(alias="heatEquationSolver")
    freestream: Optional[FreestreamTypes] = pd.Field(discriminator="model_type")
    bet_disks: Optional[List[BETDisk]] = pd.Field(alias="BETDisks")
    actuator_disks: Optional[List[ActuatorDisk]] = pd.Field(alias="actuatorDisks")
    porous_media: Optional[List[PorousMedium]] = pd.Field(alias="porousMedia")
    user_defined_dynamics: Optional[List[UserDefinedDynamic]] = pd.Field(
        alias="userDefinedDynamics"
    )
    surface_output: Optional[SurfaceOutput] = pd.Field(alias="surfaceOutput")
    volume_output: Optional[VolumeOutput] = pd.Field(alias="volumeOutput")
    slice_output: Optional[SliceOutput] = pd.Field(alias="sliceOutput")
    iso_surface_output: Optional[IsoSurfaceOutput] = pd.Field(alias="isoSurfaceOutput")
    monitor_output: Optional[MonitorOutput] = pd.Field(alias="monitorOutput")
    volume_zones: Optional[VolumeZones] = pd.Field(alias="volumeZones")
    aeroacoustic_output: Optional[AeroacousticOutput] = pd.Field(alias="aeroacousticOutput")

    def _init_check_unit_system(self, **kwargs):
        if unit_system_manager.current is None:
            raise Flow360RuntimeError(use_unit_system_msg)

        kwarg_unit_system = kwargs.pop("unit_system", kwargs.pop("unitSystem", None))
        if kwarg_unit_system is not None:
            if not isinstance(kwarg_unit_system, UnitSystem):
                kwarg_unit_system = UnitSystem.from_dict(**kwarg_unit_system)
            if kwarg_unit_system != unit_system_manager.current:
                raise Flow360RuntimeError(
                    unit_system_inconsistent_msg(
                        kwarg_unit_system.system_repr(), unit_system_manager.current.system_repr()
                    )
                )

        return kwargs

    def __init__(self, filename: str = None, **kwargs):
        if filename is not None:
            self._init_from_file(filename, **kwargs)
        else:
            kwargs = self._init_check_unit_system(**kwargs)
            super().__init__(unit_system=unit_system_manager.copy_current(), **kwargs)

    @classmethod
    def from_file(cls, filename: str) -> Flow360Params:
        """Loads a :class:`Flow360BaseModel` from .json, or .yaml file.

        Parameters
        ----------
        filename : str
            Full path to the .yaml or .json file to load the :class:`Flow360BaseModel` from.

        Returns
        -------
        :class:`Flow360Params`
            An instance of the component class calling `load`.

        Example
        -------
        >>> simulation = Flow360Params.from_file(filename='folder/sim.json') # doctest: +SKIP
        """
        return cls(filename=filename)

    def _init_from_file(self, filename, **kwargs):
        if unit_system_manager.current is not None:
            raise Flow360RuntimeError(
                "When loading params from file: Flow360Params(filename), unit context must not be used."
            )

        model_dict = self._init_handle_file(filename=filename, **kwargs)

        version = model_dict.get("version")
        unit_system = model_dict.get("unitSystem")
        if version is not None and unit_system is not None:
            if version != __version__:
                raise Flow360NotImplementedError(
                    "No updater flow between versioned cases exists as of now."
                )
            with UnitSystem.from_dict(**unit_system):
                super().__init__(**model_dict)
        else:
            self._init_with_update(model_dict)

    def _init_with_update(self, model_dict):
        legacy = Flow360ParamsLegacy(**model_dict)
        super().__init__(**legacy.update_model().dict())

    def copy(self, update=None, **kwargs) -> Flow360Params:
        if unit_system_manager.current is None:
            with self.unit_system:
                return super().copy(update=update, **kwargs)

        return super().copy(update=update, **kwargs)

    # pylint: disable=arguments-differ
    def to_solver(self) -> Flow360Params:
        """
        returns configuration object in flow360 units system
        """
        if unit_system_manager.current is None:
            with self.unit_system:
                return super().to_solver(self, exclude=["fluid_properties"])
        return super().to_solver(self, exclude=["fluid_properties"])

    def to_flow360_json(self) -> dict:
        """Generate a JSON representation of the model, as required by Flow360

        Returns
        -------
        json
            Returns JSON representation of the model.

        Example
        -------
        >>> params.to_flow360_json() # doctest: +SKIP
        """

        solver_params = self.to_solver()
        solver_params_json = solver_params.json(
            encoder=flow360_json_encoder, exclude=["version", "unit_system"]
        )
        return solver_params_json

    def append(self, params: Flow360Params, overwrite: bool = False):
        if not isinstance(params, Flow360Params):
            raise ValueError("params must be type of Flow360Params")
        super().append(params=params, overwrite=overwrite)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        allow_but_remove = ["runControl", "testControl"]
        include_hash: bool = True

    @pd.root_validator
    def check_consistency_wallFunction_and_SurfaceOutput(cls, values):
        """
        check consistency between wall function usage and surface output
        """
        boundary_types = []
        boundaries = values.get("boundaries")
        if boundaries is not None:
            boundary_types = boundaries.get_subtypes()

        surface_output_fields_root = []
        surface_output = values.get("surfaceOutput")
        if surface_output is not None:
            surface_output_fields_root = surface_output.output_fields
        if (
            "WallFunctionMetric" in surface_output_fields_root
            and WallFunction not in boundary_types
        ):
            raise ValueError(
                "'WallFunctionMetric' in 'surfaceOutput' is only valid for 'WallFunction' boundary types."
            )
        return values

    @pd.root_validator
    def check_consistency_DDES_volumeOutput(cls, values):
        """
        check consistency between DDES usage and volume output
        """
        turbulence_model_solver = values.get("turbulence_model_solver")
        model_type = None
        run_DDES = False
        if turbulence_model_solver is not None:
            model_type = turbulence_model_solver.model_type
            run_DDES = turbulence_model_solver.DDES

        volume_output = values.get("volumeOutput")
        if volume_output is not None and volume_output.output_fields is not None:
            output_fields = volume_output.output_fields
            if "SpalartAllmaras_DDES" in output_fields and not (
                model_type == "SpalartAllmaras" and run_DDES
            ):
                raise ValueError(
                    "SpalartAllmaras_DDES output can only be specified with \
                    SpalartAllmaras turbulence model and DDES turned on"
                )
            if "kOmegaSST_DDES" in output_fields and not (model_type == "kOmegaSST" and run_DDES):
                raise ValueError(
                    "kOmegaSST_DDES output can only be specified with kOmegaSST turbulence model and DDES turned on"
                )
        return values

    @pd.root_validator
    def check_tri_quad_boundaries(cls, values):
        """
        check tri_ and quad_ prefix in boundary names
        """
        return _check_tri_quad_boundaries(values)

    @pd.root_validator
    def check_duplicate_boundary_name(cls, values):
        """
        check duplicated boundary names
        """
        return _check_duplicate_boundary_name(values)


class Flow360MeshParams(Flow360BaseModel):
    """
    Flow360 mesh parameters
    """

    boundaries: MeshBoundary = pd.Field()
    sliding_interfaces: Optional[List[MeshSlidingInterface]] = pd.Field(alias="slidingInterfaces")


class UnvalidatedFlow360Params(Flow360BaseModel):
    """
    Unvalidated parameters
    """

    def __init__(self, filename: str = None, **kwargs):
        if UserConfig.do_validation:
            raise Flow360ConfigError(
                "This is DEV feature. To use it activate by: fl.UserConfig.disable_validation()."
            )
        log.warning("This is DEV feature, use it only when you know what you are doing.")
        super().__init__(filename, **kwargs)

    def to_flow360_json(self) -> dict:
        """Generate a JSON representation of the model"""

        return self.json(encoder=flow360_json_encoder)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        extra = "allow"


# Legacy models for Flow360 updater, do not expose


class BETDiskLegacy(BETDisk, LegacyModel):
    """:class:`BETDiskLegacy` class"""

    volume_name: Optional[str] = pd.Field(alias="volumeName")

    def update_model(self):
        model = {
            "rotationDirectionRule": self.rotation_direction_rule,
            "centerOfRotation": self.center_of_rotation,
            "axisOfRotation": self.axis_of_rotation,
            "numberOfBlades": self.number_of_blades,
            "radius": self.radius,
            "chordRef": self.chord_ref,
            "thickness": self.thickness,
            "nLoadingNodes": self.n_loading_nodes,
            "bladeLineChord": self.blade_line_chord,
            "initialBladeDirection": self.initial_blade_direction,
            "tipGap": self.tip_gap,
            "machNumbers": self.mach_numbers,
            "reynoldsNumbers": self.reynolds_numbers,
            "alphas": self.alphas,
            "twists": self.twists,
            "chords": self.chords,
            "sectionalPolars": self.sectional_polars,
            "sectionalRadiuses": self.sectional_radiuses,
        }

        return BETDisk.parse_obj(model)


class GeometryLegacy(Geometry, LegacyModel):
    """:class: `GeometryLegacy` class"""

    ref_area: Optional[float] = pd.Field(alias="refArea")
    moment_center: Optional[Coordinate] = pd.Field(alias="momentCenter")
    moment_length: Optional[Coordinate] = pd.Field(alias="momentLength")

    def update_model(self) -> Flow360BaseModel:
        model = {
            "momentCenter": self.moment_center,
            "momentLength": self.moment_length,
            "refArea": self.ref_area,
        }
        if self.comments is not None and self.comments.get("meshUnit") is not None:
            unit = u.unyt_quantity(1, self.comments["meshUnit"])
            model["meshUnit"] = unit
            try_add_unit(model, "momentCenter", model["meshUnit"])
            try_add_unit(model, "momentLength", model["meshUnit"])
            try_add_unit(model, "refArea", model["meshUnit"] ** 2)

        return Geometry.parse_obj(model)


class FreestreamLegacy(LegacyModel):
    """:class: `FreestreamLegacy` class"""

    Reynolds: Optional[PositiveFloat] = pd.Field()
    Mach: Optional[NonNegativeFloat] = pd.Field()
    Mach_Ref: Optional[PositiveFloat] = pd.Field(alias="MachRef")
    mu_ref: Optional[PositiveFloat] = pd.Field(alias="muRef")
    temperature: PositiveFloat = pd.Field(alias="Temperature")
    alpha: Optional[float] = pd.Field(alias="alphaAngle")
    beta: Optional[float] = pd.Field(alias="betaAngle", default=0)
    turbulent_viscosity_ratio: Optional[NonNegativeFloat] = pd.Field(
        alias="turbulentViscosityRatio"
    )

    def update_model(self) -> Flow360BaseModel:
        class _FreestreamTempModel(pd.BaseModel):
            """Helper class used to create
            the correct freestream from dict data"""

            field: FreestreamTypes = pd.Field(discriminator="model_type")

        model = {
            "field": {
                "alphaAngle": self.alpha,
                "betaAngle": self.beta,
                "turbulentViscosityRatio": self.turbulent_viscosity_ratio,
            }
        }

        # Set velocity
        if self.comments is not None:
            if self.comments.get("freestreamMeterPerSecond") is not None:
                # pylint: disable=no-member
                velocity = self.comments["freestreamMeterPerSecond"] * u.m / u.s
                try_set(model["field"], "velocity", velocity)
            elif (
                self.comments.get("speedOfSoundMeterPerSecond") is not None
                and self.Mach is not None
            ):
                # pylint: disable=no-member
                velocity = self.comments["speedOfSoundMeterPerSecond"] * self.Mach * u.m / u.s
                try_set(model["field"], "velocity", velocity)

            # Set velocity_ref
            if model["field"].get("velocity"):
                if model["field"].get("velocity") == 0:
                    model["field"]["modelType"] = "ZeroVelocity"
                else:
                    model["field"]["modelType"] = "FromVelocity"

                if (
                    self.comments.get("speedOfSoundMeterPerSecond") is not None
                    and self.Mach_Ref is not None
                ):
                    velocity_ref = (
                        # pylint: disable=no-member
                        self.comments["speedOfSoundMeterPerSecond"]
                        * self.Mach_Ref
                        * u.m
                        / u.s
                    )
                    try_set(model["field"], "velocityRef", velocity_ref)
                else:
                    model["field"]["velocityRef"] = None
        else:
            try_set(model["field"], "Reynolds", self.Reynolds)
            try_set(model["field"], "muRef", self.mu_ref)
            try_set(model["field"], "temperature", self.temperature)
            try_set(model["field"], "Mach", self.Mach)
            try_set(model["field"], "MachRef", self.Mach_Ref)

            if self.Mach is not None and self.Mach == 0:
                model["field"]["modelType"] = "ZeroMach"
            elif self.Reynolds is not None:
                model["field"]["modelType"] = "FromMachReynolds"
            else:
                model["field"]["modelType"] = "FromMach"

        return _FreestreamTempModel.parse_obj(model).field

    def extract_fluid_properties(self) -> Optional[Flow360BaseModel]:
        """Extract fluid properties from the freestream comments"""

        class _FluidPropertiesTempModel(pd.BaseModel):
            """Helper class used to create
            the correct fluid properties from dict data"""

            field: FluidPropertyTypes = pd.Field()

        model = {"field": {}}

        # pylint: disable=no-member
        try_set(model["field"], "temperature", self.temperature * u.K)

        if self.comments is not None and self.comments.get("densityKgPerCubicMeter"):
            # pylint: disable=no-member
            density = self.comments["densityKgPerCubicMeter"] * u.kg / u.m**3
            try_set(model["field"], "density", density)
        else:
            return None

        return _FluidPropertiesTempModel.parse_obj(model).field


class TimeSteppingLegacy(BaseTimeStepping, LegacyModel):
    """:class: `TimeSteppingLegacy` class"""

    physical_steps: Optional[PositiveInt] = pd.Field(alias="physicalSteps")
    time_step_size: Optional[Union[Literal["inf"], PositiveFloat]] = pd.Field(
        alias="timeStepSize", default="inf"
    )

    def update_model(self) -> Flow360BaseModel:
        class _TimeSteppingTempModel(pd.BaseModel):
            """Helper class used to create
            the correct time stepping from dict data"""

            field: TimeStepping = pd.Field(discriminator="model_type")

        model = {
            "field": {
                "CFL": self.CFL,
                "physicalSteps": self.physical_steps,
                "maxPseudoSteps": self.max_pseudo_steps,
                "timeStepSize": self.time_step_size,
            }
        }

        if (
            model["field"]["timeStepSize"] != "inf"
            and self.comments.get("timeStepSizeInSeconds") is not None
        ):
            step_unit = u.unyt_quantity(self.comments["timeStepSizeInSeconds"], "s")
            try_add_unit(model, "timeStepSize", step_unit)

        if model["field"]["timeStepSize"] == "inf" and model["field"]["physicalSteps"] == 1:
            model["field"]["modelType"] = "Steady"
        else:
            model["field"]["modelType"] = "Unsteady"

        return _TimeSteppingTempModel.parse_obj(model).field


class SlidingInterfaceLegacy(SlidingInterface, LegacyModel):
    """:class:`SlidingInterfaceLegacy` class"""

    omega: Optional[float] = pd.Field()

    def update_model(self) -> Flow360BaseModel:
        model = {
            "modelType": "FluidDynamics",
            "referenceFrame": {
                "axis": self.axis,
                # pylint: disable=no-member
                "center": self.center * u.m,
            },
        }

        try_set(model["referenceFrame"], "isDynamic", self.is_dynamic)
        try_set(model["referenceFrame"], "omegaRadians", self.omega)
        try_set(model["referenceFrame"], "omegaRadians", self.omega_radians)
        try_set(model["referenceFrame"], "omegaDegrees", self.omega_degrees)
        try_set(model["referenceFrame"], "thetaRadians", self.theta_radians)
        try_set(model["referenceFrame"], "thetaDegrees", self.theta_degrees)

        if self.comments.get("rpm") is not None:
            # pylint: disable=no-member
            omega = self.comments["rpm"] * u.rpm
            try_set(model["referenceFrame"], "omega", omega)
            if model["referenceFrame"].get("omegaRadians") is not None:
                del model["referenceFrame"]["omegaRadians"]
            if model["referenceFrame"].get("omegaDegrees") is not None:
                del model["referenceFrame"]["omegaDegrees"]

        return FluidDynamicsVolumeZone.parse_obj(model)


class Flow360ParamsLegacy(LegacyModel):
    """:class: `Flow360ParamsLegacy` class"""

    geometry: Optional[GeometryLegacy] = pd.Field()
    freestream: Optional[FreestreamLegacy] = pd.Field()
    time_stepping: Optional[TimeSteppingLegacy] = pd.Field(alias="timeStepping")
    navier_stokes_solver: Optional[NavierStokesSolverLegacy] = pd.Field(alias="navierStokesSolver")
    turbulence_model_solver: Optional[TurbulenceModelSolverLegacy] = pd.Field(
        alias="turbulenceModelSolver"
    )
    transition_model_solver: Optional[TransitionModelSolverLegacy] = pd.Field(
        alias="transitionModelSolver"
    )
    heat_equation_solver: Optional[HeatEquationSolverLegacy] = pd.Field(alias="heatEquationSolver")
    bet_disks: Optional[List[BETDiskLegacy]] = pd.Field(alias="BETDisks")
    sliding_interfaces: Optional[List[SlidingInterfaceLegacy]] = pd.Field(alias="slidingInterfaces")
    surface_output: Optional[SurfaceOutputLegacy] = pd.Field(alias="surfaceOutput")
    volume_output: Optional[VolumeOutputLegacy] = pd.Field(alias="volumeOutput")
    slice_output: Optional[SliceOutputLegacy] = pd.Field(alias="sliceOutput")
    iso_surface_output: Optional[IsoSurfaceOutputLegacy] = pd.Field(alias="isoSurfaceOutput")
    boundaries: Optional[Boundaries] = pd.Field()
    initial_condition: Optional[InitialConditions] = pd.Field(
        alias="initialCondition", discriminator="type"
    )
    actuator_disks: Optional[List[ActuatorDisk]] = pd.Field(alias="actuatorDisks")
    porous_media: Optional[List[PorousMedium]] = pd.Field(alias="porousMedia")
    user_defined_dynamics: Optional[List[UserDefinedDynamic]] = pd.Field(
        alias="userDefinedDynamics"
    )
    monitor_output: Optional[MonitorOutput] = pd.Field(alias="monitorOutput")
    volume_zones: Optional[VolumeZones] = pd.Field(alias="volumeZones")
    aeroacoustic_output: Optional[AeroacousticOutput] = pd.Field(alias="aeroacousticOutput")

    def _has_key(self, target, model_dict: dict):
        for key, value in model_dict.items():
            if key == target:
                return True
            if isinstance(value, dict):
                if self._has_key(target, value):
                    return True
        return False

    def _is_web_ui_generated(self, fluid_properties, freestream):
        return (
            fluid_properties is not None
            and freestream is not None
            and isinstance(freestream, FreestreamFromVelocity)
        )

    def update_model(self) -> Flow360BaseModel:
        params = {}

        if self.freestream is not None:
            params["freestream"] = try_update(self.freestream)
            params["fluid_properties"] = self.freestream.extract_fluid_properties()

        if self.bet_disks is not None:
            disks = []
            for disk in self.bet_disks:
                disks.append(try_update(disk))
            params["bet_disks"] = disks

        if self.sliding_interfaces is not None:
            volume_zones = {}
            for interface in self.sliding_interfaces:
                volume_zone = try_update(interface)
                volume_zones[interface.volume_name] = volume_zone
            params["volume_zones"] = VolumeZones(**volume_zones)

        if self._is_web_ui_generated(params.get("fluid_properties"), params.get("freestream")):
            context = SI_unit_system
        else:
            context = flow360_unit_system

        with context:
            params.update(
                {
                    "geometry": try_update(self.geometry),
                    "boundaries": self.boundaries,
                    "initial_condition": self.initial_condition,
                    "time_stepping": try_update(self.time_stepping),
                    "navier_stokes_solver": try_update(self.navier_stokes_solver),
                    "turbulence_model_solver": try_update(self.turbulence_model_solver),
                    "transition_model_solver": try_update(self.transition_model_solver),
                    "heat_equation_solver": try_update(self.heat_equation_solver),
                    "actuator_disks": self.actuator_disks,
                    "porous_media": self.porous_media,
                    "user_defined_dynamics": self.user_defined_dynamics,
                    "surface_output": try_update(self.surface_output),
                    "volume_output": try_update(self.volume_output),
                    "slice_output": try_update(self.slice_output),
                    "iso_surface_output": try_update(self.iso_surface_output),
                    "monitor_output": self.monitor_output,
                    "aeroacoustic_output": self.aeroacoustic_output,
                    "fluid_properties": None,
                    "volume_zones": None,
                    "bet_disks": None,
                }
            )

            model = Flow360Params(**params)
            return model

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        allow_but_remove = ["runControl", "testControl"]
