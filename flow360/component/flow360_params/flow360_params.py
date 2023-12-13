"""
Flow360 solver parameters
"""
# This is a temporary measure until flow360_temp models are merged
# pylint: disable=too-many-lines
# pylint: disable=unused-import
from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_args,
    get_type_hints,
)

import pydantic as pd
from pydantic import StrictStr
from typing_extensions import Literal

import flow360

from ...error_messages import unit_system_inconsistent_msg, use_unit_system_msg
from ...exceptions import (
    Flow360ConfigError,
    Flow360NotImplementedError,
    Flow360RuntimeError,
    Flow360ValidationError,
)
from ...log import log
from ...user_config import UserConfig

# from .updater import update
from ...version import __version__
from ..constants import constants
from ..types import Axis, Coordinate, NonNegativeFloat, PositiveFloat, PositiveInt
from ..utils import _get_value_or_none, beta_feature
from .conversions import ExtraDimensionedProperty
from .flow360_legacy import (
    LegacyModel,
    _get_output_fields,
    _try_add_unit,
    _try_set,
    _try_update,
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
from .flow360_temp import BETDisk, InitialConditions, PorousMedium, UserDefinedDynamic
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
    LinearSolver,
    NavierStokesSolver,
    NavierStokesSolverLegacy,
    NoneSolver,
    TransitionModelSolver,
    TransitionModelSolverLegacy,
    TurbulenceModelSolverLegacy,
    TurbulenceModelSolverSA,
    TurbulenceModelSolverSST,
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
    VelocityType,
    ViscosityType,
    imperial_unit_system,
    u,
    unit_system_manager,
)

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

    type = pd.Field("NoSlipWall", const=True)
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")


class SlipWall(Boundary):
    """Slip wall boundary"""

    type = pd.Field("SlipWall", const=True)


class FreestreamBoundary(Boundary):
    """Freestream boundary"""

    type = pd.Field("Freestream", const=True)
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")


class IsothermalWall(Boundary):
    """IsothermalWall boundary"""

    type = pd.Field("IsothermalWall", const=True)
    temperature: Union[PositiveFloat, StrictStr] = pd.Field(alias="Temperature")
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")


class SubsonicOutflowPressure(Boundary):
    """SubsonicOutflowPressure boundary"""

    type = pd.Field("SubsonicOutflowPressure", const=True)
    staticPressureRatio: PositiveFloat


class SubsonicOutflowMach(Boundary):
    """SubsonicOutflowMach boundary"""

    type = pd.Field("SubsonicOutflowMach", const=True)
    Mach: PositiveFloat = pd.Field(alias="MachNumber")


class SubsonicInflow(Boundary):
    """SubsonicInflow boundary"""

    type = pd.Field("SubsonicInflow", const=True)
    totalPressureRatio: PositiveFloat
    totalTemperatureRatio: PositiveFloat
    ramp_steps: Optional[PositiveInt] = pd.Field(alias="rampSteps")


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
    >>> supersonicInflow = SupersonifInflow(totalTemperatureRatio=2.1, totalPressureRatio=3.0, staticPressureRatio=1.2)
    """

    type = pd.Field("SupersonicInflow", const=True)
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
    """SlidingInterface boundary"""

    type = pd.Field("SlidingInterface", const=True)


class WallFunction(Boundary):
    """WallFunction boundary"""

    type = pd.Field("WallFunction", const=True)


class MassInflow(Boundary):
    """MassInflow boundary"""

    type = pd.Field("MassInflow", const=True)
    massFlowRate: PositiveFloat


class MassOutflow(Boundary):
    """MassOutflow boundary"""

    type = pd.Field("MassOutflow", const=True)
    massFlowRate: PositiveFloat


class SolidIsothermalWall(Boundary):
    """SolidIsothermalWall boundary"""

    type = pd.Field("SolidIsothermalWall", const=True)
    temperature: Union[PositiveFloat, StrictStr] = pd.Field(alias="Temperature")


class SolidAdiabaticWall(Boundary):
    """SolidAdiabaticWall boundary"""

    type = pd.Field("SolidAdiabaticWall", const=True)


BoundaryType = Union[
    NoSlipWall,
    SlipWall,
    FreestreamBoundary,
    IsothermalWall,
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

    type: str = pd.Field("ramp", const=True)
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

    type: str = pd.Field("adaptive", const=True)
    min: Optional[PositiveFloat] = pd.Field(default=0.1)
    max: Optional[PositiveFloat] = pd.Field(default=10000)
    max_relative_change: Optional[PositiveFloat] = pd.Field(alias="maxRelativeChange", default=1)
    convergence_limiting_factor: Optional[PositiveFloat] = pd.Field(
        alias="convergenceLimitingFactor", default=0.25
    )


# pylint: disable=E0213
class TimeStepping(Flow360BaseModel):
    """
    Time stepping component
    """

    physical_steps: Optional[PositiveInt] = pd.Field(alias="physicalSteps")
    max_pseudo_steps: Optional[PositiveInt] = pd.Field(alias="maxPseudoSteps")
    time_step_size: Optional[Union[Literal["inf"], TimeType.Positive]] = pd.Field(
        alias="timeStepSize", default="inf"
    )
    CFL: Optional[Union[RampCFL, AdaptiveCFL]] = pd.Field()

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> TimeStepping:
        """
        returns configuration object in flow360 units system
        """
        return super().to_solver(params, **kwargs)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [DeprecatedAlias(name="physical_steps", deprecated="maxPhysicalSteps")]


class _GenericBoundaryWrapper(Flow360BaseModel):
    v: BoundaryType


class Boundaries(Flow360SortableBaseModel):
    """:class:`Boundaries` class for setting up Boundaries

    Parameters
    ----------
    <boundary_name> : BoundaryType
        Supported boundary types: Union[NoSlipWall, SlipWall, FreestreamBoundary, IsothermalWall,
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


# class OldReferenceFrame(Flow360BaseModel):
#     """:class:`ReferenceFrame` class for setting up reference frame

#     Parameters
#     ----------
#     center : Coordinate
#         Coordinate representing the origin of rotation, eg. (0, 0, 0)

#     axis : Axis
#         Axis of rotation, eg. (0, 0, 1)

#     parent_volume_name : str, optional
#         Name of the volume zone that the rotating reference frame is contained in, used to compute the acceleration in
#         the nested rotating reference frame

#     theta_radians : str, optional
#         Expression for rotation angle (in radians) as a function of time

#     theta_degrees : str, optional
#         Expression for rotation angle (in degrees) as a function of time

#     omega_radians
#         Nondimensional rotating speed, radians/nondim-unit-time

#     omega_degrees
#         Nondimensional rotating speed, degrees/nondim-unit-time

#     is_dynamic
#         Whether rotation of this interface is dictated by userDefinedDynamics


#     Returns
#     -------
#     :class:`ReferenceFrame`
#         An instance of the component class ReferenceFrame.

#     Example
#     -------
#     >>> rf = ReferenceFrame(
#             center=(0, 0, 0),
#             axis=(0, 0, 1),
#             omega_radians=1
#         )
#     """

#     theta_radians: Optional[str] = pd.Field(alias="thetaRadians")
#     theta_degrees: Optional[str] = pd.Field(alias="thetaDegrees")
#     omega_radians: Optional[float] = pd.Field(alias="omegaRadians")
#     omega_degrees: Optional[float] = pd.Field(alias="omegaDegrees")
#     center: Coordinate = pd.Field(alias="centerOfRotation")
#     axis: Axis = pd.Field(alias="axisOfRotation")
#     parent_volume_name: Optional[str] = pd.Field(alias="parentVolumeName")
#     is_dynamic: Optional[bool] = pd.Field(alias="isDynamic")

#     # pylint: disable=missing-class-docstring,too-few-public-methods
#     class Config(Flow360BaseModel.Config):
#         require_one_of = [
#             "omega_radians",
#             "omega_degrees",
#             "theta_radians",
#             "theta_degrees",
#             "is_dynamic",
#         ]


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


class Geometry(Flow360BaseModel):
    """
    Geometry component
    """

    ref_area: Optional[AreaType] = pd.Field(alias="refArea", default_factory=lambda: 1.0)
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
    Freestream component
    """

    alpha: Optional[float] = pd.Field(alias="alphaAngle")
    beta: Optional[float] = pd.Field(alias="betaAngle", default=0)
    turbulent_viscosity_ratio: Optional[NonNegativeFloat] = pd.Field(
        alias="turbulentViscosityRatio"
    )


class FreestreamFromMach(FreestreamBase):
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

        if mach_ref is not None:
            mach_ref = mach_ref.v.item()

        return FreestreamFromMach(
            Mach=mach, Mach_ref=mach_ref, temperature=temperature, mu_ref=mu_ref, **solver_values
        )


class ZeroFreestreamFromVelocity(FreestreamBase):
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


# class OldFreestream(Flow360BaseModel):
#     """
#     Freestream component
#     """

#     Reynolds: Optional[PositiveFloat] = pd.Field()
#     Mach: Optional[NonNegativeFloat] = pd.Field()
#     MachRef: Optional[PositiveFloat] = pd.Field()
#     mu_ref: Optional[PositiveFloat] = pd.Field(alias="muRef")
#     temperature: PositiveFloat = pd.Field(alias="Temperature")
#     density: Optional[PositiveFloat]
#     speed: Optional[Union[Velocity, PositiveFloat]]
#     alpha: Optional[float] = pd.Field(alias="alphaAngle")
#     beta: Optional[float] = pd.Field(alias="betaAngle", default=0)
#     turbulent_viscosity_ratio: Optional[NonNegativeFloat] = pd.Field(
#         alias="turbulentViscosityRatio"
#     )

#     @pd.validator("speed", pre=True, always=True)
#     def validate_speed(cls, v):
#         """speed validator"""
#         if isinstance(v, tuple):
#             return Velocity(v=v[0], unit=v[1])
#         return v

#     def get_C_inf(self):
#         """returns speed of sound based on model's temperature"""
#         return self._speed_of_sound_from_temperature(self.temperature)

#     @classmethod
#     def _speed_of_sound_from_temperature(cls, T: float):
#         """calculates speed of sound"""
#         return math.sqrt(1.4 * constants.R * T)

#     # pylint: disable=invalid-name
#     def _mach_from_speed(self, speed):
#         if speed:
#             C_inf = self.get_C_inf()
#             if isinstance(self.speed, Velocity):
#                 self.Mach = self.speed.v / C_inf
#             else:
#                 self.Mach = self.speed / C_inf

#     @classmethod
#     def from_speed(
#         cls,
#         speed: Union[Velocity, PositiveFloat] = None,
#         temperature: PositiveFloat = 288.15,
#         density: PositiveFloat = 1.225,
#         **kwargs,
#     ) -> Freestream:
#         """class: `Freestream`

#         Parameters
#         ----------
#         speed : Union[Velocity, PositiveFloat]
#             Value for freestream speed, e.g., (100.0, 'm/s'). If no unit provided, meters per second is assumed.
#         temperature : PositiveFloat, optional
#             temeperature, by default 288.15
#         density : PositiveFloat, optional
#             density, by default 1.225

#         Returns
#         -------
#         :class: `Freestream`
#             returns Freestream object

#         Example
#         -------
#         >>> fs = Freestream.from_speed(speed=(10, "m/s"))
#         """
#         assert speed
#         fs = cls(temperature=temperature, speed=speed, density=density, **kwargs)
#         fs._mach_from_speed(fs.speed)
#         return fs

#     @export_to_flow360
#     def to_flow360_json(self, return_json: bool = True, mesh_unit_length=None):
#         """
#         returns flow360 formatted json
#         """
#         if self.Reynolds is None and self.mu_ref is None:
#             if self.density is None:
#                 raise ConfigError("density is required.")
#             viscosity = 1.458e-6 * pow(self.temperature, 1.5) / (self.temperature + 110.4)
#             self.mu_ref = (
#                 viscosity
#                 / (self.density * self.get_C_inf())
#                 / get_length_non_dim_unit(mesh_unit_length)
#             )

#         if self.speed:
#             self._mach_from_speed(self.speed)

#         if return_json:
#             return self.json()
#         return None

#     # pylint: disable=missing-class-docstring,too-few-public-methods
#     class Config(Flow360BaseModel.Config):
#         exclude_on_flow360_export = ["speed", "density"]
#         require_one_of = ["Mach", "speed"]


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

UnitSystemTypes = Union[
    SIUnitSystem, CGSUnitSystem, ImperialUnitSystem, Flow360UnitSystem, UnitSystem
]


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
    time_stepping: Optional[TimeStepping] = pd.Field(alias="timeStepping", default=TimeStepping())
    navier_stokes_solver: Optional[NavierStokesSolver] = pd.Field(alias="navierStokesSolver")
    turbulence_model_solver: Optional[TurbulenceModelSolverTypes] = pd.Field(
        alias="turbulenceModelSolver", discriminator="model_type"
    )
    transition_model_solver: Optional[TransitionModelSolver] = pd.Field(
        alias="transitionModelSolver"
    )
    heat_equation_solver: Optional[HeatEquationSolver] = pd.Field(alias="heatEquationSolver")
    freestream: Optional[FreestreamTypes] = pd.Field()
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
                name = kwarg_unit_system.get("name")
                kwarg_unit_system = None
                if name is not None:
                    if name == "SI":
                        kwarg_unit_system = SIUnitSystem()
                    elif name == "CGS":
                        kwarg_unit_system = CGSUnitSystem()
                    elif name == "Imperial":
                        kwarg_unit_system = ImperialUnitSystem()
                    elif name == "Flow360":
                        kwarg_unit_system = Flow360UnitSystem()
                    else:
                        raise Flow360RuntimeError(f"Undefined unit system name provided: {name}")
                else:
                    kwarg_unit_system = UnitSystem(**kwarg_unit_system)
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
    def from_file(cls, filename: str) -> Flow360BaseModel:
        """Loads a :class:`Flow360BaseModel` from .json, or .yaml file.

        Parameters
        ----------
        filename : str
            Full path to the .yaml or .json file to load the :class:`Flow360BaseModel` from.
        **parse_obj_kwargs
            Keyword arguments passed to either pydantic's ``parse_obj`` function when loading model.

        Returns
        -------
        :class:`Flow360BaseModel`
            An instance of the component class calling `load`.

        Example
        -------
        >>> simulation = Flow360Params.from_file(filename='folder/sim.json') # doctest: +SKIP
        """
        return cls(filename=filename)

    def _has_key(self, target, model_dict: dict):
        for key, value in model_dict.items():
            if key == target:
                return True
            else:
                if isinstance(value, dict):
                    if self._has_key(target, value):
                        return True
        return False

    def _init_from_file(self, filename, **kwargs):
        if unit_system_manager.current is not None:
            raise Flow360RuntimeError(
                "When loading params from file: Flow360Params(filename), unit context must not be used."
            )

        model_dict = self._init_handle_file(filename=filename, **kwargs)

        version = model_dict.get("version")
        unit_system = model_dict.get("unitSystem")
        if version is not None and version == __version__:
            if unit_system is None:
                with flow360.flow360_unit_system:
                    super().__init__(unit_system=unit_system_manager.copy_current(), **model_dict)
            else:
                super().__init__(**model_dict)
        else:
            self._init_with_update(model_dict)

    def _init_with_update(self, model_dict):
        try:
            super().__init__(**model_dict)
        except Exception as err_current:
            try:
                # Check if comments are present within the file
                if self._has_key("comments", model_dict):
                    with flow360.SI_unit_system:
                        legacy = Flow360ParamsLegacy(**model_dict)
                        super().__init__(**legacy.update_model().dict())
                else:
                    with flow360.flow360_unit_system:
                        legacy = Flow360ParamsLegacy(**model_dict)
                        super().__init__(**legacy.update_model().dict())
            except Exception as err_legacy:
                log.error("Tried to use current params format but following errors occured:")
                log.error(err_current)
                log.error("Tried to use legacy params format but following errors occured:")
                log.error(err_legacy)
                raise ValueError("loading from file failed")

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


""" Legacy models for Flow360 updater, do not expose """


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

    ref_area: Optional[float] = pd.Field(alias="refArea", default_factory=lambda: 1.0)
    moment_center: Optional[Coordinate] = pd.Field(alias="momentCenter")
    moment_length: Optional[Coordinate] = pd.Field(alias="momentLength")

    def update_model(self) -> Flow360BaseModel:
        model = {
            "momentCenter": self.moment_center,
            "momentLength": self.moment_length,
            "refArea": self.ref_area,
        }
        if self.comments.get("meshUnit") is not None:
            unit = u.unyt_quantity(1, self.comments["meshUnit"])
            model["meshUnit"] = unit
            _try_add_unit(model, "momentCenter", model["meshUnit"])
            _try_add_unit(model, "momentLength", model["meshUnit"])
            _try_add_unit(model, "refArea", model["meshUnit"] ** 2)

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

            freestream: FreestreamTypes = pd.Field()

        model = {
            "freestream": {
                "alphaAngle": self.alpha,
                "betaAngle": self.beta,
                "turbulentViscosityRatio": self.turbulent_viscosity_ratio,
            }
        }

        # Set velocity
        if self.comments.get("freestreamMeterPerSecond") is not None:
            # pylint: disable=no-member
            velocity = self.comments["freestreamMeterPerSecond"] * u.m / u.s
            _try_set(model["freestream"], "velocity", velocity)
        elif self.comments.get("speedOfSoundMeterPerSecond") is not None and self.Mach is not None:
            # pylint: disable=no-member
            velocity = self.comments["speedOfSoundMeterPerSecond"] * self.Mach * u.m / u.s
            _try_set(model["freestream"], "velocity", velocity)

        if model["freestream"].get("velocity"):
            # Set velocity_ref
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
                _try_set(model["freestream"], "velocityRef", velocity_ref)
            else:
                model["freestream"]["velocityRef"] = None
        else:
            _try_set(model["freestream"], "Reynolds", self.Reynolds)
            _try_set(model["freestream"], "muRef", self.mu_ref)
            _try_set(model["freestream"], "temperature", self.temperature)
            _try_set(model["freestream"], "Mach", self.Mach)
            _try_set(model["freestream"], "MachRef", self.Mach_Ref)

        return _FreestreamTempModel.parse_obj(model).freestream

    def extract_fluid_properties(self) -> Optional[Flow360BaseModel]:
        """Extract fluid properties from the freestream comments"""

        class _FluidPropertiesTempModel(pd.BaseModel):
            """Helper class used to create
            the correct fluid properties from dict data"""

            fluid: FluidPropertyTypes = pd.Field()

        model = {"fluid": {}}

        # pylint: disable=no-member
        _try_set(model["fluid"], "temperature", self.temperature * u.K)

        if self.comments.get("densityKgPerCubicMeter"):
            # pylint: disable=no-member
            density = self.comments["densityKgPerCubicMeter"] * u.kg / u.m**3
            _try_set(model["fluid"], "density", density)
        else:
            return None

        return _FluidPropertiesTempModel.parse_obj(model).fluid


class TimeSteppingLegacy(TimeStepping, LegacyModel):
    """:class: `TimeSteppingLegacy` class"""

    time_step_size: Optional[Union[Literal["inf"], PositiveFloat]] = pd.Field(
        alias="timeStepSize", default="inf"
    )

    def update_model(self) -> Flow360BaseModel:
        model = {
            "CFL": self.CFL,
            "physicalSteps": self.physical_steps,
            "maxPseudoSteps": self.max_pseudo_steps,
            "timeStepSize": self.time_step_size,
        }

        if (
            model["timeStepSize"] != "inf"
            and self.comments.get("timeStepSizeInSeconds") is not None
        ):
            step_unit = u.unyt_quantity(self.comments["timeStepSizeInSeconds"], "s")
            _try_add_unit(model, "timeStepSize", step_unit)

        return TimeStepping.parse_obj(model)


class SlidingInterfaceLegacy(SlidingInterface, LegacyModel):
    """:class:`SlidingInterfaceLegacy` class"""

    omega: Optional[float] = pd.Field()

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_one_of = SlidingInterface.Config.require_one_of + ["omega"]

    def update_model(self) -> Flow360BaseModel:
        model = {
            "modelType": "FluidDynamics",
            "referenceFrame": {
                "axis": self.axis,
                # pylint: disable=no-member
                "center": self.center * u.m,
            },
        }

        _try_set(model["referenceFrame"], "isDynamic", self.is_dynamic)
        _try_set(model["referenceFrame"], "omegaRadians", self.omega)
        _try_set(model["referenceFrame"], "omegaRadians", self.omega_radians)
        _try_set(model["referenceFrame"], "omegaDegrees", self.omega_degrees)
        _try_set(model["referenceFrame"], "thetaRadians", self.theta_radians)
        _try_set(model["referenceFrame"], "thetaDegrees", self.theta_degrees)

        if self.comments.get("rpm") is not None:
            # pylint: disable=no-member
            omega = self.comments["rpm"] * u.rpm
            _try_set(model["referenceFrame"], "omega", omega)
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

    def update_model(self) -> Flow360BaseModel:
        model = Flow360Params()
        model.geometry = _try_update(self.geometry)
        model.boundaries = self.boundaries
        model.initial_condition = self.initial_condition
        model.time_stepping = _try_update(self.time_stepping)
        model.navier_stokes_solver = _try_update(self.navier_stokes_solver)
        model.turbulence_model_solver = _try_update(self.turbulence_model_solver)
        model.transition_model_solver = _try_update(self.transition_model_solver)
        model.heat_equation_solver = _try_update(self.heat_equation_solver)
        model.freestream = _try_update(self.freestream)

        if self.freestream is not None:
            model.fluid_properties = self.freestream.extract_fluid_properties()

        if self.bet_disks is not None:
            disks = []
            for disk in self.bet_disks:
                disks.append(_try_update(disk))
            model.bet_disks = disks

        model.actuator_disks = self.actuator_disks
        model.porous_media = self.porous_media
        model.user_defined_dynamics = self.user_defined_dynamics
        model.surface_output = _try_update(self.surface_output)
        model.volume_output = _try_update(self.volume_output)
        model.slice_output = _try_update(self.slice_output)
        model.iso_surface_output = _try_update(self.iso_surface_output)
        model.monitor_output = self.monitor_output

        if self.sliding_interfaces is not None:
            volume_zones = {}
            for interface in self.sliding_interfaces:
                volume_zone = _try_update(interface)
                volume_zones[interface.volume_name] = volume_zone
            model.volume_zones = VolumeZones(**volume_zones)

        model.aeroacoustic_output = self.aeroacoustic_output

        return model
