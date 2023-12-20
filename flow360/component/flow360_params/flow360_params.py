"""
Flow360 solver parameters
"""
# This is a temporary measure until flow360_temp models are merged
# pylint: disable=too-many-lines
# pylint: disable=unused-import
from __future__ import annotations

from abc import ABCMeta
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

from ...exceptions import ConfigError, Flow360NotImplementedError, ValidationError
from ...log import log
from ...user_config import UserConfig
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
from .flow360_output import (
    AeroacousticOutput,
    AnimationSettings,
    AnimationSettingsExtended,
    IsoSurfaceOutput,
    IsoSurfaces,
    MonitorOutput,
    Monitors,
    ProbeMonitor,
    SliceOutput,
    Slices,
    SurfaceIntegralMonitor,
    SurfaceOutput,
    Surfaces,
    VolumeOutput,
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
    KOmegaSST,
    LinearSolver,
    NavierStokesSolver,
    NoneSolver,
    SpalartAllmaras,
    TransitionModelSolver,
    TurbulenceModelSolverTypes,
)
from .unit_system import (
    AngularVelocityType,
    AreaType,
    CGS_unit_system,
    DensityType,
    LengthType,
    PressureType,
    SI_unit_system,
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


# pylint: disable=invalid-name
def get_time_non_dim_unit(mesh_unit_length, C_inf, extra_msg=""):
    """
    returns time non-dimensionalisation
    """

    if mesh_unit_length is None or C_inf is None:
        required = ["mesh_unit", "mesh_unit_length"]
        raise ConfigError(f"You need to provide one of {required} AND C_inf {extra_msg}")
    return mesh_unit_length / C_inf


def get_length_non_dim_unit(mesh_unit_length, extra_msg=""):
    """
    returns length non-dimensionalisation
    """
    if mesh_unit_length is None:
        required = ["mesh_unit", "mesh_unit_length"]
        raise ConfigError(f"You need to provide one of {required} {extra_msg}")
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

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


class SlipWall(Boundary):
    """Slip wall boundary"""

    type = pd.Field("SlipWall", const=True)


class FreestreamBoundary(Boundary):
    """Freestream boundary"""

    type = pd.Field("Freestream", const=True)
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


class IsothermalWall(Boundary):
    """IsothermalWall boundary"""

    type = pd.Field("IsothermalWall", const=True)
    temperature: Union[PositiveFloat, StrictStr] = pd.Field(alias="Temperature")
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


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

    type = pd.Field("HeatFluxWall", const=True)
    heat_flux: Union[float, StrictStr] = pd.Field(alias="heatFlux")
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="velocity")


class SubsonicOutflowPressure(Boundary):
    """SubsonicOutflowPressure boundary"""

    type = pd.Field("SubsonicOutflowPressure", const=True)
    static_pressure_ratio: PositiveFloat = pd.Field(alias="staticPressureRatio")


class SubsonicOutflowMach(Boundary):
    """SubsonicOutflowMach boundary"""

    type = pd.Field("SubsonicOutflowMach", const=True)
    Mach: PositiveFloat = pd.Field(alias="MachNumber")


class SubsonicInflow(Boundary):
    """SubsonicInflow boundary"""

    type = pd.Field("SubsonicInflow", const=True)
    total_pressure_ratio: PositiveFloat = pd.Field(alias="totalPressureRatio")
    total_temperature_ratio: PositiveFloat = pd.Field(alias="totalTemperatureRatio")
    ramp_steps: Optional[PositiveInt] = pd.Field(alias="rampSteps")
    velocity_direction: Optional[BoundaryVelocityType] = pd.Field(alias="velocityDirection")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


class SupersonicInflow(Boundary):
    """:class:`SupersonicInflow` class for specifying the full fluid state at supersonic inflow boundaries

    Parameters
    ----------
    totalTemperatureRatio : PositiveFloat
        Ratio of total temperature to static temperature at the inlet.

    totalPressureRatio: PositiveFloat
        Ratio of the total pressure to static pressure at the inlet.

    staticPressureRatio: PositiveFloat
        Ratio of the inlet static pressure to the freestream static pressure. Default freestream static pressure in
        Flow360 = 1.0/gamma.

    velocityDirection: BoundaryVelocityType
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
    total_temperature_ratio: PositiveFloat = pd.Field(alias="totalTemperatureRatio")
    total_pressure_ratio: PositiveFloat = pd.Field(alias="totalPressureRatio")
    static_pressure_ratio: PositiveFloat = pd.Field(alias="staticPressureRatio")
    velocity_direction: Optional[BoundaryVelocityType] = pd.Field(alias="velocityDirection")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


class SlidingInterfaceBoundary(Boundary):
    """:class: `SlidingInterface` boundary"""

    type = pd.Field("SlidingInterface", const=True)


class WallFunction(Boundary):
    """:class: `WallFunction` boundary"""

    type = pd.Field("WallFunction", const=True)


class MassInflow(Boundary):
    """:class: `MassInflow` boundary"""

    type = pd.Field("MassInflow", const=True)
    mass_flow_rate: PositiveFloat = pd.Field(alias="massFlowRate")
    ramp_steps: Optional[PositiveInt] = pd.Field(alias="rampSteps")


class MassOutflow(Boundary):
    """:class: `MassOutflow` boundary"""

    type = pd.Field("MassOutflow", const=True)
    mass_flow_rate: PositiveFloat = pd.Field(alias="massFlowRate")
    ramp_steps: Optional[PositiveInt] = pd.Field(alias="rampSteps")


class SolidIsothermalWall(Boundary):
    """:class: `SolidIsothermalWall` boundary"""

    type = pd.Field("SolidIsothermalWall", const=True)
    temperature: Union[PositiveFloat, StrictStr] = pd.Field(alias="Temperature")


class SolidAdiabaticWall(Boundary):
    """:class: `SolidAdiabaticWall` boundary"""

    type = pd.Field("SolidAdiabaticWall", const=True)


class TranslationallyPeriodic(Boundary):
    """:class: `TranslationallyPeriodic` boundary"""

    type = pd.Field("TranslationallyPeriodic", const=True)
    paired_patch_name: Optional[str] = pd.Field(alias="pairedPatchName")
    translation_vector: Optional[Vector] = pd.Field(alias="translationVector")


class RotationallyPeriodic(Boundary):
    """:class: `RotationallyPeriodic` boundary"""

    type = pd.Field("RotationallyPeriodic", const=True)
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
    def _get_widgets(cls) -> dict[str, str]:
        return {
            "center": "vector3",
            "axisThrust": "vector3"
        }


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
    def _get_widgets(cls) -> dict[str, str]:
        return {
            "centerOfRotation": "vector3"
        }


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
    max_pseudo_steps: Optional[pd.conint(gt=0, le=100000)] = pd.Field(alias="maxPseudoSteps")
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
        require_unit_system_context = True
        deprecated_aliases = [DeprecatedAlias(name="physical_steps", deprecated="maxPhysicalSteps")]


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
    def _get_widgets(cls) -> dict[str, str]:
        return {
            "additionalProperties/velocity/value": "vector3",
            "additionalProperties/Velocity/value": "vector3",
            "additionalProperties/velocityDirection/value": "vector3",
            "additionalProperties/translationVector": "vector3",
            "additionalProperties/axisOfRotation": "vector3"
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

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


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
        require_unit_system_context = True


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

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


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

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


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

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


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
    def _get_optional_objects(cls) -> List[str]:
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
    def _get_widgets(cls) -> dict[str, str]:
        return {
            "additionalProperties/referenceFrame/centerOfRotation/value": "vector3",
            "additionalProperties/referenceFrame/axisOfRotation": "vector3",
        }


class Geometry(Flow360BaseModel):
    """
    :class: Geometry component
    """

    ref_area: Optional[AreaType.Positive] = pd.Field(alias="refArea", default_factory=lambda: 1.0)
    moment_center: Optional[LengthType.Point] = pd.Field(
        alias="momentCenter", default_factory=lambda: (0, 0, 0)
    )
    moment_length: Optional[LengthType.Moment] = pd.Field(
        alias="momentLength", default_factory=lambda: (1, 1, 1)
    )
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
        require_unit_system_context = True


class FreestreamBase(Flow360BaseModel, metaclass=ABCMeta):
    """
    :class: Freestream component
    """

    alpha: Optional[float] = pd.Field(alias="alphaAngle", default=0)
    beta: Optional[float] = pd.Field(alias="betaAngle", default=0)
    turbulent_viscosity_ratio: Optional[NonNegativeFloat] = pd.Field(
        alias="turbulentViscosityRatio"
    )


class FreestreamFromMach(FreestreamBase):
    """
    :class: Freestream component using Mach numbers
    """

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

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


class ZeroFreestreamFromVelocity(FreestreamBase):
    """
    :class: Zero velocity freestream component using dimensioned velocity
    """

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

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


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

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


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

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


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
        raise NotImplementedError("USstandardAtmosphere not implemented yet.")

    def to_fluid_properties(self) -> _FluidProperties:
        """Converts the instance to _FluidProperties, incorporating temperature, pressure, density, and viscosity."""

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_unit_system_context = True


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
    def _get_widgets(cls) -> dict[str, str]:
        return {
            "centerOfRotation": "vector3",
            "axisOfRotation": "vector3",
            "initialBladeDirection": "vector3"
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
    def _get_widgets(cls) -> dict[str, str]:
        return {
            "DarcyCoefficient": "vector3",
            "ForchheimerCoefficient": "vector3",
            "volumeZone/center": "vector3",
            "volumeZone/lengths": "vector3",
            "volumeZone/axes/items": "vector3",
            "volumeZone/windowingLengths": "vector3"
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

    # save unit system for future use, for example processing results: TODO:
    # unit_system: UnitSystem = pd.Field(alias='unitSystem', default_factory=unit_system_manager.copy_current)
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

    # save unit system for future use, for example processing results: TODO:
    # validator: what if context is different than what is provided to the constructor (should it be ignored?)
    # how copy should work? copy unit_system as well or use context?
    # @pd.validator('unit_system')
    # def should_be_consistent_with_context(cls, v):
    #     if v == unit_system_manager.current:
    #         return v
    #   raise ValueError(f'unit_system is inconsistent from context unit system: \n{v}\n{unit_system_manager.current}')

    # save unit system for future use, for example processing results: TODO:
    def _handle_unit_system_init(self, init_dict):
        """
        handling unit systems:
        if no unit_system provided and no context:
        - save with the model flow360 unit system, use flow360 context to infer units - NEED TO THINK ABOUT IT!!!
        if unit_system provided and no context:
        - safe with the model provided unit system, use flow360 context to infer units
        if unit_system not provided and context:
        - safe with the model context unit system, dont use additional context (already in the context)
        if unit_system provided and context provided:
        - show warning, save context as model unit_system, show warning that model unit system changed.
        """

    # unit_system = init_dict.get('unit_system', init_dict.get('unitSystem', None))
    # use_unit_system_as_context = None
    # save_unit_system_with_model = None

    # if not isinstance(unit_system, (UnitSystem, None)):
    #     unit_system = UnitSystem(unit_system)

    # if unit_system is not None:
    #     if not isinstance(unit_system, UnitSystem):
    #         unit_system = UnitSystem(unit_system)

    #     if unit_system_manager.current is None:
    #         save_unit_system_with_model = unit_system
    #         use_unit_system_as_context = UnitSystem(base_system="Flow360", verbose=False)

    #     if unit_system != unit_system_manager.current:
    #         log.warning(
    #                 f'Trying to initialize {self.__class__.__name__} with unit system: '
    #                 f'{unit_system.system_repr()} '
    #                 f'inside context: {unit_system_manager.current.system_repr()} '
    #                 f'context unit system will be saved with the model.'
    #             )
    #         return unit_system_manager.current

    # return save_unit_system_with_model, use_unit_system_as_context

    def __init__(self, filename: str = None, **kwargs):
        # if filename:
        #     obj = self.from_file(filename=filename)
        #     init_dict = obj.dict()
        # else:
        #     init_dict = kwargs
        # save_unit_system_with_model, use_unit_system_as_context = self._handle_unit_system_init(init_dict)

        if unit_system_manager.current is None:
            with UnitSystem(base_system="Flow360", verbose=False):
                super().__init__(filename, **kwargs)
        else:
            super().__init__(filename, **kwargs)

    # pylint: disable=arguments-differ
    def to_solver(self) -> Flow360Params:
        """
        returns configuration object in flow360 units system
        """
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
        solver_params_json = solver_params.json(encoder=flow360_json_encoder)
        return solver_params_json

    def append(self, params: Flow360Params, overwrite: bool = False):
        if not isinstance(params, Flow360Params):
            raise ValueError("params must be type of Flow360Params")
        super().append(params=params, overwrite=overwrite)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        allow_but_remove = ["runControl", "testControl"]

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
            raise ConfigError(
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
