"""
Flow360 solver parameters
"""
# This is a temporary measure until flow360_temp models are merged
# pylint: disable=too-many-lines
# pylint: disable=unused-import
from __future__ import annotations

import math
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
from ..types import Axis, Coordinate, NonNegativeFloat, PositiveFloat, PositiveInt
from ..utils import _get_value_or_none, beta_feature
from .conversions import ExtraDimensionedProperty
from .flow360_output import (
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
    LinearSolver,
    NavierStokesSolver,
    NoneSolver,
    TransitionModelSolver,
    TurbulenceModelSolverSA,
    TurbulenceModelSolverSST,
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
    Density : PositiveFloat
        Fluid density (non-dimensional). Default freestream fluid density in Flow360 is 1.0.

    Velocity: Tuple[float, float, float]
        Fluid velocity (non-dimensional), normalized to Mach number.
    Pressure: PositiveFloat
        Fluid pressure (non-dimensional). Default freestream fluid pressure in Flow360 = 1.0/gamma = 1.0/1.4.

    Returns
    -------
    :class:`SupersonicInflow`
        An instance of the component class SupersonicInflow.

    Example
    -------
    >>> supersonicInflow = SupersonifInflow(density=1.0, velocity=[2.5, 0.1, 0], pressure=0.7) # doctest: +SKIP
    """

    type = pd.Field("SupersonicInflow", const=True)
    density: PositiveFloat = pd.Field(alias="Density")
    velocity: Tuple[float, float, float] = pd.Field(alias="Velocity")
    pressure: PositiveFloat = pd.Field(alias="Pressure")


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
            raise ValidationError(
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

    T_solid: Union[PositiveFloat, StrictStr]


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

    omega_radians: AngularVelocityType
        Nondimensional rotating speed, radians/nondim-unit-time


    Returns
    -------
    :class:`ReferenceFrameOmegaRadians`
        An instance of the component class ReferenceFrameOmegaRadians.

    """

    omega_radians: AngularVelocityType = pd.Field(alias="omegaRadians")
    center: LengthType.Point = pd.Field(alias="centerOfRotation")
    axis: Axis = pd.Field(alias="axisOfRotation")


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
        omega_radians = solver_values.pop("omega")
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


class AeroacousticOutput(Flow360BaseModel):
    """:class:`AeroacousticOutput` class for configuring output data about acoustic pressure signals

    Parameters
    ----------
    observers : List[Coordinate]
        List of observer locations at which time history of acoustic pressure signal is stored in aeroacoustic output
        file. The observer locations can be outside the simulation domain, but cannot be inside the solid surfaces of
        the simulation domain.
    animation_frequency: Union[PositiveInt, Literal[-1]], optional
        Frame frequency in the animation
    animation_frequency_offset: int, optional
        Animation frequency offset

    Returns
    -------
    :class:`AeroacousticOutput`
        An instance of the component class AeroacousticOutput.

    Example
    -------
    >>> aeroacoustics = AeroacousticOutput(observers=[(0, 0, 0), (1, 1, 1)], animation_frequency=1)
    """

    animation_frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequency"
    )
    animation_frequency_offset: Optional[int] = pd.Field(alias="animationFrequencyOffset")
    patch_type: Optional[str] = pd.Field("solid", const=True, alias="patchType")
    observers: List[Coordinate] = pd.Field()
    write_per_surface_output: Optional[bool] = pd.Field(False, alias="writePerSurfaceOutput")


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
    velocity_ref: Optional[VelocityType.Positive] = pd.Field()

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
    velocity_ref: VelocityType.Positive = pd.Field()

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
        raise NotImplementedError("USstandardAtmosphere not implemented yet.")

    def to_fluid_properties(self) -> _FluidProperties:
        """Converts the instance to _FluidProperties, incorporating temperature, pressure, density, and viscosity."""


# pylint: disable=no-member
air = AirDensityTemperature(temperature=288.15 * u.K, density=1.225 * u.kg / u.m**3)

FluidPropertyTypes = Union[AirDensityTemperature, AirPressureTemperature]


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
