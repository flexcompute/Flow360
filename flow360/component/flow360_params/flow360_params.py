"""
Flow360 solver parameters
"""

# pylint: disable=too-many-lines
# pylint: disable=unused-import
from __future__ import annotations

import json
import math
import os
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

from flow360 import units

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
from ..utils import _get_value_or_none
from .boundaries import BoundaryType, WallFunction
from .conversions import ExtraDimensionedProperty
from .flow360_legacy import (
    LegacyModel,
    get_output_fields,
    try_add_discriminator,
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
    SurfaceIntegralMonitor,
    SurfaceOutput,
    SurfaceOutputLegacy,
    Surfaces,
    VolumeOutput,
    VolumeOutputLegacy,
)
from .initial_condition import InitialConditions
from .params_base import (
    Conflicts,
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
    KOmegaSST,
    LinearSolver,
    NavierStokesSolver,
    NavierStokesSolverLegacy,
    NavierStokesSolverType,
    NoneSolver,
    SpalartAllmaras,
    TransitionModelSolver,
    TransitionModelSolverLegacy,
    TurbulenceModelSolverLegacy,
    TurbulenceModelSolverType,
)
from .time_stepping import (
    BaseTimeStepping,
    SteadyTimeStepping,
    TimeStepping,
    UnsteadyTimeStepping,
)
from .turbulence_quantities import TurbulenceQuantitiesType
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
    UnitSystemType,
    VelocityType,
    ViscosityType,
    flow360_unit_system,
    imperial_unit_system,
    u,
    unit_system_manager,
)
from .updater import updater
from .validations import (
    _check_aero_acoustics,
    _check_bet_disks_3d_coefficients_in_polars,
    _check_bet_disks_alphas_in_order,
    _check_bet_disks_duplicate_chords_or_twists,
    _check_bet_disks_number_of_defined_polars,
    _check_cht_solver_settings,
    _check_consistency_ddes_unsteady,
    _check_consistency_ddes_volume_output,
    _check_consistency_temperature,
    _check_consistency_wall_function_and_surface_output,
    _check_duplicate_boundary_name,
    _check_equation_eval_frequency_for_unsteady_simulations,
    _check_incompressible_navier_stokes_solver,
    _check_numerical_dissipation_factor_output,
    _check_periodic_boundary_mapping,
    _check_tri_quad_boundaries,
)
from .volume_zones import (
    FluidDynamicsVolumeZone,
    HeatTransferVolumeZone,
    ReferenceFrameType,
    VolumeZoneType,
)


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


class _GenericBoundaryWrapper(Flow360BaseModel):
    v: BoundaryType = pd.Field(discriminator="type")


class Boundaries(Flow360SortableBaseModel):
    """:class:`Boundaries` class for setting up Boundaries

    Parameters
    ----------
    <boundary_name> : BoundaryType
        Supported boundary types: Union[NoSlipWall, SlipWall, FreestreamBoundary, IsothermalWall, HeatFluxWall,
                                        SubsonicOutflowPressure, SubsonicOutflowMach, SubsonicInflow,
                                        SupersonicInflow, SlidingInterfaceBoundary, WallFunction,
                                        MassInflow, MassOutflow, SolidIsothermalWall, SolidAdiabaticWall,
                                        RiemannInvariant, VelocityInflow, PressureOutflow, SymmetryPlane]

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

    # pylint: disable=no-self-argument
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

    # pylint: disable=no-self-argument
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
    :class: Geometry component
    """

    ref_area: Optional[AreaType.Positive] = pd.Field(alias="refArea", displayed="Reference area")
    moment_center: Optional[LengthType.Point] = pd.Field(alias="momentCenter")
    ##Note: moment_length does not allow negative components I failed to enforce that here after attempts
    moment_length: Optional[LengthType.Moment] = pd.Field(alias="momentLength")
    mesh_unit: Optional[LengthType.Positive] = pd.Field(alias="meshUnit")

    # pylint: disable=arguments-differ
    def to_solver(self, params: Flow360Params, **kwargs) -> Geometry:
        """
        returns configuration object in flow360 units system
        """
        # Adds defaults:
        if self.moment_center is None:
            self.moment_center = (0, 0, 0) * units.flow360_length_unit
        if self.ref_area is None:
            self.ref_area = 1 * units.flow360_area_unit
        if self.moment_length is None:
            self.moment_length = (1.0, 1.0, 1.0) * units.flow360_length_unit
        if self.mesh_unit is None:
            self.mesh_unit = 1 * units.flow360_length_unit

        return super().to_solver(params, exclude=["mesh_unit"], **kwargs)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        allow_but_remove = ["meshName", "endianness"]


class FreestreamBase(Flow360BaseModel, metaclass=ABCMeta):
    """
    :class: Freestream component
    """

    model_type: str
    alpha: Optional[float] = pd.Field(alias="alphaAngle", default=0, displayed="Alpha angle [deg]")
    beta: Optional[float] = pd.Field(alias="betaAngle", default=0, displayed="Beta angle [deg]")
    turbulent_viscosity_ratio: Optional[NonNegativeFloat] = pd.Field(
        alias="turbulentViscosityRatio"
    )
    ## Legacy update pending.
    ## The validation for turbulenceQuantities (make sure we have correct combinations, maybe in root validator)
    ## is also pending. TODO
    turbulence_quantities: Optional[TurbulenceQuantitiesType] = pd.Field(
        alias="turbulenceQuantities"
    )

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        conflicting_fields = [
            Conflicts(field1="turbulent_viscosity_ratio", field2="turbulence_quantities")
        ]
        exclude_on_flow360_export = ["model_type"]


class FreestreamFromMach(FreestreamBase):
    """
    :class: Freestream component using Mach number
    """

    model_type: Literal["FromMach"] = pd.Field("FromMach", alias="modelType", const=True)
    Mach: PositiveFloat = pd.Field(displayed="Mach number")
    Mach_ref: Optional[PositiveFloat] = pd.Field(alias="MachRef", displayed="Reference Mach number")
    mu_ref: PositiveFloat = pd.Field(alias="muRef", displayed="Dynamic viscosity [non-dim]")
    temperature: Union[Literal[-1], PositiveFloat] = pd.Field(
        alias="Temperature", displayed="Temperature [K]"
    )

    # pylint: disable=arguments-differ, unused-argument
    def to_solver(self, params: Flow360Params, **kwargs) -> FreestreamFromMach:
        """
        returns configuration object in flow360 units system
        """
        return self.copy()


class FreestreamFromMachReynolds(FreestreamBase):
    """
    :class: Freestream component using Mach and Reynolds number
    """

    model_type: Literal["FromMachReynolds"] = pd.Field(
        "FromMachReynolds", alias="modelType", const=True
    )
    Mach: PositiveFloat = pd.Field(displayed="Mach number")
    Mach_ref: Optional[PositiveFloat] = pd.Field(alias="MachRef", displayed="Reference Mach number")
    Reynolds: Union[pd.confloat(gt=0, allow_inf_nan=False), Literal["inf"]] = pd.Field(
        displayed="Reynolds number"
    )
    temperature: Union[Literal[-1], PositiveFloat] = pd.Field(
        alias="Temperature", displayed="Temperature [K]"
    )

    # pylint: disable=arguments-differ, unused-argument
    def to_solver(self, params: Flow360Params, **kwargs) -> FreestreamFromMach:
        """
        returns configuration object in flow360 units system
        """
        return self.copy()


class ZeroFreestream(FreestreamBase):
    """
    :class: Zero velocity freestream component
    """

    model_type: Literal["ZeroMach"] = pd.Field("ZeroMach", alias="modelType", const=True)
    Mach: Literal[0] = pd.Field(0, const=True, displayed="Mach number")
    Mach_ref: pd.confloat(gt=1.0e-12) = pd.Field(alias="MachRef", displayed="Reference Mach number")
    mu_ref: PositiveFloat = pd.Field(alias="muRef", displayed="Dynamic viscosity [non-dim]")
    temperature: Union[Literal[-1], PositiveFloat] = pd.Field(
        alias="Temperature", displayed="Temperature [K]"
    )

    # pylint: disable=arguments-differ, unused-argument
    def to_solver(self, params: Flow360Params, **kwargs) -> ZeroFreestream:
        """
        returns configuration object in flow360 units system
        """
        return self.copy()


class FreestreamFromVelocity(FreestreamBase):
    """
    :class: Freestream component using dimensioned velocity
    """

    model_type: Literal["FromVelocity"] = pd.Field("FromVelocity", alias="modelType", const=True)
    velocity: VelocityType.Positive = pd.Field()
    velocity_ref: Optional[VelocityType.Positive] = pd.Field(
        alias="velocityRef", displayed="Reference velocity"
    )

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
    velocity_ref: VelocityType.Positive = pd.Field(
        alias="velocityRef", displayed="Reference velocity"
    )

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


FreestreamType = Union[
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
    pressure: PressureType.Positive = pd.Field()
    density: DensityType.Positive = pd.Field()
    viscosity: ViscosityType.Positive = pd.Field()

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

    model_type: Literal["AirPressure"] = pd.Field("AirPressure", alias="modelType", const=True)
    pressure: PressureType.Positive = pd.Field()
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

    model_type: Literal["AirDensity"] = pd.Field("AirDensity", alias="modelType", const=True)
    density: DensityType.Positive = pd.Field()
    temperature: TemperatureType = pd.Field()

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

FluidPropertyType = Union[AirDensityTemperature, AirPressureTemperature]


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
    twists: List[BETDiskTwist] = pd.Field(displayed="BET disk twists")
    chords: List[BETDiskChord] = pd.Field(displayed="BET disk chords")
    sectional_polars: List[BETDiskSectionalPolar] = pd.Field(
        alias="sectionalPolars", displayed="Sectional polars"
    )
    sectional_radiuses: List[float] = pd.Field(
        alias="sectionalRadiuses", displayed="Sectional radiuses"
    )

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_bet_disks_alphas_in_order(cls, values):
        """
        check order of alphas in BET disks
        """
        return _check_bet_disks_alphas_in_order(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_bet_disks_duplicate_chords_or_twists(cls, values):
        """
        check duplication of radial locations in chords or twists
        """
        return _check_bet_disks_duplicate_chords_or_twists(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_bet_disks_number_of_defined_polars(cls, values):
        """
        check number of polars
        """
        return _check_bet_disks_number_of_defined_polars(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_bet_disks_3d_coefficients_in_polars(cls, values):
        """
        check dimension of force coefficients in polars
        """
        return _check_bet_disks_3d_coefficients_in_polars(values)


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


class UserDefinedDynamic(Flow360BaseModel):
    """:class:`UserDefinedDynamic` class"""

    name: str = pd.Field(alias="dynamicsName")
    input_vars: List[str] = pd.Field(alias="inputVars")
    constants: Optional[Dict[str, float]] = pd.Field()
    output_vars: Optional[Dict[str, str]] = pd.Field(alias="outputVars")
    state_vars_initial_value: List[str] = pd.Field(alias="stateVarsInitialValue")
    update_law: List[str] = pd.Field(alias="updateLaw")
    input_boundary_patches: Optional[List[str]] = pd.Field(alias="inputBoundaryPatches")
    output_target_name: Optional[str] = pd.Field(alias="outputTargetName")


# pylint: disable=too-many-instance-attributes
class Flow360Params(Flow360BaseModel):
    """
    Flow360 solver parameters
    """

    unit_system: UnitSystemType = pd.Field(alias="unitSystem", mutable=False, discriminator="name")
    version: str = pd.Field(__version__, mutable=False)

    geometry: Optional[Geometry] = pd.Field(Geometry())
    fluid_properties: Optional[FluidPropertyType] = pd.Field(
        alias="fluidProperties", discriminator="model_type"
    )
    boundaries: Boundaries = pd.Field()
    initial_condition: Optional[InitialConditions] = pd.Field(
        alias="initialCondition", discriminator="type"
    )
    time_stepping: Optional[TimeStepping] = pd.Field(
        alias="timeStepping", default=SteadyTimeStepping(), discriminator="model_type"
    )
    turbulence_model_solver: Optional[TurbulenceModelSolverType] = pd.Field(
        alias="turbulenceModelSolver", discriminator="model_type"
    )
    transition_model_solver: Optional[TransitionModelSolver] = pd.Field(
        alias="transitionModelSolver"
    )
    heat_equation_solver: Optional[HeatEquationSolver] = pd.Field(alias="heatEquationSolver")
    freestream: FreestreamType = pd.Field(discriminator="model_type")
    bet_disks: Optional[List[BETDisk]] = pd.Field(alias="BETDisks")
    actuator_disks: Optional[List[ActuatorDisk]] = pd.Field(alias="actuatorDisks")
    porous_media: Optional[List[PorousMedium]] = pd.Field(alias="porousMedia")
    user_defined_dynamics: Optional[List[UserDefinedDynamic]] = pd.Field(
        alias="userDefinedDynamics"
    )
    surface_output: Optional[SurfaceOutput] = pd.Field(
        alias="surfaceOutput", default=SurfaceOutput()
    )
    volume_output: Optional[VolumeOutput] = pd.Field(alias="volumeOutput")
    slice_output: Optional[SliceOutput] = pd.Field(alias="sliceOutput")
    iso_surface_output: Optional[IsoSurfaceOutput] = pd.Field(alias="isoSurfaceOutput")
    monitor_output: Optional[MonitorOutput] = pd.Field(alias="monitorOutput")
    volume_zones: Optional[VolumeZones] = pd.Field(alias="volumeZones")
    aeroacoustic_output: Optional[AeroacousticOutput] = pd.Field(alias="aeroacousticOutput")

    navier_stokes_solver: Optional[NavierStokesSolverType] = pd.Field(alias="navierStokesSolver")

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

    # pylint: disable=super-init-not-called
    def __init__(self, filename: str = None, legacy_fallback: bool = False, **kwargs):
        if filename is not None or legacy_fallback:
            self._init_no_context(filename, legacy_fallback, **kwargs)
        else:
            self._init_with_context(**kwargs)

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

    def _init_with_context(self, **kwargs):
        kwargs = self._init_check_unit_system(**kwargs)
        super().__init__(unit_system=unit_system_manager.copy_current(), **kwargs)

    def _init_no_context(self, filename, legacy_fallback=False, **kwargs):
        if unit_system_manager.current is not None:
            raise Flow360RuntimeError(
                "When loading params from file: Flow360Params(filename), "
                "or from dict with the legacy_fallback flag set unit "
                "context must not be used."
            )

        if legacy_fallback:
            model_dict = self._init_handle_dict(**kwargs)
        else:
            model_dict = self._init_handle_file(filename=filename, **kwargs)

        version = model_dict.pop("version", None)
        unit_system = model_dict.get("unitSystem")
        if version is not None and unit_system is not None:
            if version != __version__:
                model_dict = updater(
                    version_from=version, version_to=__version__, params_as_dict=model_dict
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

    def flow360_json(self) -> str:
        """Generate a JSON representation of the model, as required by Flow360

        Returns
        -------
        json
            Returns JSON representation of the model.

        Example
        -------
        >>> params.flow360_json() # doctest: +SKIP
        """

        solver_params = self.to_solver()
        solver_params.set_will_export_to_flow360(True)
        solver_params_json = solver_params.json(encoder=flow360_json_encoder)
        return solver_params_json

    def flow360_dict(self) -> dict:
        """Generate a dict representation of the model, as required by Flow360

        Returns
        -------
        dict
            Returns dict representation of the model.

        Example
        -------
        >>> params.flow360_dict() # doctest: +SKIP
        """

        flow360_dict = json.loads(self.flow360_json())
        return flow360_dict

    def to_flow360_json(self, filename: str) -> NoReturn:
        """Exports :class:`Flow360Params` instance to .json file

        Example
        -------
        >>> params.to_flow360_json() # doctest: +SKIP
        """

        flow360_dict = self.flow360_dict()
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(flow360_dict, fh, indent=4, sort_keys=True)

    def append(self, params: Flow360Params, overwrite: bool = False):
        if not isinstance(params, Flow360Params):
            raise ValueError("params must be type of Flow360Params")
        super().append(params=params, overwrite=overwrite)

    @classmethod
    def construct(cls, filename: str = None, **kwargs) -> Flow360Params:
        """
        Creates a new model from trusted or pre-validated data.
        Default values are respected, but no other validation is performed.
        Behaves as if `Config.extra = 'allow'` was set since it adds all passed values
        """

        if filename is not None:
            model_dict = cls._init_handle_file(filename=filename, **kwargs)
        else:
            model_dict = kwargs

        # the default .construct() method will return field by both alias and field name so preprocessing here before
        # passing to .construct() method
        for name, field in cls.__fields__.items():
            if field.alt_alias and field.alias in model_dict:
                model_dict[name] = model_dict.pop(field.alias)

        return super().construct(**model_dict)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        allow_but_remove = ["runControl", "testControl"]
        include_hash: bool = True
        exclude_on_flow360_export = ["version", "unit_system"]

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_consistency_wall_function_and_surface_output(cls, values):
        """
        check consistency between wall function usage and surface output
        """
        return _check_consistency_wall_function_and_surface_output(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_temperature_consistency(cls, values):
        """
        check if temperature values in freestream and fluid_properties match
        """
        return _check_consistency_temperature(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_consistency_ddes_volume_output(cls, values):
        """
        check consistency between delayed detached eddy simulation and volume output
        """
        return _check_consistency_ddes_volume_output(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_tri_quad_boundaries(cls, values):
        """
        check tri_ and quad_ prefix in boundary names
        """
        return _check_tri_quad_boundaries(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_duplicate_boundary_name(cls, values):
        """
        check duplicated boundary names
        """
        return _check_duplicate_boundary_name(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_cht_solver_settings(cls, values):
        """
        check conjugate heat transfer settings
        """
        return _check_cht_solver_settings(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_equation_eval_frequency_for_unsteady_simulations(cls, values):
        """
        check equation evaluation frequency for unsteady simulations
        """
        return _check_equation_eval_frequency_for_unsteady_simulations(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_consistency_ddes_unsteady(cls, values):
        """
        check consistency between delayed detached eddy and unsteady simulation
        """
        return _check_consistency_ddes_unsteady(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_periodic_boundary_mapping(cls, values):
        """
        check periodic boundary mapping
        """
        return _check_periodic_boundary_mapping(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_aero_acoustics(cls, values):
        """
        check aeroacoustics settings
        """
        return _check_aero_acoustics(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_incompressible_navier_stokes_solver(cls, values):
        """
        check incompressible Navier-Stokes solver
        """
        return _check_incompressible_navier_stokes_solver(values)

    # pylint: disable=no-self-argument
    @pd.root_validator
    def check_numerical_dissipation_factor_output(cls, values):
        """
        Detect output of numericalDissipationFactor if not enabled.
        """
        return _check_numerical_dissipation_factor_output(values)


class Flow360MeshParams(Flow360BaseModel):
    """
    Flow360 mesh parameters
    """

    boundaries: MeshBoundary = pd.Field()
    sliding_interfaces: Optional[List[MeshSlidingInterface]] = pd.Field(alias="slidingInterfaces")

    def flow360_json(self, return_json: bool = True):
        """Generate a JSON representation of the model, as required by Flow360

        Parameters
        ----------
        return_json : bool, optional
            whether to return value or return None, by default True

        Returns
        -------
        json
            If return_json==True, returns JSON representation of the model.

        Example
        -------
        >>> params.to_flow360_json() # doctest: +SKIP
        """
        if return_json:
            return self.json()
        return None


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

    def flow360_json(self) -> str:
        """Generate a JSON representation of the model"""

        return self.json(encoder=flow360_json_encoder)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        extra = "allow"


# Legacy models for Flow360 updater, do not expose


class BETDiskLegacy(BETDisk, LegacyModel):
    """:class:`BETDiskLegacy` class"""

    def __init__(self, *args, **kwargs):
        with Flow360UnitSystem(verbose=False):
            super().__init__(*args, **kwargs)

    volume_name: Optional[str] = pd.Field(alias="volumeName")

    def update_model(self):
        model = {
            "rotationDirectionRule": self.rotation_direction_rule,
            "centerOfRotation": self.center_of_rotation,
            "axisOfRotation": self.axis_of_rotation,
            "numberOfBlades": self.number_of_blades,
            "radius": self.radius,
            "omega": self.omega,
            "chordRef": self.chord_ref,
            "thickness": self.thickness,
            "nLoadingNodes": self.n_loading_nodes,
            "bladeLineChord": self.blade_line_chord,
            "initialBladeDirection": self.initial_blade_direction,
            "tipGap": self.tip_gap,
            "MachNumbers": self.mach_numbers,
            "ReynoldsNumbers": self.reynolds_numbers,
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

    Reynolds: Optional[Union[pd.confloat(gt=0, allow_inf_nan=False), Literal["inf"]]] = pd.Field(
        displayed="Reynolds number"
    )
    Mach: Optional[NonNegativeFloat] = pd.Field()
    Mach_Ref: Optional[PositiveFloat] = pd.Field(alias="MachRef")
    mu_ref: Optional[PositiveFloat] = pd.Field(alias="muRef")
    temperature: Union[Literal[-1], PositiveFloat] = pd.Field(alias="Temperature")
    alpha: Optional[float] = pd.Field(alias="alphaAngle")
    beta: Optional[float] = pd.Field(alias="betaAngle", default=0)
    turbulent_viscosity_ratio: Optional[NonNegativeFloat] = pd.Field(
        alias="turbulentViscosityRatio"
    )

    def update_model(self) -> Flow360BaseModel:
        class _FreestreamTempModel(pd.BaseModel):
            """Helper class used to create
            the correct freestream from dict data"""

            field: FreestreamType = pd.Field(discriminator="model_type")

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
            velocity = model["field"].get("velocity")
            if velocity is not None:
                if velocity == 0:
                    model["field"]["modelType"] = "ZeroVelocity"
                    model["field"]["velocity"] = 0
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

            field: FluidPropertyType = pd.Field()

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
        "inf", alias="timeStepSize"
    )
    physical_steps: Optional[PositiveInt] = pd.Field(1, alias="physicalSteps")

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

        time_step = model["field"]["timeStepSize"]

        steady_state = isinstance(time_step, str) and time_step == "inf"

        if (
            steady_state
            and self.comments is not None
            and self.comments.get("timeStepSizeInSeconds") is not None
        ):
            step_unit = u.unyt_quantity(self.comments["timeStepSizeInSeconds"], "s")
            try_add_unit(model["field"], "timeStepSize", step_unit)

        if steady_state and model["field"]["physicalSteps"] == 1:
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
                "center": self.center * flow360_unit_system.length,
            },
        }

        try_set(model["referenceFrame"], "isDynamic", self.is_dynamic)
        try_set(model["referenceFrame"], "omegaRadians", self.omega)
        try_set(model["referenceFrame"], "omegaRadians", self.omega_radians)
        try_set(model["referenceFrame"], "omegaDegrees", self.omega_degrees)
        try_set(model["referenceFrame"], "thetaRadians", self.theta_radians)
        try_set(model["referenceFrame"], "thetaDegrees", self.theta_degrees)

        if self.comments is not None and self.comments.get("rpm") is not None:
            # pylint: disable=no-member
            omega = self.comments["rpm"] * u.rpm
            try_set(model["referenceFrame"], "omega", omega)

            if model["referenceFrame"].get("omegaRadians") is not None:
                del model["referenceFrame"]["omegaRadians"]
            if model["referenceFrame"].get("omegaDegrees") is not None:
                del model["referenceFrame"]["omegaDegrees"]

        options = ["OmegaRadians", "OmegaDegrees", "Expression", "Dynamic", "ReferenceFrame"]

        try_add_discriminator(model, "referenceFrame/modelType", options, FluidDynamicsVolumeZone)

        return FluidDynamicsVolumeZone.parse_obj(model)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_one_of = SlidingInterface.Config.require_one_of + ["omega"]


class BoundariesLegacy(Boundaries):
    """Legacy Boundaries class"""

    def __init__(self, *args, **kwargs):
        with Flow360UnitSystem(verbose=False):
            super().__init__(*args, **kwargs)


class VolumeZonesLegacy(VolumeZones):
    """Legacy VolumeZones class"""

    def __init__(self, *args, **kwargs):
        class _ReferenceFrameTempModel(pd.BaseModel):
            field: ReferenceFrameType = pd.Field(discriminator="model_type")

        options = ["OmegaRadians", "OmegaDegrees", "Expression", "Dynamic", "ReferenceFrame"]

        with Flow360UnitSystem(verbose=False):
            # Try to add discriminators to every volume zone,
            # to be removed (or rather, moved into update_model)
            # later after we fully decouple legacy models from
            # current models
            for value in kwargs.values():
                frame = value.get("referenceFrame")
                if frame is not None:
                    model = {"field": frame}
                    model = try_add_discriminator(
                        model, "field/modelType", options, _ReferenceFrameTempModel
                    )
                    value["referenceFrame"] = model["field"]

                volume_zone_type = value.get("modelType")

                if volume_zone_type == "HeatEquation":
                    value["modelType"] = HeatTransferVolumeZone.__fields__["model_type"].default
                if volume_zone_type == "NavierStokes":
                    value["modelType"] = FluidDynamicsVolumeZone.__fields__["model_type"].default

            super().__init__(*args, **kwargs)


class Flow360ParamsLegacy(LegacyModel):
    """:class: `Flow360ParamsLegacy` class"""

    geometry: Optional[GeometryLegacy] = pd.Field()
    freestream: Optional[FreestreamLegacy] = pd.Field()
    time_stepping: Optional[TimeSteppingLegacy] = pd.Field(alias="timeStepping")
    navier_stokes_solver: Optional[NavierStokesSolverLegacy] = pd.Field(alias="navierStokesSolver")
    turbulence_model_solver: Optional[Union[NoneSolver, TurbulenceModelSolverLegacy]] = pd.Field(
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
    boundaries: Optional[BoundariesLegacy] = pd.Field()
    # Needs decoupling from current model
    initial_condition: Optional[InitialConditions] = pd.Field(
        alias="initialCondition", discriminator="type"
    )
    # Needs decoupling from current model
    actuator_disks: Optional[List[ActuatorDisk]] = pd.Field(alias="actuatorDisks")
    # Needs decoupling from current model
    porous_media: Optional[List[PorousMedium]] = pd.Field(alias="porousMedia")
    # Needs decoupling from current model
    user_defined_dynamics: Optional[List[UserDefinedDynamic]] = pd.Field(
        alias="userDefinedDynamics"
    )
    # Needs decoupling from current model
    monitor_output: Optional[MonitorOutput] = pd.Field(alias="monitorOutput")
    volume_zones: Optional[VolumeZonesLegacy] = pd.Field(alias="volumeZones")
    # Needs decoupling from current model
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

        params["volume_zones"] = self.volume_zones
        if self.sliding_interfaces is not None:
            volume_zones = {}
            for interface in self.sliding_interfaces:
                volume_zone = try_update(interface)
                volume_name = interface.volume_name
                if isinstance(interface.volume_name, list):
                    volume_name = interface.volume_name[0]

                volume_zones[volume_name] = volume_zone
            params["volume_zones"] = VolumeZones(**volume_zones)
        elif self.volume_zones is not None:
            params["volume_zones"] = self.volume_zones

        if self._is_web_ui_generated(params.get("fluid_properties"), params.get("freestream")):
            context = SIUnitSystem(verbose=False)
        else:
            context = Flow360UnitSystem(verbose=False)

        with context:
            # Freestream, fluid properties, BET disks and volume zones filled beforehand.
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
                }
            )

            model = Flow360Params(**params)
            return model

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        allow_but_remove = ["runControl", "testControl"]
