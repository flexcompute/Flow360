"""Operating conditions for the simulation framework."""

from typing import Optional, Tuple, Union

import pydantic as pd

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.cached_model_base import CachedModelBase
from flow360.component.simulation.models.material import Air, FluidMaterialTypes
from flow360.component.simulation.unit_system import (
    AngleType,
    DensityType,
    LengthType,
    PressureType,
    TemperatureType,
    VelocityType,
    ViscosityType,
)
from flow360.log import log

# pylint: disable=no-member
VelocityVectorType = Union[VelocityType.Vector, Tuple[pd.StrictStr, pd.StrictStr, pd.StrictStr]]


class ThermalStateCache(Flow360BaseModel):
    """[INTERNAL] Cache for thermal state inputs"""

    # pylint: disable=no-member
    constructor: Optional[str] = None
    altitude: Optional[LengthType.Positive] = None
    temperature_offset: Optional[TemperatureType] = None
    temperature: Optional[TemperatureType.Positive] = None
    density: Optional[DensityType.Positive] = None
    material: Optional[FluidMaterialTypes] = None


class ThermalState(CachedModelBase):
    """
    Represents the thermal state of a fluid with specific properties.

    Attributes:
    -----------
    temperature : TemperatureType.Positive
        The temperature of the fluid, initialized to 288.15 K. This field is frozen and should not be modified after
        construction.
    density : DensityType.Positive
        The density of the fluid, initialized to 1.225 kg/m^3. This field is frozen and should not be modified after
        construction.
    material : FluidMaterialTypes
        The type of fluid material, initialized to Air(). This field is frozen and should not be modified after
        construction.
    """

    # pylint: disable=fixme
    # TODO: romove frozen and throw warning if temperature/density is modified after construction from atmospheric model
    temperature: TemperatureType.Positive = pd.Field(288.15 * u.K, frozen=True)
    density: DensityType.Positive = pd.Field(1.225 * u.kg / u.m**3, frozen=True)
    material: FluidMaterialTypes = pd.Field(Air(), frozen=True)
    _cached: ThermalStateCache = ThermalStateCache()

    # pylint: disable=no-self-argument, not-callable, unused-argument
    @CachedModelBase.model_constructor
    @pd.validate_call
    def from_standard_atmosphere(
        cls, altitude: LengthType.Positive = 0 * u.m, temperature_offset: TemperatureType = 0 * u.K
    ):
        """Constructs a thermal state from the standard atmosphere model."""
        # pylint: disable=fixme
        # TODO: add standard atmosphere implementation
        density = 1.225 * u.kg / u.m**3
        temperature = 288.15 * u.K

        state = cls(
            density=density,
            temperature=temperature,
            material=Air(),
        )

        return state

    @property
    def altitude(self) -> Optional[LengthType.Positive]:
        """Return user specified altitude."""
        if not self._cached.altitude:
            log.warning("Altitude not provided from input")
        return self._cached.altitude

    @property
    def temperature_offset(self) -> Optional[TemperatureType]:
        """Return user specified temperature offset."""
        if not self._cached.altitude:
            log.warning("Temperature offset not provided from input")
        return self._cached.temperature_offset

    @property
    def speed_of_sound(self) -> VelocityType.Positive:
        """Computes speed of sound."""
        # pylint: disable=fixme
        # TODO: implement
        # return self.material.speed_of_sound(self.temperature)
        return 343 * u.m / u.s

    @property
    def pressure(self) -> PressureType.Positive:
        """Computes pressure."""
        # pylint: disable=fixme
        # TODO: implement
        return 1.013e5 * u.Pa

    @property
    def dynamic_viscosity(self) -> ViscosityType.Positive:
        """Computes dynamic viscosity."""
        # pylint: disable=fixme
        # TODO: implement
        # return self.material.dynamic_viscosity(self.temperature)
        return 1.825e-5 * u.Pa * u.s

    @pd.validate_call
    def mu_ref(self, mesh_unit: LengthType.Positive) -> pd.PositiveFloat:
        """Computes nondimensional dynamic viscosity."""
        # TODO: use unit system for nondimensionalization
        return (self.dynamic_viscosity / (self.speed_of_sound * self.density * mesh_unit)).v.item()


class GenericReferenceConditionCache(Flow360BaseModel):
    """[INTERNAL] Cache for GenericReferenceCondition inputs"""

    constructor: Optional[str] = None
    velocity_magnitude: Optional[VelocityType.Positive] = None
    thermal_state: Optional[ThermalState] = None
    mach: Optional[pd.PositiveFloat] = None


class AerospaceConditionCache(Flow360BaseModel):
    """[INTERNAL] Cache for AerospaceCondition inputs"""

    constructor: Optional[str] = None
    alpha: Optional[AngleType] = None
    beta: Optional[AngleType] = None
    reference_velocity_magnitude: Optional[VelocityType.Positive] = None
    velocity_magnitude: Optional[VelocityType.NonNegative] = None
    thermal_state: Optional[ThermalState] = pd.Field(None, alias="atmosphere")
    mach: Optional[pd.NonNegativeFloat] = None
    reference_mach: Optional[pd.PositiveFloat] = None


class GenericReferenceCondition(CachedModelBase):
    """
    Operating condition defines the physical (non-geometrical) reference values for the problem.
    """

    velocity_magnitude: VelocityType.Positive
    thermal_state: ThermalState = ThermalState()
    _cached: GenericReferenceConditionCache = GenericReferenceConditionCache()

    # pylint: disable=no-self-argument, not-callable
    @CachedModelBase.model_constructor
    @pd.validate_call
    def from_mach(
        cls,
        mach: pd.PositiveFloat,
        thermal_state: ThermalState = ThermalState(),
    ):
        """Constructs a reference condition from Mach number and thermal state."""
        velocity_magnitude = mach * thermal_state.speed_of_sound
        return cls(velocity_magnitude=velocity_magnitude, thermal_state=thermal_state)

    @property
    def mach(self) -> pd.PositiveFloat:
        """Computes Mach number."""
        return self.velocity_magnitude / self.thermal_state.speed_of_sound


class AerospaceCondition(CachedModelBase):
    """A specialized GenericReferenceCondition for aerospace applications."""

    # pylint: disable=fixme
    # TODO: valildate reference_velocity_magnitude defined if velocity_magnitude=0
    alpha: AngleType = 0 * u.deg
    beta: AngleType = 0 * u.deg
    velocity_magnitude: VelocityType.NonNegative
    thermal_state: ThermalState = pd.Field(ThermalState(), alias="atmosphere")
    reference_velocity_magnitude: Optional[VelocityType.Positive] = None
    _cached: AerospaceConditionCache = AerospaceConditionCache()

    # pylint: disable=too-many-arguments, no-self-argument, not-callable
    @CachedModelBase.model_constructor
    @pd.validate_call
    def from_mach(
        cls,
        mach: pd.PositiveFloat,
        alpha: AngleType = 0 * u.deg,
        beta: AngleType = 0 * u.deg,
        thermal_state: ThermalState = ThermalState(),
        reference_mach: Optional[pd.PositiveFloat] = None,
    ):
        """Constructs a `AerospaceCondition` from Mach number and thermal state."""

        velocity_magnitude = mach * thermal_state.speed_of_sound

        reference_velocity_magnitude = (
            reference_mach * thermal_state.speed_of_sound if reference_mach else None
        )
        return cls(
            velocity_magnitude=velocity_magnitude,
            alpha=alpha,
            beta=beta,
            thermal_state=thermal_state,
            reference_velocity_magnitude=reference_velocity_magnitude,
        )

    # pylint: disable=no-self-argument, not-callable
    @CachedModelBase.model_constructor
    @pd.validate_call
    def from_stationary(
        cls,
        reference_velocity_magnitude: VelocityType.Positive,
        thermal_state: ThermalState = ThermalState(),
    ):
        """Constructs a `AerospaceCondition` for stationary conditions."""
        return cls(
            velocity_magnitude=0 * u.m / u.s,
            thermal_state=thermal_state,
            reference_velocity_magnitude=reference_velocity_magnitude,
        )

    @property
    def mach(self) -> pd.PositiveFloat:
        """Computes Mach number."""
        return self.velocity_magnitude / self.thermal_state.speed_of_sound

    # pylint: disable=fixme
    # TODO:  Add after model validation that reference_velocity_magnitude is set when velocity_magnitude is 0


# pylint: disable=fixme
# TODO: AutomotiveCondition
OperatingConditionTypes = Union[GenericReferenceCondition, AerospaceCondition]
