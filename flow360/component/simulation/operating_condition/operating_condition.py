"""Operating conditions for the simulation framework."""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import pydantic as pd
from typing_extensions import Self

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.multi_constructor_model_base import (
    MultiConstructorBaseModel,
)
from flow360.component.simulation.models.material import Air, FluidMaterialTypes
from flow360.component.simulation.operating_condition.atmosphere_model import (
    StandardAtmosphereModel,
)
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
VelocityVectorType = Union[Tuple[pd.StrictStr, pd.StrictStr, pd.StrictStr], VelocityType.Vector]


class ThermalStateCache(Flow360BaseModel):
    """[INTERNAL] Cache for thermal state inputs"""

    # pylint: disable=no-member
    altitude: Optional[LengthType] = None
    temperature_offset: Optional[TemperatureType] = None


class ThermalState(MultiConstructorBaseModel):
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
    type_name: Literal["ThermalState"] = pd.Field("ThermalState", frozen=True)
    temperature: TemperatureType.Positive = pd.Field(288.15 * u.K, frozen=True)
    density: DensityType.Positive = pd.Field(1.225 * u.kg / u.m**3, frozen=True)
    material: FluidMaterialTypes = pd.Field(Air(), frozen=True)
    private_attribute_input_cache: ThermalStateCache = ThermalStateCache()
    private_attribute_constructor: Literal["from_standard_atmosphere", "default"] = pd.Field(
        default="default", frozen=True
    )

    # pylint: disable=no-self-argument, not-callable, unused-argument
    @MultiConstructorBaseModel.model_constructor
    @pd.validate_call
    def from_standard_atmosphere(
        cls,
        altitude: LengthType = 0 * u.m,
        temperature_offset: TemperatureType = 0 * u.K,
    ):
        """Constructs a thermal state from the standard atmosphere model.

        Parameters:
        altitude (LengthType): The altitude at which the state is calculated.
        temperature_offset (TemperatureType): The offset to be applied to the standard temperature.

        Returns:
        ThermalState: The thermal state at the given altitude.
        """
        standard_atmosphere_model = StandardAtmosphereModel(
            altitude.in_units(u.m).value, temperature_offset.in_units(u.K).value
        )
        # Construct and return the thermal state
        state = cls(
            density=standard_atmosphere_model.density * u.kg / u.m**3,
            temperature=standard_atmosphere_model.temperature * u.K,
            material=Air(),
        )
        return state

    @property
    def altitude(self) -> Optional[LengthType]:
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
        return self.material.get_speed_of_sound(self.temperature)

    @property
    def pressure(self) -> PressureType.Positive:
        """Computes pressure."""
        return self.material.get_pressure(self.density, self.temperature)

    @property
    def dynamic_viscosity(self) -> ViscosityType.Positive:
        """Computes dynamic viscosity."""
        return self.material.get_dynamic_viscosity(self.temperature)

    # TODO: should we make this private_attribute?
    @pd.validate_call
    def _mu_ref(self, mesh_unit: LengthType.Positive) -> pd.PositiveFloat:
        """Computes nondimensional dynamic viscosity."""
        # TODO: use unit system for nondimensionalization
        return (self.dynamic_viscosity / (self.speed_of_sound * self.density * mesh_unit)).v.item()


class GenericReferenceConditionCache(Flow360BaseModel):
    """[INTERNAL] Cache for GenericReferenceCondition inputs"""

    thermal_state: Optional[ThermalState] = None
    mach: Optional[pd.PositiveFloat] = None


class GenericReferenceCondition(MultiConstructorBaseModel):
    """
    Operating condition defines the physical (non-geometrical) reference values for the problem.
    """

    type_name: Literal["GenericReferenceCondition"] = pd.Field(
        "GenericReferenceCondition", frozen=True
    )
    velocity_magnitude: VelocityType.Positive
    thermal_state: ThermalState = ThermalState()
    private_attribute_input_cache: GenericReferenceConditionCache = GenericReferenceConditionCache()

    # pylint: disable=no-self-argument, not-callable
    @MultiConstructorBaseModel.model_constructor
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


class AerospaceConditionCache(Flow360BaseModel):
    """[INTERNAL] Cache for AerospaceCondition inputs"""

    mach: Optional[pd.NonNegativeFloat] = None
    alpha: Optional[AngleType] = None
    beta: Optional[AngleType] = None
    thermal_state: Optional[ThermalState] = pd.Field(None, alias="atmosphere")
    reference_mach: Optional[pd.PositiveFloat] = None


class AerospaceCondition(MultiConstructorBaseModel):
    """A specialized GenericReferenceCondition for aerospace applications."""

    # pylint: disable=fixme
    # TODO: valildate reference_velocity_magnitude defined if velocity_magnitude=0
    type_name: Literal["AerospaceCondition"] = pd.Field("AerospaceCondition", frozen=True)
    alpha: AngleType = 0 * u.deg
    beta: AngleType = 0 * u.deg
    velocity_magnitude: VelocityType.NonNegative
    thermal_state: ThermalState = pd.Field(ThermalState(), alias="atmosphere")
    reference_velocity_magnitude: Optional[VelocityType.Positive] = None
    private_attribute_input_cache: AerospaceConditionCache = AerospaceConditionCache()

    # pylint: disable=too-many-arguments, no-self-argument, not-callable
    @MultiConstructorBaseModel.model_constructor
    @pd.validate_call
    def from_mach(
        cls,
        mach: pd.NonNegativeFloat,
        alpha: AngleType = 0 * u.deg,
        beta: AngleType = 0 * u.deg,
        thermal_state: ThermalState = ThermalState(),
        reference_mach: Optional[pd.PositiveFloat] = None,
    ):
        """
        Constructs a `AerospaceCondition` from Mach number and thermal state.

        Note:
        Decided to move `velocity==0 ref_velocity is not None` check to dedicated validator because user can
        still construct by just calling AerospaceCondition()
        """

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

    @pd.model_validator(mode="after")
    def check_valid_reference_velocity(self) -> Self:
        """Ensure reference velocity is provided when freestream velocity is 0."""
        if self.velocity_magnitude.value == 0 and self.reference_velocity_magnitude is None:
            raise ValueError(
                "Reference velocity magnitude/Mach must be provided when freestream velocity magnitude/Mach is 0."
            )
        return self

    @property
    def mach(self) -> pd.PositiveFloat:
        """Computes Mach number."""
        return self.velocity_magnitude / self.thermal_state.speed_of_sound

    # pylint: disable=fixme
    # TODO:  Add after model validation that reference_velocity_magnitude is set when velocity_magnitude is 0


# pylint: disable=fixme
# TODO: AutomotiveCondition
OperatingConditionTypes = Union[GenericReferenceCondition, AerospaceCondition]
