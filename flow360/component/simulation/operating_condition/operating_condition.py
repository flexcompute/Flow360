"""Operating conditions for the simulation framework."""

from typing import Literal, Optional, Tuple, Union

import pydantic as pd
from typing_extensions import Self

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.expressions import StringExpression
from flow360.component.simulation.framework.multi_constructor_model_base import (
    MultiConstructorBaseModel,
)
from flow360.component.simulation.models.material import Air, Water
from flow360.component.simulation.operating_condition.atmosphere_model import (
    StandardAtmosphereModel,
)
from flow360.component.simulation.unit_system import (
    AbsoluteTemperatureType,
    AngleType,
    DeltaTemperatureType,
    DensityType,
    LengthType,
    PressureType,
    VelocityType,
    ViscosityType,
)
from flow360.component.simulation.validation.validation_context import (
    CASE,
    CaseField,
    ConditionalField,
    context_validator,
    get_validation_info,
)
from flow360.log import log

# pylint: disable=no-member
VelocityVectorType = Union[
    Tuple[StringExpression, StringExpression, StringExpression], VelocityType.Vector
]


class ThermalStateCache(Flow360BaseModel):
    """[INTERNAL] Cache for thermal state inputs"""

    # pylint: disable=no-member
    altitude: Optional[LengthType] = None
    temperature_offset: Optional[DeltaTemperatureType] = None


class ThermalState(MultiConstructorBaseModel):
    """
    Represents the thermal state of a fluid with specific properties.

    Example
    -------

    >>> fl.ThermalState(
    ...     temperature=300 * fl.u.K,
    ...     density=1.225 * fl.u.kg / fl.u.m**3,
    ...     material=fl.Air()
    ... )

    ====
    """

    # pylint: disable=fixme
    # TODO: remove frozen and throw warning if temperature/density is modified after construction from atmospheric model
    type_name: Literal["ThermalState"] = pd.Field("ThermalState", frozen=True)
    temperature: AbsoluteTemperatureType = pd.Field(
        288.15 * u.K, frozen=True, description="The temperature of the fluid."
    )
    density: DensityType.Positive = pd.Field(
        1.225 * u.kg / u.m**3, frozen=True, description="The density of the fluid."
    )
    material: Air = pd.Field(Air(), frozen=True, description="The material of the fluid.")
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
        temperature_offset: DeltaTemperatureType = 0 * u.K,
    ):
        """
        Constructs a :class:`ThermalState` instance from the standard atmosphere model.

        Parameters
        ----------
        altitude : LengthType, optional
            The altitude at which the thermal state is calculated. Defaults to ``0 * u.m``.
        temperature_offset : DeltaTemperatureType, optional
            The temperature offset to be applied to the standard temperature at the given altitude.
            Defaults to ``0 * u.K``.

        Returns
        -------
        ThermalState
            A thermal state representing the atmospheric conditions at the specified altitude and temperature offset.

        Notes
        -----
        - This method uses the :class:`StandardAtmosphereModel` to compute the standard atmospheric
          conditions based on the given altitude.
        - The ``temperature_offset`` allows for adjustments to the standard temperature, simulating
          non-standard atmospheric conditions.

        Examples
        --------
        Create a thermal state at an altitude of 10,000 meters:

        >>> thermal_state = ThermalState.from_standard_atmosphere(altitude=10000 * u.m)
        >>> thermal_state.temperature
        <calculated_temperature>
        >>> thermal_state.density
        <calculated_density>

        Apply a temperature offset of -5 Fahrenheit at 5,000 meters:

        >>> thermal_state = ThermalState.from_standard_atmosphere(
        ...     altitude=5000 * u.m,
        ...     temperature_offset=-5 * u.delta_degF
        ... )
        >>> thermal_state.temperature
        <adjusted_temperature>
        >>> thermal_state.density
        <adjusted_density>
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
        if not self.private_attribute_input_cache.altitude:
            log.warning("Altitude not provided from input")
        return self.private_attribute_input_cache.altitude

    @property
    def temperature_offset(self) -> Optional[DeltaTemperatureType]:
        """Return user specified temperature offset."""
        if not self.private_attribute_input_cache.temperature_offset:
            log.warning("Temperature offset not provided from input")
        return self.private_attribute_input_cache.temperature_offset

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


class GenericReferenceConditionCache(Flow360BaseModel):
    """[INTERNAL] Cache for GenericReferenceCondition inputs"""

    thermal_state: Optional[ThermalState] = None
    mach: Optional[pd.PositiveFloat] = None


class GenericReferenceCondition(MultiConstructorBaseModel):
    """
    Operating condition defines the physical (non-geometrical) reference values for the problem.

    Example
    -------

    - Define :class:`GenericReferenceCondition` with :py:meth:`from_mach`:

      >>> fl.GenericReferenceCondition.from_mach(
      ...     mach=0.2,
      ...     thermal_state=ThermalState(),
      ... )

    - Define :class:`GenericReferenceCondition` with :py:attr:`velocity_magnitude`:

      >>> fl.GenericReferenceCondition(velocity_magnitude=40 * fl.u.m / fl.u.s)

    ====
    """

    type_name: Literal["GenericReferenceCondition"] = pd.Field(
        "GenericReferenceCondition", frozen=True
    )
    velocity_magnitude: Optional[VelocityType.Positive] = ConditionalField(
        context=CASE,
        description="Freestream velocity magnitude. Used as reference velocity magnitude"
        + " when :py:attr:`reference_velocity_magnitude` is not specified. Cannot change once specified.",
        frozen=True,
    )
    thermal_state: ThermalState = pd.Field(
        ThermalState(),
        description="Reference and freestream thermal state. Defaults to US standard atmosphere at sea level.",
    )
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
        return (self.velocity_magnitude / self.thermal_state.speed_of_sound).value

    @pd.field_validator("thermal_state", mode="after")
    @classmethod
    def _update_input_cache(cls, value, info: pd.ValidationInfo):
        setattr(info.data["private_attribute_input_cache"], info.field_name, value)
        return value


class AerospaceConditionCache(Flow360BaseModel):
    """[INTERNAL] Cache for AerospaceCondition inputs"""

    mach: Optional[pd.NonNegativeFloat] = None
    reynolds: Optional[pd.PositiveFloat] = None
    project_length_unit: Optional[LengthType.Positive] = None
    alpha: Optional[AngleType] = None
    beta: Optional[AngleType] = None
    temperature: Optional[AbsoluteTemperatureType] = None
    thermal_state: Optional[ThermalState] = pd.Field(None, alias="atmosphere")
    reference_mach: Optional[pd.PositiveFloat] = None


class AerospaceCondition(MultiConstructorBaseModel):
    """
    Operating condition for aerospace applications. Defines both reference parameters used to compute nondimensional
    coefficients in postprocessing and the default :class:`Freestream` boundary condition for the simulation.

    Example
    -------

    - Define :class:`AerospaceCondition` with :py:meth:`from_mach`:

      >>> fl.AerospaceCondition.from_mach(
      ...     mach=0,
      ...     alpha=-90 * fl.u.deg,
      ...     thermal_state=fl.ThermalState(),
      ...     reference_mach=0.69,
      ... )

    - Define :class:`AerospaceCondition` with :py:attr:`velocity_magnitude`:

      >>> fl.AerospaceCondition(velocity_magnitude=40 * fl.u.m / fl.u.s)

    ====
    """

    type_name: Literal["AerospaceCondition"] = pd.Field("AerospaceCondition", frozen=True)
    alpha: AngleType = ConditionalField(0 * u.deg, description="The angle of attack.", context=CASE)
    beta: AngleType = ConditionalField(0 * u.deg, description="The side slip angle.", context=CASE)
    velocity_magnitude: Optional[VelocityType.NonNegative] = ConditionalField(
        description="Freestream velocity magnitude. Used as reference velocity magnitude"
        + " when :py:attr:`reference_velocity_magnitude` is not specified.",
        context=CASE,
        frozen=True,
    )
    thermal_state: ThermalState = pd.Field(
        ThermalState(),
        alias="atmosphere",
        description="Reference and freestream thermal state. Defaults to US standard atmosphere at sea level.",
    )
    reference_velocity_magnitude: Optional[VelocityType.Positive] = CaseField(
        None,
        description="Reference velocity magnitude. Is required when :py:attr:`velocity_magnitude` is 0.",
        frozen=True,
    )
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
        Constructs an :class:`AerospaceCondition` instance from a Mach number and thermal state.

        Parameters
        ----------
        mach : float
            Freestream Mach number (non-negative).
            Used as reference Mach number when ``reference_mach`` is not specified.
        alpha : AngleType, optional
            The angle of attack. Defaults to ``0 * u.deg``.
        beta : AngleType, optional
            The side slip angle. Defaults to ``0 * u.deg``.
        thermal_state : ThermalState, optional
            Reference and freestream thermal state. Defaults to US standard atmosphere at sea level.
        reference_mach : float, optional
            Reference Mach number (positive). If provided, calculates the reference velocity magnitude.

        Returns
        -------
        AerospaceCondition
            An instance of :class:`AerospaceCondition` with the calculated velocity magnitude and provided parameters.

        Notes
        -----
        - The ``velocity_magnitude`` is calculated as ``mach * thermal_state.speed_of_sound``.
        - If ``reference_mach`` is provided, the ``reference_velocity_magnitude`` is calculated as
          ``reference_mach * thermal_state.speed_of_sound``.

        Examples
        --------
        Create an aerospace condition with a Mach number of 0.85:

        >>> condition = AerospaceCondition.from_mach(mach=0.85)
        >>> condition.velocity_magnitude
        <calculated_value>

        Specify angle of attack and side slip angle:

        >>> condition = AerospaceCondition.from_mach(mach=0.85, alpha=5 * u.deg, beta=2 * u.deg)

        Include a custom thermal state and reference Mach number:

        >>> custom_thermal = ThermalState(temperature=250 * u.K)
        >>> condition = AerospaceCondition.from_mach(
        ...     mach=0.85,
        ...     thermal_state=custom_thermal,
        ...     reference_mach=0.8
        ... )
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

    # pylint: disable=too-many-arguments
    @MultiConstructorBaseModel.model_constructor
    @pd.validate_call
    def from_mach_reynolds(
        cls,
        mach: pd.PositiveFloat,
        reynolds: pd.PositiveFloat,
        project_length_unit: Optional[LengthType.Positive],
        alpha: AngleType = 0 * u.deg,
        beta: AngleType = 0 * u.deg,
        temperature: AbsoluteTemperatureType = 288.15 * u.K,
        reference_mach: Optional[pd.PositiveFloat] = None,
    ):
        """
        Create an `AerospaceCondition` from Mach number and Reynolds number.

        This function computes the thermal state based on the given Mach number,
        Reynolds number, and temperature, and returns an `AerospaceCondition` object
        initialized with the computed thermal state and given aerodynamic angles.

        Parameters
        ----------
        mach : NonNegativeFloat
            Freestream Mach number (must be non-negative).
        reynolds : PositiveFloat
            Freestream Reynolds number defined with mesh unit (must be positive).
        project_length_unit: LengthType.Positive
            Project length unit.
        alpha : AngleType, optional
            Angle of attack. Default is 0 degrees.
        beta : AngleType, optional
            Sideslip angle. Default is 0 degrees.
        temperature : AbsoluteTemperatureType, optional
            Freestream static temperature (must be a positive temperature value). Default is 288.15 Kelvin.
        reference_mach : PositiveFloat, optional
            Reference Mach number. Default is None.

        Returns
        -------
        AerospaceCondition
            An instance of :class:`AerospaceCondition` with calculated velocity, thermal state and provided parameters.

        Example
        -------
        Example usage:

        >>> condition = operating_condition_from_mach_reynolds(
        ...     mach=0.85,
        ...     reynolds=1e6,
        ...     project_length_unit=1 * u.mm,
        ...     temperature=288.15 * u.K,
        ...     alpha=2.0 * u.deg,
        ...     beta=0.0 * u.deg,
        ...     reference_mach=0.85,
        ... )
        >>> print(condition)
        AerospaceCondition(...)

        """

        if temperature.units is u.K and temperature.value == 288.15:
            log.info("Default value of 288.15 K will be used as temperature.")

        if project_length_unit is None:
            validation_info = get_validation_info()
            if validation_info is None or validation_info.project_length_unit is None:
                raise ValueError("Project length unit must be provided.")
            project_length_unit = validation_info.project_length_unit

        material = Air()

        velocity = mach * material.get_speed_of_sound(temperature)

        density = (
            reynolds
            * material.get_dynamic_viscosity(temperature)
            / (velocity * project_length_unit)
        )

        thermal_state = ThermalState(temperature=temperature, density=density)

        velocity_magnitude = mach * thermal_state.speed_of_sound

        reference_velocity_magnitude = (
            reference_mach * thermal_state.speed_of_sound if reference_mach else None
        )

        log.info(
            """Density and viscosity were calculated based on input data, ThermalState will be automatically created."""
        )

        # pylint: disable=no-value-for-parameter
        return cls(
            velocity_magnitude=velocity_magnitude,
            alpha=alpha,
            beta=beta,
            thermal_state=thermal_state,
            reference_velocity_magnitude=reference_velocity_magnitude,
        )

    @pd.model_validator(mode="after")
    @context_validator(context=CASE)
    def check_valid_reference_velocity(self) -> Self:
        """Ensure reference velocity is provided when freestream velocity is 0."""
        if (
            self.velocity_magnitude is not None
            and self.velocity_magnitude.value == 0
            and self.reference_velocity_magnitude is None
        ):
            raise ValueError(
                "Reference velocity magnitude/Mach must be provided when freestream velocity magnitude/Mach is 0."
            )
        return self

    @property
    def mach(self) -> pd.PositiveFloat:
        """Computes Mach number."""
        return (self.velocity_magnitude / self.thermal_state.speed_of_sound).value

    @pd.field_validator("alpha", "beta", "thermal_state", mode="after")
    @classmethod
    def _update_input_cache(cls, value, info: pd.ValidationInfo):
        setattr(info.data["private_attribute_input_cache"], info.field_name, value)
        return value

    @pd.validate_call
    def flow360_reynolds_number(self, length_unit: LengthType.Positive):
        """
        Computes length_unit based Reynolds number.
        :math:`Re = \\rho_{\\infty} \\cdot U_{\\infty} \\cdot L_{grid}/\\mu_{\\infty}` where

        - :math:`\\rho_{\\infty}` is the freestream fluid density.
        - :math:`U_{\\infty}` is the freestream velocity magnitude.
        - :math:`L_{grid}` is physical length represented by unit length in the given mesh/geometry file.
        - :math:`\\mu_{\\infty}` is the dynamic eddy viscosity of the fluid of freestream.

        Parameters
        ----------
        length_unit : LengthType.Positive
            Physical length represented by unit length in the given mesh/geometry file.
        """

        return (
            self.thermal_state.density
            * self.velocity_magnitude
            * length_unit
            / self.thermal_state.dynamic_viscosity
        ).value


class LiquidOperatingCondition(Flow360BaseModel):
    """
    Operating condition for simulation of water as the only material.

    Example
    -------

    >>> fl.LiquidOperatingCondition(
    ...     velocity_magnitude=10 * fl.u.m / fl.u.s,
    ...     alpha=-90 * fl.u.deg,
    ...     beta=0 * fl.u.deg,
    ...     material=fl.Water(name="Water"),
    ...     reference_velocity_magnitude=5 * fl.u.m / fl.u.s,
    ... )

    ====
    """

    type_name: Literal["LiquidOperatingCondition"] = pd.Field(
        "LiquidOperatingCondition", frozen=True
    )
    alpha: AngleType = ConditionalField(0 * u.deg, description="The angle of attack.", context=CASE)
    beta: AngleType = ConditionalField(0 * u.deg, description="The side slip angle.", context=CASE)
    velocity_magnitude: Optional[VelocityType.NonNegative] = ConditionalField(
        context=CASE,
        description="Incoming flow velocity magnitude. Used as reference velocity magnitude"
        + " when :py:attr:`reference_velocity_magnitude` is not specified. Cannot change once specified.",
        frozen=True,
    )
    reference_velocity_magnitude: Optional[VelocityType.Positive] = CaseField(
        None,
        description="Reference velocity magnitude. Is required when :py:attr:`velocity_magnitude` is 0."
        " Used as the velocity scale for nondimensionalization.",
        frozen=True,
    )
    material: Water = pd.Field(
        Water(name="Water"),
        description="Type of liquid material used.",
    )

    @pd.model_validator(mode="after")
    @context_validator(context=CASE)
    def check_valid_reference_velocity(self) -> Self:
        """Ensure reference velocity is provided when inflow velocity is 0."""
        if (
            self.velocity_magnitude is not None
            and self.velocity_magnitude.value == 0
            and self.reference_velocity_magnitude is None
        ):
            raise ValueError(
                "Reference velocity magnitude must be provided when inflow velocity magnitude is 0."
            )
        return self


# pylint: disable=fixme
# TODO: AutomotiveCondition
OperatingConditionTypes = Union[
    GenericReferenceCondition, AerospaceCondition, LiquidOperatingCondition
]
