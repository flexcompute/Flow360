"""Material classes for the simulation framework."""

from typing import Literal, Union

import pydantic as pd
import unyt as u
from numpy import sqrt

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.physical_dimensions import (
    AbsoluteTemperature,
    Density,
    MolarMass,
    Pressure,
    SpecificHeatCapacity,
    ThermalConductivity,
    Velocity,
    Viscosity,
)


def compute_cp_over_r(coeffs, temperature):
    """
    Compute cp/R from NASA 9-coefficient polynomial.

    cp/R = a0*T^-2 + a1*T^-1 + a2 + a3*T + a4*T^2 + a5*T^3 + a6*T^4
    """
    temp = temperature
    return (
        coeffs[0] * temp ** (-2)
        + coeffs[1] * temp ** (-1)
        + coeffs[2]
        + coeffs[3] * temp
        + coeffs[4] * temp**2
        + coeffs[5] * temp**3
        + coeffs[6] * temp**4
    )


def compute_gamma_from_coefficients(coeffs, temperature):
    """
    Compute specific heat ratio (gamma) from NASA 9-coefficient polynomial.

    gamma = cp/cv = (cp/R) / (cp/R - 1)
    """
    cp_r = compute_cp_over_r(coeffs, temperature)
    cv_r = cp_r - 1
    return cp_r / cv_r


class MaterialBase(Flow360BaseModel):
    """
    Basic properties required to define a material.
    For example: young's modulus, viscosity as an expression of temperature, etc.
    """

    type: str = pd.Field()
    name: str = pd.Field()


class NASA9CoefficientSet(Flow360BaseModel):
    """
    Represents a set of 9 NASA polynomial coefficients for a specific temperature range.
    """

    type_name: Literal["NASA9CoefficientSet"] = pd.Field("NASA9CoefficientSet", frozen=True)
    temperature_range_min: AbsoluteTemperature.Float64 = pd.Field(
        description="Minimum temperature for which this coefficient set is valid."
    )
    temperature_range_max: AbsoluteTemperature.Float64 = pd.Field(
        description="Maximum temperature for which this coefficient set is valid."
    )
    coefficients: list[float] = pd.Field(
        description="Nine NASA polynomial coefficients [a0, a1, a2, a3, a4, a5, a6, a7, a8]. "
        "a0-a6 are cp/R polynomial coefficients, a7 is the enthalpy integration constant, "
        "and a8 is the entropy integration constant."
    )

    @pd.field_validator("coefficients", mode="after")
    @classmethod
    def validate_coefficients(cls, v):
        """Validate that exactly 9 coefficients are provided."""
        if len(v) != 9:
            raise ValueError(f"NASA 9-coefficient polynomial requires exactly 9 coefficients, got {len(v)}")
        return v

    @pd.field_validator("temperature_range_max", mode="after")
    @classmethod
    def validate_temperature_range_order(cls, v, info):
        """Validate that temperature_range_min < temperature_range_max."""
        t_min = info.data.get("temperature_range_min")
        if t_min is not None:
            t_min_k = t_min.to("K").v.item()
            t_max_k = v.to("K").v.item()
            if t_min_k >= t_max_k:
                raise ValueError(f"temperature_range_min ({t_min}) must be less than " f"temperature_range_max ({v})")
        return v


class NASA9Coefficients(Flow360BaseModel):
    """
    NASA 9-coefficient polynomial coefficients for computing temperature-dependent thermodynamic properties.
    """

    type_name: Literal["NASA9Coefficients"] = pd.Field("NASA9Coefficients", frozen=True)
    temperature_ranges: list[NASA9CoefficientSet] = pd.Field(
        min_length=1,
        max_length=5,
        description="List of NASA 9-coefficient sets for different temperature ranges. "
        "Must be ordered by increasing temperature and be continuous. Maximum 5 ranges supported.",
    )

    @pd.model_validator(mode="after")
    def validate_temperature_continuity(self):
        """Validate that temperature ranges are continuous and non-overlapping."""
        for i in range(len(self.temperature_ranges) - 1):
            current_max = self.temperature_ranges[i].temperature_range_max
            next_min = self.temperature_ranges[i + 1].temperature_range_min
            if current_max != next_min:
                raise ValueError(
                    f"Temperature ranges must be continuous: range {i} max "
                    f"({current_max}) must equal range {i+1} min ({next_min})"
                )
        return self

    @pd.validate_call
    def get_coefficients_at_temperature(self, temp_k: float) -> list:
        """
        Get the NASA 9 coefficients for a given temperature.
        """
        for coeff_set in self.temperature_ranges:
            t_min = coeff_set.temperature_range_min.to("K").v.item()
            t_max = coeff_set.temperature_range_max.to("K").v.item()
            if t_min <= temp_k <= t_max:
                return list(coeff_set.coefficients)

        return list(self.temperature_ranges[0].coefficients)


class Sutherland(Flow360BaseModel):
    """
    Represents Sutherland's law for calculating dynamic viscosity.
    """

    reference_viscosity: Viscosity.NonNegativeFloat64 = pd.Field(
        description="The reference dynamic viscosity at the reference temperature."
    )
    reference_temperature: AbsoluteTemperature.Float64 = pd.Field(
        description="The reference temperature associated with the reference viscosity."
    )
    effective_temperature: AbsoluteTemperature.Float64 = pd.Field(
        description="The effective temperature constant used in Sutherland's formula."
    )

    @pd.validate_call
    def get_dynamic_viscosity(self, temperature: AbsoluteTemperature.Float64) -> Viscosity.NonNegativeFloat64:
        """
        Calculates the dynamic viscosity at a given temperature using Sutherland's law.
        """
        return self.reference_viscosity * float(
            pow(temperature / self.reference_temperature, 1.5)
            * (self.reference_temperature + self.effective_temperature)
            / (temperature + self.effective_temperature)
        )


# Default laminar Schmidt number applied to a species whose schmidt_number is
# left unspecified in a multi-species transport declaration. Documented here so
# the translator and the Python model agree on the same fill value.
DEFAULT_SCHMIDT_NUMBER = 0.7

# Default turbulent Schmidt number Sc_t for the constant-Schmidt mixture
# diffusion closure D_t = mu_t / (rho * Sc_t). Distinct from DEFAULT_SCHMIDT_NUMBER
# above: Sc_t is a *modelling closure* for the turbulent flux, not a molecular
# property, and is conventionally 0.7-0.9 in engineering CFD. The numerical value
# happens to coincide with the laminar default here, but the two are physically
# different quantities and are kept separate so reader intent is unambiguous.
DEFAULT_TURBULENT_SCHMIDT_NUMBER = 0.7


class FrozenSpecies(Flow360BaseModel):
    """
    A single entry in a pre-mixed thermally-perfect gas (``ThermallyPerfectGas``).

    "Frozen" refers to the composition: the species' mass fraction is fixed at
    the time the simulation parameters are built, and the per-species NASA-9
    coefficients are blended into a single mixture polynomial before the
    simulation runs. The runtime then sees one fixed-composition gas, never
    transports the individual species.

    For *actively-transported* species (variable composition, runtime mass-fraction
    transport equations, composition-dependent thermo/transport), use the
    :class:`Species` class with :class:`SpeciesTransportModel` instead.
    """

    type_name: Literal["FrozenSpecies"] = pd.Field("FrozenSpecies", frozen=True)
    name: str = pd.Field(description="Species name (e.g., 'N2', 'O2', 'Ar')")
    nasa_9_coefficients: NASA9Coefficients = pd.Field(description="NASA 9-coefficient polynomial for this species")
    mass_fraction: pd.PositiveFloat = pd.Field(
        description="Mass fraction of this species (must sum to 1 across all species in mixture)"
    )


class Species(Flow360BaseModel):
    """
    An actively-transported chemical species in a variable-composition gas mixture.

    Unlike :class:`FrozenSpecies` (an entry in a pre-mixed gas whose composition is
    fixed when the simulation parameters are built), ``Species`` describes a species
    the runtime transports as a mass-fraction field. The full set of thermodynamic
    and transport properties needed for the multi-species path is therefore required
    at construction time:

    - NASA-9 polynomial for cp(T), h(T), s(T) (frozen-composition / non-reacting
      regime; see the data-source citation in any library entry for the validity
      window).
    - Molecular weight, to form the mixture gas constant
      ``R_mix = R_u * sum_s(Y_s / W_s)``.
    - Per-species Sutherland viscosity model, blended into the mixture viscosity
      via Wilke's rule.
    - Laminar Schmidt number (defaults to ``DEFAULT_SCHMIDT_NUMBER``), used by the
      constant-Schmidt Fickian diffusion closure.

    Used inside :class:`SpeciesTransportModel`. The ``mass_fraction`` value defines
    this species' contribution to the *reference* mixture composition (which the
    translator emits as ``referenceStateNondim``); at runtime the actual mass
    fraction varies in space and time.
    """

    type_name: Literal["Species"] = pd.Field("Species", frozen=True)
    name: str = pd.Field(description="Species name (e.g., 'N2', 'O2', 'H2O')")
    nasa_9_coefficients: NASA9Coefficients = pd.Field(
        description="NASA 9-coefficient polynomial for this species' thermodynamic data."
    )
    mass_fraction: pd.PositiveFloat = pd.Field(
        description="Reference mass fraction of this species in the mixture. Mass fractions "
        "across all species in a SpeciesTransportModel must sum to 1.0."
    )
    molecular_weight: MolarMass.PositiveFloat64 = pd.Field(
        description="Molecular weight (molar mass) of this species, e.g. ``28.0134 * u.g / u.mol``. "
        "Used to form the mixture gas constant R_mix = R_u * sum_s(Y_s / W_s). "
        "Any per-mole equivalent is accepted (g/mol, kg/mol, kg/kmol). Absolute units "
        "matter only at the Python layer: the translator collapses each species' W_s into "
        "a dimensionless ``inverseMolecularWeightRatioToReference = W_ref / W_s`` before "
        "the solver sees it, so the value the solver consumes is unit-agnostic."
    )
    dynamic_viscosity: Sutherland = pd.Field(
        description="Per-species Sutherland viscosity model, used by Wilke's rule "
        "to form the mixture dynamic viscosity."
    )
    schmidt_number: pd.PositiveFloat = pd.Field(
        DEFAULT_SCHMIDT_NUMBER,
        description="Laminar Schmidt number Sc_s = mu / (rho * D_s) for this species, used by "
        f"the constant-Schmidt Fickian diffusion model. Defaults to {DEFAULT_SCHMIDT_NUMBER}.",
    )

    @classmethod
    def from_library(cls, name: str) -> "Species":
        """Look up a species in the curated library and return an independent copy.

        Returns a fresh :class:`Species` carrying the library's NASA-9 polynomial,
        molecular weight, and Sutherland viscosity, with ``mass_fraction`` set
        to a 1.0 placeholder. Override any field by assigning to the returned
        object's attribute; :class:`Species` has ``validate_assignment=True`` so
        the field validators re-run on each assignment.

        Parameters
        ----------
        name :
            Canonical library species name (e.g. ``"N2"``, ``"O2"``, ``"H2O"``).
            See the library module docstring for the full roster.

        Returns
        -------
        Species
            An independent deep copy of the library entry. Mutating the returned
            object does not affect the library or any subsequent call.

        Raises
        ------
        ValueError
            If ``name`` is not in the library. The error message lists all
            available species names.

        Examples
        --------
        >>> n2 = Species.from_library("N2")
        >>> n2.mass_fraction = 0.767
        >>> o2 = Species.from_library("O2")
        >>> o2.mass_fraction = 0.233
        >>> o2.schmidt_number = 0.72
        >>> # Custom Sutherland override for a special application:
        >>> h2o = Species.from_library("H2O")
        >>> h2o.mass_fraction = 0.05
        >>> h2o.dynamic_viscosity = Sutherland(
        ...     reference_viscosity=1.2e-5 * u.Pa * u.s,
        ...     reference_temperature=350 * u.K,
        ...     effective_temperature=900 * u.K,
        ... )
        """
        # Lazy import to avoid a circular dependency at module load.
        from flow360_schema.models.simulation.models.species_library import _build_species

        return _build_species(name)


def _validate_unique_species_names(species: "list[FrozenSpecies | Species]") -> None:
    """Validate that all species have unique names."""
    names = [s.name for s in species]
    if len(names) != len(set(names)):
        duplicates = [name for name in names if names.count(name) > 1]
        raise ValueError(f"Species names must be unique. Duplicates found: {set(duplicates)}")


def _validate_temperature_ranges_match(species: "list[FrozenSpecies | Species]") -> None:
    """Validate all species share identical temperature range boundaries."""
    if len(species) < 2:
        return
    ref_ranges = species[0].nasa_9_coefficients.temperature_ranges
    for s in species[1:]:
        ranges = s.nasa_9_coefficients.temperature_ranges
        if len(ranges) != len(ref_ranges):
            raise ValueError(
                f"Species '{s.name}' has {len(ranges)} temperature ranges, "
                f"but '{species[0].name}' has {len(ref_ranges)}. "
                "All species must have the same number of temperature ranges."
            )
        for i, (r1, r2) in enumerate(zip(ref_ranges, ranges, strict=False)):
            if (
                r1.temperature_range_min != r2.temperature_range_min
                or r1.temperature_range_max != r2.temperature_range_max
            ):
                raise ValueError(
                    f"Temperature range {i} boundaries mismatch between species "
                    f"'{species[0].name}' and '{s.name}'. "
                    "All species must use the same temperature range boundaries."
                )


class ThermallyPerfectGas(Flow360BaseModel):
    """
    Multi-species thermally perfect gas model.
    """

    type_name: Literal["ThermallyPerfectGas"] = pd.Field("ThermallyPerfectGas", frozen=True)
    species: list[FrozenSpecies] = pd.Field(
        min_length=1,
        description="List of species with their NASA 9 coefficients and mass fractions. "
        "Mass fractions must sum to 1.0.",
    )

    @pd.model_validator(mode="after")
    def validate_mass_fractions_sum_to_one(self):
        """Validate that mass fractions sum to 1 and re-normalize if within tolerance."""
        total = sum(s.mass_fraction for s in self.species)
        if abs(total - 1.0) > 1.0e-3:
            raise ValueError(f"Mass fractions must sum to 1.0, got {total}")
        if total != 1.0:
            for s in self.species:
                s.mass_fraction = s.mass_fraction / total
        return self

    @pd.model_validator(mode="after")
    def validate_unique_species_names(self):
        """Validate that all species have unique names."""
        _validate_unique_species_names(self.species)
        return self

    @pd.model_validator(mode="after")
    def validate_temperature_ranges_match(self):
        """Validate all species have matching temperature range boundaries."""
        _validate_temperature_ranges_match(self.species)
        return self


class SpeciesTransportModel(Flow360BaseModel):
    """
    Variable-composition (multi-species) transport model.

    Unlike :class:`ThermallyPerfectGas` (a pre-mixed model whose species mass
    fractions only define a fixed mixture), this model declares an *explicitly
    transported* set of species: the runtime solves N-1 mass-fraction transport
    equations and forms composition-dependent thermodynamics and transport. Each
    species must therefore carry its molecular weight and a per-species viscosity
    model (for the Wilke mixture rule); the laminar Schmidt number defaults to
    ``DEFAULT_SCHMIDT_NUMBER`` when omitted.

    The species ``mass_fraction`` values define the reference mixture composition
    that the translator emits as ``referenceStateNondim``.
    """

    type_name: Literal["SpeciesTransportModel"] = pd.Field("SpeciesTransportModel", frozen=True)
    species: list[Species] = pd.Field(
        min_length=1,
        description="List of transported species. Reference mass fractions must sum to 1.0; "
        "molecular weight, Sutherland viscosity, and Schmidt number are required at construction "
        "time by the Species type itself.",
    )
    turbulent_schmidt_number: pd.PositiveFloat = pd.Field(
        DEFAULT_TURBULENT_SCHMIDT_NUMBER,
        description="Turbulent Schmidt number Sc_t used to form the turbulent species "
        "diffusivity D_t = mu_t / (rho * Sc_t). Distinct from the laminar Schmidt number "
        "Sc on each Species (Sc_t is a turbulence-closure modelling constant, not a "
        "molecular property).",
    )
    diffusion_model: Literal["constantSchmidt"] = pd.Field(
        "constantSchmidt",
        description="Mass-diffusion closure. Currently, only constant-Schmidt Fickian diffusion is supported.",
    )

    @pd.model_validator(mode="after")
    def validate_mass_fractions_sum_to_one(self):
        """Validate that reference mass fractions sum to 1 and re-normalize if within tolerance."""
        total = sum(s.mass_fraction for s in self.species)
        if abs(total - 1.0) > 1.0e-3:
            raise ValueError(f"Mass fractions must sum to 1.0, got {total}")
        if total != 1.0:
            for s in self.species:
                s.mass_fraction = s.mass_fraction / total
        return self

    @pd.model_validator(mode="after")
    def validate_unique_species_names(self):
        """Validate that all species have unique names."""
        _validate_unique_species_names(self.species)
        return self

    @pd.model_validator(mode="after")
    def validate_temperature_ranges_match(self):
        """Validate all species have matching temperature range boundaries."""
        _validate_temperature_ranges_match(self.species)
        return self


class Air(MaterialBase):
    """
    Calorically-perfect-gas (CPG) preset for air: constant ``gamma = 1.4``, constant
    ``R = 287.0529 J/(kg*K)``, and a standard Sutherland viscosity model.

    For temperature-dependent thermodynamics (NASA-9 polynomials) or variable-
    composition multi-species transport, use :class:`Gas` instead. Air rejects
    ``thermally_perfect_gas`` and ``species_transport_model`` at construction time
    with a message pointing at the Gas replacement.
    """

    type: Literal["air"] = pd.Field("air", frozen=True)
    name: str = pd.Field("air")
    dynamic_viscosity: Sutherland | Viscosity.NonNegativeFloat64 = pd.Field(
        Sutherland(
            reference_viscosity=1.716e-5 * u.Pa * u.s,
            reference_temperature=273.15 * u.K,
            effective_temperature=110.4 * u.K,
        ),
        description=(
            "The dynamic viscosity model or value for air. Defaults to a `Sutherland` "
            "model with standard atmospheric conditions."
        ),
    )
    prandtl_number: pd.PositiveFloat = pd.Field(
        0.72,
        description="Laminar Prandtl number. Default is 0.72 for air.",
    )
    turbulent_prandtl_number: pd.PositiveFloat = pd.Field(
        0.9,
        description="Turbulent Prandtl number. Default is 0.9.",
    )

    @pd.model_validator(mode="before")
    @classmethod
    def _reject_gas_only_fields(cls, data):
        """Reject ``thermally_perfect_gas`` / ``species_transport_model`` at construction.

        Both moved to :class:`Gas`; Air is now a CPG-only preset. Catching them in a
        before-validator lets us emit a paste-ready replacement signature instead of
        Pydantic's generic ``Extra inputs are not permitted`` message.
        """
        if not isinstance(data, dict):
            return data
        if "thermally_perfect_gas" in data:
            raise ValueError(
                "Air is now a calorically-perfect-gas (constant gamma) preset and "
                "no longer accepts 'thermally_perfect_gas'. For temperature-dependent "
                "thermodynamics, use fl.Gas(name=..., dynamic_viscosity=..., "
                "thermally_perfect_gas=fl.ThermallyPerfectGas(species=[...])) instead."
            )
        if "species_transport_model" in data:
            raise ValueError(
                "Air no longer accepts 'species_transport_model'. For variable-"
                "composition multi-species transport, use fl.Gas(name=..., "
                "dynamic_viscosity=..., species_transport_model=fl.SpeciesTransportModel("
                "species=[...])) instead."
            )
        return data

    def get_specific_heat_ratio(self, temperature: AbsoluteTemperature.Float64) -> pd.PositiveFloat:
        """CPG: constant gamma = 1.4 (temperature-independent)."""
        del temperature  # unused; CPG gamma is constant.
        return 1.4

    @property
    def gas_constant(self) -> SpecificHeatCapacity.PositiveFloat64:
        """Standard-air specific gas constant: R = R_u / W_air = 287.0529 J/(kg*K)."""
        return 287.0529 * u.m**2 / u.s**2 / u.K

    @pd.validate_call
    def get_pressure(
        self, density: Density.PositiveFloat64, temperature: AbsoluteTemperature.Float64
    ) -> Pressure.PositiveFloat64:
        """Ideal-gas pressure: p = rho * R * T."""
        temperature = temperature.to("K")
        return density * self.gas_constant * temperature

    @pd.validate_call
    def get_speed_of_sound(self, temperature: AbsoluteTemperature.Float64) -> Velocity.PositiveFloat64:
        """Speed of sound: sqrt(gamma * R * T) with gamma = 1.4."""
        temperature = temperature.to("K")
        return sqrt(1.4 * self.gas_constant * temperature)

    @pd.validate_call
    def get_dynamic_viscosity(self, temperature: AbsoluteTemperature.Float64) -> Viscosity.NonNegativeFloat64:
        """Dynamic viscosity at T from the configured Sutherland model (or constant)."""
        if temperature.units is u.degC or temperature.units is u.degF:
            temperature = temperature.to("K")
        if isinstance(self.dynamic_viscosity, Sutherland):
            return self.dynamic_viscosity.get_dynamic_viscosity(temperature)
        return self.dynamic_viscosity


class Gas(MaterialBase):
    """
    Customizable gas material -- either thermally perfect (NASA-9 polynomials) or
    variable-composition multi-species transport.

    Exactly one of :attr:`thermally_perfect_gas` or :attr:`species_transport_model`
    must be set; the two are mutually exclusive. For the constant-gamma standard-air
    case use the :class:`Air` preset instead.
    """

    type: Literal["gas"] = pd.Field("gas", frozen=True)
    name: str = pd.Field(description="Name identifying this gas (e.g. 'methane-air', 'N2/He mix').")
    dynamic_viscosity: Sutherland | Viscosity.NonNegativeFloat64 | None = pd.Field(
        None,
        description=(
            "Material-level viscosity model used for the ``thermally_perfect_gas`` path "
            "(both the freestream ``muRef`` for non-dimensionalization and the runtime "
            "viscosity). Required when ``thermally_perfect_gas`` is set; ignored when "
            "``species_transport_model`` is set, because viscosity in that case is "
            "Wilke-mixed from the per-species Sutherland fits at every temperature, "
            "including the freestream reference."
        ),
    )
    prandtl_number: pd.PositiveFloat = pd.Field(
        0.72,
        description="Laminar Prandtl number. Default is 0.72.",
    )
    turbulent_prandtl_number: pd.PositiveFloat = pd.Field(
        0.9,
        description="Turbulent Prandtl number. Default is 0.9.",
    )
    thermally_perfect_gas: ThermallyPerfectGas | None = pd.Field(
        None,
        description=(
            "Pre-mixed thermally perfect gas model (NASA 9-coefficient polynomials). "
            "Mutually exclusive with `species_transport_model`."
        ),
    )
    species_transport_model: SpeciesTransportModel | None = pd.Field(
        None,
        description=(
            "Variable-composition multi-species transport. The runtime transports the "
            "declared species' mass fractions and forms composition-dependent thermo "
            "and transport. Mutually exclusive with `thermally_perfect_gas`."
        ),
    )

    @pd.model_validator(mode="after")
    def _validate_exactly_one_gas_model(self):
        has_tpg = self.thermally_perfect_gas is not None
        has_stm = self.species_transport_model is not None
        if not (has_tpg or has_stm):
            raise ValueError(
                "Gas requires exactly one of 'thermally_perfect_gas' or "
                "'species_transport_model' to be set. For constant gamma = 1.4 standard "
                "air, use fl.Air() instead."
            )
        if has_tpg and has_stm:
            raise ValueError(
                "Gas accepts only one of 'thermally_perfect_gas' or "
                "'species_transport_model' -- they would silently compete to define the "
                "gas. Set only the one that describes your physics."
            )
        # dynamic_viscosity is only consumed by the TPG path (both for runtime
        # viscosity and the freestream muRef). For STM, viscosity is Wilke-mixed
        # from per-species data, so the material-level field would silently sit
        # unused -- omit it instead.
        if has_tpg and self.dynamic_viscosity is None:
            raise ValueError(
                "Gas with 'thermally_perfect_gas' requires 'dynamic_viscosity' to be set "
                "(used for both the runtime viscosity and the freestream muRef)."
            )
        return self

    def get_specific_heat_ratio(self, temperature: AbsoluteTemperature.Float64) -> pd.PositiveFloat:
        """Specific heat ratio gamma at T from the configured gas model.

        For ``species_transport_model``, gamma is the proper mass+mole-weighted mixture
        value at the reference composition. For ``thermally_perfect_gas``, it is the
        mass-weighted NASA-9 blend.
        """
        temp_k = temperature.to("K").v.item()
        if self.species_transport_model is not None:
            cp_over_r_mix = self._compute_species_cp_over_r_mix(temp_k)
            return cp_over_r_mix / (cp_over_r_mix - 1.0)
        coeffs = self._get_coefficients_at_temperature(temp_k)
        return compute_gamma_from_coefficients(coeffs, temp_k)

    def _get_coefficients_at_temperature(self, temp_k: float) -> list:
        """Effective NASA-9 coefficients at T (pre-mixed TPG path; mass-weighted blend)."""
        coeffs = [0.0] * 9
        for species in self.thermally_perfect_gas.species:
            species_coeffs = species.nasa_9_coefficients.get_coefficients_at_temperature(temp_k)
            for i in range(9):
                coeffs[i] += species.mass_fraction * species_coeffs[i]
        return coeffs

    def _compute_species_cp_over_r_mix(self, temp_k: float) -> float:
        """cp_mix / R_mix at ``temp_k`` from species_transport_model reference composition.

        Per the standard mixture relations:
            cp_mix / R_mix = sum_s Y_s (cp_s / R_s) (W_ref / W_s) / sum_s Y_s (W_ref / W_s)
        where W_ref = 1 / sum_s (Y_s / W_s) is the mixture molar mass.
        """
        stm = self.species_transport_model
        inv_w_ref = sum(s.mass_fraction / s.molecular_weight.to("kg/mol").v.item() for s in stm.species)
        w_ref = 1.0 / inv_w_ref
        numer, denom = 0.0, 0.0
        for s in stm.species:
            w_s = s.molecular_weight.to("kg/mol").v.item()
            ratio = w_ref / w_s
            species_coeffs = s.nasa_9_coefficients.get_coefficients_at_temperature(temp_k)
            cp_s_over_r_s = compute_cp_over_r(species_coeffs, temp_k)
            numer += s.mass_fraction * cp_s_over_r_s * ratio
            denom += s.mass_fraction * ratio
        return numer / denom

    @property
    def gas_constant(self) -> SpecificHeatCapacity.PositiveFloat64:
        """Mixture gas constant.

        For ``species_transport_model``: R_mix = R_u * sum_s(Y_s / W_s) at the
        reference composition. For ``thermally_perfect_gas`` only: falls back to the
        standard-air value 287.0529 J/(kg*K) because FrozenSpecies does not carry
        molecular weight (carried over from the legacy Air-as-TPG path). Users with a
        non-air pre-mixed gas should prefer species_transport_model with explicit W_s.
        """
        if self.species_transport_model is not None:
            R_universal_J_per_molK = 8.314462618  # CODATA 2018 universal gas constant
            sum_y_over_w_mol_per_kg = sum(
                s.mass_fraction / s.molecular_weight.to("kg/mol").v.item() for s in self.species_transport_model.species
            )
            return R_universal_J_per_molK * sum_y_over_w_mol_per_kg * u.m**2 / u.s**2 / u.K
        return 287.0529 * u.m**2 / u.s**2 / u.K

    @pd.validate_call
    def get_pressure(
        self, density: Density.PositiveFloat64, temperature: AbsoluteTemperature.Float64
    ) -> Pressure.PositiveFloat64:
        """Ideal-gas pressure: p = rho * R * T."""
        temperature = temperature.to("K")
        return density * self.gas_constant * temperature

    @pd.validate_call
    def get_speed_of_sound(self, temperature: AbsoluteTemperature.Float64) -> Velocity.PositiveFloat64:
        """Speed of sound: sqrt(gamma * R * T)."""
        temperature = temperature.to("K")
        gamma = self.get_specific_heat_ratio(temperature)
        return sqrt(gamma * self.gas_constant * temperature)

    @pd.validate_call
    def get_dynamic_viscosity(self, temperature: AbsoluteTemperature.Float64) -> Viscosity.NonNegativeFloat64:
        """Dynamic viscosity at T.

        For ``species_transport_model``, returns Wilke mixture viscosity at the reference
        composition. For ``thermally_perfect_gas`` only, returns the material-level
        ``dynamic_viscosity`` (Sutherland or constant).
        """
        if temperature.units is u.degC or temperature.units is u.degF:
            temperature = temperature.to("K")
        if self.species_transport_model is not None:
            return self._compute_species_mixture_viscosity(temperature)
        if isinstance(self.dynamic_viscosity, Sutherland):
            return self.dynamic_viscosity.get_dynamic_viscosity(temperature)
        return self.dynamic_viscosity

    def _compute_species_mixture_viscosity(
        self, temperature: AbsoluteTemperature.Float64
    ) -> Viscosity.NonNegativeFloat64:
        """Wilke mixture viscosity from species_transport_model at the reference composition.

        Wilke's rule [Wilke 1950, J. Chem. Phys. 18(4)]:
            mu_mix = sum_s [ x_s * mu_s / sum_k (x_k * Phi_sk) ]
        with mole fractions x_s = (Y_s / W_s) / sum_k (Y_k / W_k) and
            Phi_sk = (1 + sqrt(mu_s/mu_k) * (W_k/W_s)^(1/4))^2 / sqrt(8 * (1 + W_s/W_k)).
        """
        stm = self.species_transport_model
        species = stm.species

        T = temperature.to("K")
        mw = [s.molecular_weight.to("kg/mol").v.item() for s in species]
        mu = [s.dynamic_viscosity.get_dynamic_viscosity(T).to("Pa*s").v.item() for s in species]
        y = [s.mass_fraction for s in species]

        moles_per_kg = [y_s / w_s for y_s, w_s in zip(y, mw, strict=True)]
        total = sum(moles_per_kg)
        x = [m / total for m in moles_per_kg]

        n = len(species)
        mu_mix = 0.0
        for i in range(n):
            denom = 0.0
            for j in range(n):
                phi_num = (1.0 + (mu[i] / mu[j]) ** 0.5 * (mw[j] / mw[i]) ** 0.25) ** 2
                phi_den = (8.0 * (1.0 + mw[i] / mw[j])) ** 0.5
                denom += x[j] * (phi_num / phi_den)
            mu_mix += x[i] * mu[i] / denom

        return mu_mix * u.Pa * u.s


class SolidMaterial(MaterialBase):
    """
    Represents the solid material properties for heat transfer volume.
    """

    type: Literal["solid"] = pd.Field("solid", frozen=True)
    name: str = pd.Field(frozen=True, description="Name of the solid material.")
    thermal_conductivity: ThermalConductivity.PositiveFloat64 = pd.Field(
        frozen=True, description="Thermal conductivity of the material."
    )
    density: Density.PositiveFloat64 | None = pd.Field(None, frozen=True, description="Density of the material.")
    specific_heat_capacity: SpecificHeatCapacity.PositiveFloat64 | None = pd.Field(
        None, frozen=True, description="Specific heat capacity of the material."
    )


aluminum = SolidMaterial(
    name="aluminum",
    thermal_conductivity=235 * u.kg / u.s**3 * u.m / u.K,
    density=2710 * u.kg / u.m**3,
    specific_heat_capacity=903 * u.m**2 / u.s**2 / u.K,
)


class Water(MaterialBase):
    """
    Water material used for :class:`LiquidOperatingCondition`
    """

    type: Literal["water"] = pd.Field("water", frozen=True)
    name: str = pd.Field(frozen=True, description="Custom name of the water with given property.")
    density: Density.PositiveFloat64 | None = pd.Field(
        1000 * u.kg / u.m**3, frozen=True, description="Density of the water."
    )
    dynamic_viscosity: Viscosity.NonNegativeFloat64 = pd.Field(
        0.001002 * u.kg / u.m / u.s, frozen=True, description="Dynamic viscosity of the water."
    )


SolidMaterialTypes = SolidMaterial
FluidMaterialTypes = Union[Air, Gas, Water]
