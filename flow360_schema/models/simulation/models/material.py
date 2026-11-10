"""Material classes for the simulation framework."""

from typing import Literal, Union

import pydantic as pd
import unyt as u
from numpy import linspace, log, sqrt

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


# Largest jump tolerated between adjacent NASA-9 ranges, in cp/R, h/(R T) and s/R.
# Set by how precisely coefficients are transcribed, not by the fits themselves.
NASA9_CONTINUITY_TOLERANCE = 1e-4


# Ideal-gas lower bound on cp/R (translational modes alone give cv = 3/2 R). The solver holds cp at
# this floor, so a fit below it would be clamped silently unless it is rejected here. Fits of
# monatomic species sit exactly on the bound and can dip under it by fit error, hence the tolerance.
MINIMUM_CP_OVER_R = 2.5
CP_OVER_R_FIT_TOLERANCE = 1e-3
# gamma = (cp/R) / (cp/R - 1) at the bound: 5/3, the monatomic value.
MAXIMUM_SPECIFIC_HEAT_RATIO = MINIMUM_CP_OVER_R / (MINIMUM_CP_OVER_R - 1.0)


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


def compute_h_over_rt(coeffs, temperature):
    """
    Compute h/(R T) from NASA 9-coefficient polynomial, including the a7 constant.

    h/(R T) = -a0*T^-2 + a1*ln(T)/T + a2 + a3*T/2 + a4*T^2/3 + a5*T^3/4 + a6*T^4/5 + a7/T
    """
    temp = temperature
    return (
        -coeffs[0] * temp ** (-2)
        + coeffs[1] * log(temp) / temp
        + coeffs[2]
        + coeffs[3] * temp / 2.0
        + coeffs[4] * temp**2 / 3.0
        + coeffs[5] * temp**3 / 4.0
        + coeffs[6] * temp**4 / 5.0
        + coeffs[7] / temp
    )


def compute_s_over_r(coeffs, temperature):
    """
    Compute s/R at standard pressure from NASA 9-coefficient polynomial, including a8.

    s/R = -a0*T^-2/2 - a1*T^-1 + a2*ln(T) + a3*T + a4*T^2/2 + a5*T^3/3 + a6*T^4/4 + a8
    """
    temp = temperature
    return (
        -coeffs[0] * temp ** (-2) / 2.0
        - coeffs[1] * temp ** (-1)
        + coeffs[2] * log(temp)
        + coeffs[3] * temp
        + coeffs[4] * temp**2 / 2.0
        + coeffs[5] * temp**3 / 3.0
        + coeffs[6] * temp**4 / 4.0
        + coeffs[8]
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

    @pd.model_validator(mode="after")
    def validate_cp_over_r_at_least_ideal_gas_minimum(self):
        """Validate cp/R >= 5/2 over the range, the ideal-gas minimum the solver floors cp at."""
        t_min_k = self.temperature_range_min.to("K").v.item()
        t_max_k = self.temperature_range_max.to("K").v.item()
        temperatures = linspace(t_min_k, t_max_k, 101)
        cp_over_r = compute_cp_over_r(self.coefficients, temperatures)
        i_min = int(cp_over_r.argmin())
        if cp_over_r[i_min] < MINIMUM_CP_OVER_R - CP_OVER_R_FIT_TOLERANCE:
            raise ValueError(
                f"cp/R must be at least 5/2 over the temperature range, got {cp_over_r[i_min]:.4g} "
                f"at {temperatures[i_min]:.4g} K; an ideal gas has cp/R >= 5/2 (specific heat ratio "
                "gamma <= 5/3), and the solver holds cp at that floor."
            )
        return self


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

    @pd.model_validator(mode="after")
    def validate_thermo_continuity(self):
        """Validate that adjacent NASA-9 ranges agree on cp, h and s where they meet.

        Consecutive ranges are separate fits of one substance, so the polynomials must
        agree at the boundary they share. Nothing in the fit enforces that: cp comes out
        continuous on its own, but h and s also depend on the a7 and a8 integration
        constants, which are set per range and so can disagree. A jump in h drives the
        solver's nodal caloric surrogate to a negative heat capacity and aborts the run;
        a jump in s is quieter but reaches stagnation pressure as exp(ds/R).

        The tolerance is set by transcription, not by the fits: the curated library
        agrees to 5e-7 in cp/R and s/R and to 5e-5 in h/(R T), so 1e-4 passes every
        entry while catching discontinuities orders of magnitude smaller than the ones
        that break the solver.
        """
        for i in range(len(self.temperature_ranges) - 1):
            low = self.temperature_ranges[i]
            high = self.temperature_ranges[i + 1]
            t_b = low.temperature_range_max.to("K").v.item()
            for label, function, culprit in (
                ("cp/R", compute_cp_over_r, "the cp/R coefficients a0-a6"),
                ("h/(R T)", compute_h_over_rt, "the a7 enthalpy constant"),
                ("s/R", compute_s_over_r, "the a8 entropy constant"),
            ):
                jump = float(function(high.coefficients, t_b)) - float(function(low.coefficients, t_b))
                if abs(jump) > NASA9_CONTINUITY_TOLERANCE:
                    raise ValueError(
                        f"NASA-9 ranges {i} and {i+1} disagree on {label} at their shared "
                        f"boundary {t_b:.4g} K by {jump:.4g}, which exceeds the tolerance of "
                        f"{NASA9_CONTINUITY_TOLERANCE:g}. Consecutive ranges are fits of the "
                        "same substance and must agree where they meet; a discontinuity here "
                        f"makes the solver's thermodynamics inconsistent. Check {culprit} of "
                        "both ranges, and transcribe the coefficients at full precision."
                    )
        return self

    @classmethod
    @pd.validate_call
    def constant_gamma(cls, gamma: pd.PositiveFloat):
        """Coefficients of a calorically perfect gas with specific heat ratio ``gamma`` in (1, 5/3]."""
        if gamma <= 1.0:
            raise ValueError(f"The specific heat ratio gamma must be greater than 1, got {gamma}.")
        if gamma > MAXIMUM_SPECIFIC_HEAT_RATIO:
            raise ValueError(
                f"The specific heat ratio gamma must be at most 5/3 (cp/R >= 5/2 for an ideal gas), got {gamma}."
            )
        return cls(
            temperature_ranges=[
                NASA9CoefficientSet(
                    temperature_range_min=200 * u.K,
                    temperature_range_max=6000 * u.K,
                    coefficients=[0.0, 0.0, gamma / (gamma - 1.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                )
            ]
        )

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


def compute_mixture_molar_mass(species) -> float:
    """Molar mass of the reference mixture, ``W_ref = 1 / sum_s(Y_s / W_s)``, in kg/mol.

    Single source for the mixture molar mass used by the client's thermodynamic blends
    and by the translator (frozen blend weights ``Y_s W_ref / W_s`` and the transported
    ``inverseMolecularWeightRatioToReference``).
    """
    return 1.0 / sum(s.mass_fraction / s.molecular_weight.to("kg/mol").v.item() for s in species)


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


class Species(Flow360BaseModel):
    """
    A chemical species of a gas mixture.

    The same ``Species`` serves both species treatments on :class:`Fluid`: with a
    :class:`FrozenSpeciesSolver` the composition is fixed when the simulation parameters
    are built, while a :class:`SpeciesTransportSolver` transports each species as a
    mass-fraction field. The full set of thermodynamic and transport properties needed for
    the multi-species path is therefore required at construction time:

    - NASA-9 polynomial for cp(T), h(T), s(T) (frozen-composition / non-reacting
      regime; see the data-source citation in any library entry for the validity
      window).
    - Molecular weight, to form the mixture gas constant
      ``R_mix = R_u * sum_s(Y_s / W_s)``.
    - Per-species Sutherland viscosity model, blended into the mixture viscosity
      via Wilke's rule.
    - Laminar Schmidt number (defaults to ``DEFAULT_SCHMIDT_NUMBER``), used by the
      constant-Schmidt Fickian diffusion closure.

    The ``mass_fraction`` value defines this species' contribution to the *reference*
    mixture composition. With a :class:`FrozenSpeciesSolver` that composition is the
    one the solver runs with; with a :class:`SpeciesTransportSolver` the translator emits
    it as ``referenceStateNondim`` and the actual mass fraction varies in space and time.
    """

    type_name: Literal["Species"] = pd.Field("Species", frozen=True)
    name: str = pd.Field(description="Species name (e.g., 'N2', 'O2', 'H2O')")
    nasa_9_coefficients: NASA9Coefficients = pd.Field(
        description="NASA 9-coefficient polynomial for this species' thermodynamic data."
    )
    mass_fraction: pd.confloat(ge=0, le=1) = pd.Field(
        description="Reference mass fraction of this species in the mixture, within [0, 1]; "
        "zero is legal (e.g. a species only injected through a boundary). Mass fractions "
        "across all species of a Gas must sum to 1.0. Boundary and initial-condition "
        "compositions are set separately, on the boundary models and the Fluid "
        "initial condition's species_mass_fractions."
    )

    def preprocess(self, **kwargs):
        # molecular_weight is deliberately unit-agnostic (the translator
        # collapses it to the dimensionless W_ref/W_s); mol is not a base
        # dimension of the solver unit system, so exclude it from conversion.
        exclude = list(kwargs.pop("exclude", None) or [])
        exclude.append("molecular_weight")
        return super().preprocess(exclude=exclude, **kwargs)

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


def _validate_unique_species_names(species: "list[Species]") -> None:
    """Validate that all species have unique names."""
    names = [s.name for s in species]
    if len(names) != len(set(names)):
        duplicates = [name for name in names if names.count(name) > 1]
        raise ValueError(f"Species names must be unique. Duplicates found: {set(duplicates)}")


def _validate_temperature_ranges_match(species: "list[Species]") -> None:
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


class Air(MaterialBase):
    """
    Calorically-perfect-gas (CPG) preset for air: constant ``gamma = 1.4``, constant
    ``R = 287.0529 J/(kg*K)``, and a standard Sutherland viscosity model.

    For temperature-dependent thermodynamics (NASA-9 polynomials) or variable-
    composition multi-species transport, use :class:`Gas` instead.
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
    Customizable gas material defined by its species. The material describes the
    molecules only; how the species are solved is chosen on :class:`Fluid`:
    :class:`FrozenSpeciesSolver` fixes the composition (no transport equations are
    solved), :class:`SpeciesTransportSolver` transports the species' mass fractions.
    For the constant-gamma standard-air case use the :class:`Air` preset instead.

    Common species are available from a curated library, so the per-species data
    need not be typed out by hand:

    >>> n2 = fl.Species.from_library("N2")
    >>> n2.mass_fraction = 0.767
    >>> o2 = fl.Species.from_library("O2")
    >>> o2.mass_fraction = 0.233
    >>> fl.Gas(name="air", species=[n2, o2])

    The library covers ``N2, O2, Ar, He, H2, CO, CO2, CH4, H2O``; see
    :meth:`Species.from_library` for the per-field overrides, and construct
    :class:`Species` directly for anything outside the catalogue.

    With a :class:`FrozenSpeciesSolver` and more than one species the solver receives a
    single Sutherland law: its reference viscosity is Wilke's mixture value at the
    freestream reference temperature (exact there) and its constant is fitted to Wilke's
    rule over the species' NASA-9 temperature range. The fit is within 0.02% for N2/O2 air
    and within about 2% for any pair of library species, but a single Sutherland law
    cannot follow Wilke's rule exactly when the species' constants differ widely
    (e.g. H2O against N2). Set ``dynamic_viscosity`` to use your own mixture law or
    constant instead of the fit.
    """

    type: Literal["gas"] = pd.Field("gas", frozen=True)
    name: str = pd.Field(description="Name identifying this gas (e.g. 'methane-air', 'N2/He mix').")
    prandtl_number: pd.PositiveFloat = pd.Field(
        0.72,
        description="Laminar Prandtl number. Default is 0.72.",
    )
    turbulent_prandtl_number: pd.PositiveFloat = pd.Field(
        0.9,
        description="Turbulent Prandtl number. Default is 0.9.",
    )
    species: list[Species] = pd.Field(
        min_length=1,
        description="Species making up the gas, with their NASA-9 coefficients, molecular "
        "weights, Sutherland laws, Schmidt numbers and reference mass fractions. Reference mass "
        "fractions must sum to 1.0. Viscosity is Wilke-mixed from the per-species Sutherland laws.",
    )
    dynamic_viscosity: Sutherland | Viscosity.NonNegativeFloat64 | None = pd.Field(
        None,
        description="Optional mixture viscosity model or constant value used instead of the "
        "Sutherland law fitted to Wilke's rule from the per-species laws. Applies to both the "
        "freestream ``muRef`` and the runtime viscosity. Only valid with a "
        "``FrozenSpeciesSolver``; transported species form viscosity from the local composition.",
    )

    @pd.model_validator(mode="before")
    @classmethod
    def _reject_species_transport_model(cls, values):
        if isinstance(values, dict) and "species_transport_model" in values:
            raise ValueError(
                "`Gas.species_transport_model` was removed: list the species directly with "
                "`Gas(species=[...])` and choose the treatment on `Fluid`: "
                "`species_transport_solver=FrozenSpeciesSolver()` for a fixed composition, or "
                "`SpeciesTransportSolver(turbulent_schmidt_number=...)` to transport it. The "
                "initial composition goes in `FluidInitialCondition(species_mass_fractions={...})`."
            )
        return values

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

    def get_specific_heat_ratio(self, temperature: AbsoluteTemperature.Float64) -> pd.PositiveFloat:
        """Specific heat ratio gamma at T: the mass+mole-weighted mixture value at the
        reference composition."""
        temp_k = temperature.to("K").v.item()
        cp_over_r_mix = self._compute_species_cp_over_r_mix(temp_k)
        return cp_over_r_mix / (cp_over_r_mix - 1.0)

    def _compute_species_cp_over_r_mix(self, temp_k: float) -> float:
        """cp_mix / R_mix at ``temp_k`` from the species reference composition.

        Per the standard mixture relations:
            cp_mix / R_mix = sum_s Y_s (cp_s / R_s) (W_ref / W_s) / sum_s Y_s (W_ref / W_s)
        where W_ref = 1 / sum_s (Y_s / W_s) is the mixture molar mass.
        """
        w_ref = compute_mixture_molar_mass(self.species)
        numer, denom = 0.0, 0.0
        for s in self.species:
            w_s = s.molecular_weight.to("kg/mol").v.item()
            ratio = w_ref / w_s
            species_coeffs = s.nasa_9_coefficients.get_coefficients_at_temperature(temp_k)
            cp_s_over_r_s = compute_cp_over_r(species_coeffs, temp_k)
            numer += s.mass_fraction * cp_s_over_r_s * ratio
            denom += s.mass_fraction * ratio
        return numer / denom

    @property
    def gas_constant(self) -> SpecificHeatCapacity.PositiveFloat64:
        """Mixture gas constant R_mix = R_u * sum_s(Y_s / W_s) at the reference composition."""
        R_universal_J_per_molK = 8.314462618  # CODATA 2018 universal gas constant
        w_ref = compute_mixture_molar_mass(self.species)
        return R_universal_J_per_molK / w_ref * u.m**2 / u.s**2 / u.K

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
        """Dynamic viscosity at T: Wilke mixture viscosity at the reference composition,
        or the ``dynamic_viscosity`` override when one is set."""
        if temperature.units is u.degC or temperature.units is u.degF:
            temperature = temperature.to("K")
        if self.dynamic_viscosity is not None:
            if isinstance(self.dynamic_viscosity, Sutherland):
                return self.dynamic_viscosity.get_dynamic_viscosity(temperature)
            return self.dynamic_viscosity
        return self._compute_species_mixture_viscosity(temperature)

    def _compute_species_mixture_viscosity(
        self, temperature: AbsoluteTemperature.Float64
    ) -> Viscosity.NonNegativeFloat64:
        """Wilke mixture viscosity of the species at the reference composition.

        Wilke's rule [Wilke 1950, J. Chem. Phys. 18(4)]:
            mu_mix = sum_s [ x_s * mu_s / sum_k (x_k * Phi_sk) ]
        with mole fractions x_s = (Y_s / W_s) / sum_k (Y_k / W_k) and
            Phi_sk = (1 + sqrt(mu_s/mu_k) * (W_k/W_s)^(1/4))^2 / sqrt(8 * (1 + W_s/W_k)).
        """
        species = self.species

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
