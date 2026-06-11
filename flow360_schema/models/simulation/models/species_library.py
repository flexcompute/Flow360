"""Curated library of common gas species for variable-composition transport.

This module ships a small, hand-vetted catalogue of pure-species thermodynamic and
transport data so users can write

    fl.Species.from_library("N2", mass_fraction=0.767)

instead of typing out 18 NASA-9 coefficients, a molecular weight, and a Sutherland
triple for every species in a multi-species ``SpeciesTransportModel``. The library
backs :meth:`flow360_schema.models.simulation.models.material.Species.from_library`;
users still have the override path (any kwarg passed to ``from_library`` overrides
the corresponding library default) and the fully-custom path
(``Species(...)`` directly) for species not in the catalogue.

Roster:
    N2, O2, Ar, He, H2, CO, CO2, CH4, H2O.

NO is intentionally absent: its primary use cases are chemistry-driven
(NOx emissions, hypersonic air chemistry, atmospheric NO/NO2 cycling), none of
which the non-reacting transport path models. Users who need NO can construct it
directly via :class:`Species` with their preferred data source.

Data sources
============

**NASA-9 thermodynamic polynomials** (``nasa9_low`` and ``nasa9_high``):

    McBride, B. J., Zehe, M. J., Gordon, S., "NASA Glenn Coefficients for
    Calculating Thermodynamic Properties of Individual Species,"
    NASA/TP--2002-211556, September 2002.
    Mirror used for transcription:
    https://shepherd.caltech.edu/EDL/PublicResources/sdt/SDToolbox/data/NASA9/nasa9.dat
    (Shepherd group, Caltech Explosion Dynamics Lab; verified against the
    publication's primary tables for N2, O2, Ar; NO transcription also verified
    but the species is intentionally deferred -- see the v1-roster note above).

    All entries use the standard two-range NASA-9 partition with the same
    breakpoints: ``LIBRARY_TEMPERATURE_RANGES`` below. The native NASA Glenn
    database includes a third 6000-20000 K range for many species; the library
    intentionally truncates to two ranges to match the typical aerospace CFD
    operating envelope and to keep mixtures with the existing temperature-range
    matching validator trivially satisfied.

    For N2, O2, Ar the library uses the in-repo coefficients from
    ``localTests/shared_utils/exact_riemann_solver.py`` (transcribed from
    NASA/TP-2002-211556 at slightly lower precision) for byte-equality with the
    Riemann-solver validation tests. The Caltech mirror values are bit-equal in
    the leading 7-8 digits.

**Molecular weights** (``molecular_weight``):
    IUPAC Commission on Isotopic Abundances and Atomic Weights (CIAAW) standard
    atomic weights, as embedded in the NASA Glenn database above. Cross-checked
    against any chemistry handbook.

**Sutherland viscosity constants** (``sutherland_mu_ref``, ``sutherland_S``):
    Two-parameter Sutherland fits to NIST WebBook viscosity data
    (https://webbook.nist.gov/). The underlying NIST reference equations cite:

        Lemmon, E. W. and Jacobsen, R. T., "Viscosity and Thermal Conductivity
        Equations for Nitrogen, Oxygen, Argon, and Air," International Journal
        of Thermophysics, Vol. 25, No. 1, 2004, pp. 21-69.

    for N2/O2/Ar/Air, and species-specific NIST reference equations for the
    others (e.g., IAPWS-95 for H2O).

    Each fit covers a documented temperature window noted in the entry's
    ``sutherland_fit_window`` field. The fits were validated against the
    canonical published values

        F. M. White, "Viscous Fluid Flow," 2nd ed., McGraw-Hill, 1991, Table 1-2.
        (As reproduced in Kim, Han, Kim, "Numerical Analysis of Flow
        Characteristics of An Atmospheric Plasma Torch," arXiv:physics/0410237.)

    The fitted S values may differ from those published values by 5-50% --
    Sutherland's two-parameter form is sensitive to the fitting window. The
    fits in this library are typically more accurate against NIST data over
    the v1 NASA-9 range (200-6000 K) than the canonical textbook values, which
    are usually fit over a narrower window near T_ref. See ``fit_sutherland.py``
    (development tool, not shipped) for the fitting procedure.

    The ``H2O`` entry carries a known limitation: water's polar molecular
    structure makes the two-parameter Sutherland fit a poor approximation in
    principle, even though the numerical fit accuracy within the validated
    window is excellent. Users running water-vapor-dominated flows should
    override ``dynamic_viscosity`` with an application-specific viscosity model.

**Frozen-composition assumption.** The NASA-9 polynomials provided here are
valid in the *non-reacting / frozen-composition* regime: each species is
treated as a stable molecule across its tabulated range. High-temperature
dissociation (H2O above ~2000 K, CO2 above ~2500 K, CH4 thermal cracking
above ~1500 K, diatomics 3000-4000 K) requires a reacting-chemistry solver
and is out of scope for this library.
"""

import unyt as u

from flow360_schema.models.simulation.models.material import (
    NASA9Coefficients,
    NASA9CoefficientSet,
    Species,
    Sutherland,
)

# All library entries share the same two-range NASA-9 partition; storing this
# once (rather than per species) enforces the invariant that mixtures of
# library species automatically pass the SpeciesTransportModel temperature-range
# matching validator.
LIBRARY_TEMPERATURE_RANGES = (
    (200.0 * u.K, 1000.0 * u.K),
    (1000.0 * u.K, 6000.0 * u.K),
)

# Reference temperature for all Sutherland fits in this library.
SUTHERLAND_REFERENCE_TEMPERATURE = 273.15 * u.K


def _nasa9(low: tuple, high: tuple) -> NASA9Coefficients:
    """Build a NASA-9 polynomial spanning the library's standard two ranges."""
    return NASA9Coefficients(
        temperature_ranges=[
            NASA9CoefficientSet(
                temperature_range_min=LIBRARY_TEMPERATURE_RANGES[0][0],
                temperature_range_max=LIBRARY_TEMPERATURE_RANGES[0][1],
                coefficients=list(low),
            ),
            NASA9CoefficientSet(
                temperature_range_min=LIBRARY_TEMPERATURE_RANGES[1][0],
                temperature_range_max=LIBRARY_TEMPERATURE_RANGES[1][1],
                coefficients=list(high),
            ),
        ]
    )


def _sutherland(mu_ref, S) -> Sutherland:
    """Build a Sutherland model at the library's standard reference temperature."""
    return Sutherland(
        reference_viscosity=mu_ref,
        reference_temperature=SUTHERLAND_REFERENCE_TEMPERATURE,
        effective_temperature=S,
    )


_REGISTRY: dict[str, Species] = {
    # -----------------------------------------------------------------------
    # Air components
    # -----------------------------------------------------------------------
    # N2: air component (~78% by mass). NASA-9 coefficients bit-equal to the
    # in-repo Riemann-solver data (localTests/shared_utils/exact_riemann_solver.py).
    # Sutherland: NIST fit, 200-1000 K window; max err 5.2% over 200-2000 K.
    # Source: NASA/TP-2002-211556; Lemmon & Jacobsen, Int. J. Thermophys. 25(1), 2004.
    "N2": Species(
        name="N2",
        nasa_9_coefficients=_nasa9(
            low=(
                2.21037150e04,
                -3.81846182e02,
                6.08273836,
                -8.53091441e-03,
                1.38464610e-05,
                -9.62579362e-09,
                2.51970561e-12,
                -1.04396091e03,
                -1.04765254e01,
            ),
            high=(
                5.87712406e05,
                -2.23924969e03,
                6.06694922,
                -6.13968556e-04,
                1.49180673e-07,
                -1.92309843e-11,
                1.06194817e-15,
                1.28320618e04,
                -1.58637463e01,
            ),
        ),
        mass_fraction=1.0,
        molecular_weight=28.0134 * u.g / u.mol,
        dynamic_viscosity=_sutherland(1.6537e-05 * u.Pa * u.s, 129.99 * u.K),
    ),
    # O2: air component (~21% by mass). NASA-9 bit-equal to Riemann-solver data.
    # Sutherland: NIST fit, 200-1000 K window; max err 4.2% over 200-2000 K.
    "O2": Species(
        name="O2",
        nasa_9_coefficients=_nasa9(
            low=(
                -3.42556342e04,
                4.84700097e02,
                1.119010961,
                4.29388924e-03,
                -6.83630052e-07,
                -2.0233727e-09,
                1.039040018e-12,
                -3.39145487e03,
                1.84969947e01,
            ),
            high=(
                -1.037939022e06,
                2.344830282e03,
                1.819732036,
                1.267847582e-03,
                -2.188067988e-07,
                2.053719572e-11,
                -8.19346705e-16,
                -1.689010929e04,
                1.738716506e01,
            ),
        ),
        mass_fraction=1.0,
        molecular_weight=31.9988 * u.g / u.mol,
        dynamic_viscosity=_sutherland(1.9042e-05 * u.Pa * u.s, 147.49 * u.K),
    ),
    # Ar: air component (~1% by mass). Monatomic noble gas; gamma = 5/3, cp/R = 2.5.
    # NASA-9 bit-equal to Riemann-solver data. Sutherland NIST fit, 200-1000 K;
    # max err 3.6% over 200-2000 K.
    "Ar": Species(
        name="Ar",
        nasa_9_coefficients=_nasa9(
            low=(0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -7.45375e02, 4.37967491),
            high=(
                2.010538475e01,
                -5.99266107e-02,
                2.500069401,
                -3.99214116e-08,
                1.20527214e-11,
                -1.819015576e-15,
                1.078576636e-19,
                -7.449939610e02,
                4.379180110,
            ),
        ),
        mass_fraction=1.0,
        molecular_weight=39.948 * u.g / u.mol,
        dynamic_viscosity=_sutherland(2.0903e-05 * u.Pa * u.s, 170.02 * u.K),
    ),
    # -----------------------------------------------------------------------
    # Combustion products
    # -----------------------------------------------------------------------
    # H2O: water vapor. Sutherland's two-parameter form is a poor approximation
    # for polar molecules like H2O; the large fitted S (~1150 K) reflects this.
    # NIST fit, 400-1000 K window; max err 0.64% within window but extrapolates
    # poorly. For water-vapor-dominated flows, override `dynamic_viscosity` with
    # an application-specific model. Dissociation matters above ~2000 K; the
    # NASA-9 polynomial assumes frozen composition above that.
    "H2O": Species(
        name="H2O",
        nasa_9_coefficients=_nasa9(
            low=(
                -3.94796083e04,
                5.75573102e02,
                9.31782653e-01,
                7.22271286e-03,
                -7.34255737e-06,
                4.95504349e-09,
                -1.336933246e-12,
                -3.30397431e04,
                1.724205775e01,
            ),
            high=(
                1.034972096e06,
                -2.412698562e03,
                4.64611078,
                2.291998307e-03,
                -6.83683048e-07,
                9.42646893e-11,
                -4.82238053e-15,
                -1.384286509e04,
                -7.97814851,
            ),
        ),
        mass_fraction=1.0,
        molecular_weight=18.01528 * u.g / u.mol,
        dynamic_viscosity=_sutherland(8.1078e-06 * u.Pa * u.s, 1150.55 * u.K),
    ),
    # CO2: combustion product. Dissociates to CO + 1/2 O2 above ~2500 K; frozen
    # composition assumed. NIST Sutherland fit, 220-1000 K; max err 1.20%.
    "CO2": Species(
        name="CO2",
        nasa_9_coefficients=_nasa9(
            low=(
                4.94378364e04,
                -6.26429208e02,
                5.30181336,
                2.503600571e-03,
                -2.124700099e-07,
                -7.6914868e-10,
                2.849979913e-13,
                -4.52818986e04,
                -7.04879010,
            ),
            high=(
                1.176969434e05,
                -1.788801467e03,
                8.29154353,
                -9.22477831e-05,
                4.86963541e-09,
                -1.892063841e-12,
                6.33067509e-16,
                -3.908345010e04,
                -2.652683962e01,
            ),
        ),
        mass_fraction=1.0,
        molecular_weight=44.0095 * u.g / u.mol,
        dynamic_viscosity=_sutherland(1.3701e-05 * u.Pa * u.s, 272.24 * u.K),
    ),
    # CO: combustion intermediate. Sutherland NIST fit, 200-500 K window
    # (limited by NIST data coverage at 1 atm); max err 0.19% within window;
    # extrapolates with degraded accuracy at higher temperatures.
    "CO": Species(
        name="CO",
        nasa_9_coefficients=_nasa9(
            low=(
                1.489027557e04,
                -2.922250947e02,
                5.72445841,
                -8.17613694e-03,
                1.456885983e-05,
                -1.087733246e-08,
                3.027905485e-12,
                -1.303069697e04,
                -7.85917928,
            ),
            high=(
                4.61915856e05,
                -1.944685748e03,
                5.91664709,
                -5.66423407e-04,
                1.398802571e-07,
                -1.787664983e-11,
                9.6208504e-16,
                -2.465738441e03,
                -1.387402604e01,
            ),
        ),
        mass_fraction=1.0,
        molecular_weight=28.0101 * u.g / u.mol,
        dynamic_viscosity=_sutherland(1.6610e-05 * u.Pa * u.s, 113.49 * u.K),
    ),
    # -----------------------------------------------------------------------
    # Fuels
    # -----------------------------------------------------------------------
    # H2: hydrogen fuel. Dissociates above ~3000 K; frozen composition assumed.
    # NIST Sutherland fit, 200-1000 K; max err 6.3% within window.
    "H2": Species(
        name="H2",
        nasa_9_coefficients=_nasa9(
            low=(
                4.07832281e04,
                -8.00918545e02,
                8.21470167,
                -1.269714360e-02,
                1.753604930e-05,
                -1.202860160e-08,
                3.368093160e-12,
                2.682484380e03,
                -3.043788660e01,
            ),
            high=(
                5.608123380e05,
                -8.371491340e02,
                2.975363040,
                1.252249930e-03,
                -3.740718420e-07,
                5.936628250e-11,
                -3.606995730e-15,
                5.339815850e03,
                -2.202764050,
            ),
        ),
        mass_fraction=1.0,
        molecular_weight=2.01588 * u.g / u.mol,
        dynamic_viscosity=_sutherland(8.2348e-06 * u.Pa * u.s, 124.61 * u.K),
    ),
    # CH4: methane fuel. Thermally cracks (pyrolysis) above ~1000-1500 K; the
    # NASA-9 polynomial assumes frozen composition above that. Sutherland NIST
    # fit, 200-600 K (limited by NIST data coverage at 1 atm + cracking above
    # ~1000 K); max err 0.38% within window.
    "CH4": Species(
        name="CH4",
        nasa_9_coefficients=_nasa9(
            low=(
                -1.766545730e05,
                2.785477820e03,
                -1.20193547e01,
                3.914625880e-02,
                -3.611656080e-05,
                2.018387940e-08,
                -4.955772150e-12,
                -2.331011560e04,
                8.901075390e01,
            ),
            high=(
                3.746265700e06,
                -1.388851340e04,
                2.054029820e01,
                -1.944196930e-03,
                4.323871450e-07,
                -4.061012780e-11,
                1.643159270e-15,
                7.565988680e04,
                -1.222977672e02,
            ),
        ),
        mass_fraction=1.0,
        molecular_weight=16.04246 * u.g / u.mol,
        dynamic_viscosity=_sutherland(1.0257e-05 * u.Pa * u.s, 178.53 * u.K),
    ),
    # -----------------------------------------------------------------------
    # Inert tracer
    # -----------------------------------------------------------------------
    # He: inert noble gas; gamma = 5/3, cp/R = 2.5 across both ranges. Commonly
    # used as a tracer or lifting gas. NIST Sutherland fit, 200-1000 K;
    # max err 6.7% over 200-1500 K.
    "He": Species(
        name="He",
        nasa_9_coefficients=_nasa9(
            low=(0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -7.45375e02, 9.287239740e-01),
            high=(0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, -7.45375e02, 9.287239740e-01),
        ),
        mass_fraction=1.0,
        molecular_weight=4.002602 * u.g / u.mol,
        dynamic_viscosity=_sutherland(1.8382e-05 * u.Pa * u.s, 122.81 * u.K),
    ),
}


def _build_species(name: str) -> Species:
    """Look up ``name`` in the library and return an independent deep copy.

    The returned Species carries the library's NASA-9 polynomial, molecular weight,
    and Sutherland viscosity, with ``mass_fraction`` placeholder-set to 1.0.
    Override any field by assigning to the returned object's attribute -- pydantic
    ``validate_assignment`` runs the field validators on every assignment.

    Imported lazily by :meth:`Species.from_library` to keep that classmethod's
    body free of import-time work.
    """
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown species {name!r}. Available species in the library: {available}. "
            "For species not in the library, construct Species(...) directly."
        )
    return _REGISTRY[name].model_copy(deep=True)
