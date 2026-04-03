"""Tests for material model validators (NASA9 coefficients, ThermallyPerfectGas)."""

import pytest

import flow360 as fl
from flow360.component.simulation.models.material import (
    FrozenSpecies,
    Gas,
    NASA9Coefficients,
    NASA9CoefficientSet,
    Sutherland,
    ThermallyPerfectGas,
)
from flow360.component.simulation.unit_system import SI_unit_system

# =============================================================================
# NASA9CoefficientSet Tests
# =============================================================================


def test_nasa9_coefficient_set_valid():
    """Test creating a valid NASA9CoefficientSet with exactly 9 coefficients."""
    with SI_unit_system:
        coeff_set = NASA9CoefficientSet(
            temperature_range_min=200.0 * fl.u.K,
            temperature_range_max=1000.0 * fl.u.K,
            coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
    assert len(coeff_set.coefficients) == 9


def test_nasa9_coefficient_set_wrong_count_raises():
    """Test that providing wrong number of coefficients raises ValueError."""
    with SI_unit_system:
        with pytest.raises(ValueError, match="requires exactly 9 coefficients"):
            NASA9CoefficientSet(
                temperature_range_min=200.0 * fl.u.K,
                temperature_range_max=1000.0 * fl.u.K,
                coefficients=[0.0, 0.0, 3.5],  # Only 3 coefficients
            )


def test_nasa9_coefficient_set_too_many_coefficients_raises():
    """Test that providing too many coefficients raises ValueError."""
    with SI_unit_system:
        with pytest.raises(ValueError, match="requires exactly 9 coefficients"):
            NASA9CoefficientSet(
                temperature_range_min=200.0 * fl.u.K,
                temperature_range_max=1000.0 * fl.u.K,
                coefficients=[0.0] * 10,  # 10 coefficients
            )


# =============================================================================
# NASA9Coefficients Tests
# =============================================================================


def test_nasa9_coefficients_single_range_valid():
    """Test creating NASA9Coefficients with a single temperature range."""
    with SI_unit_system:
        coeffs = NASA9Coefficients(
            temperature_ranges=[
                NASA9CoefficientSet(
                    temperature_range_min=200.0 * fl.u.K,
                    temperature_range_max=6000.0 * fl.u.K,
                    coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                )
            ]
        )
    assert len(coeffs.temperature_ranges) == 1


def test_nasa9_coefficients_multiple_ranges_continuous_valid():
    """Test creating NASA9Coefficients with continuous temperature ranges."""
    with SI_unit_system:
        coeffs = NASA9Coefficients(
            temperature_ranges=[
                NASA9CoefficientSet(
                    temperature_range_min=200.0 * fl.u.K,
                    temperature_range_max=1000.0 * fl.u.K,
                    coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
                NASA9CoefficientSet(
                    temperature_range_min=1000.0 * fl.u.K,
                    temperature_range_max=6000.0 * fl.u.K,
                    coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ),
            ]
        )
    assert len(coeffs.temperature_ranges) == 2


def test_nasa9_coefficients_discontinuous_ranges_raises():
    """Test that discontinuous temperature ranges raise ValueError."""
    with SI_unit_system:
        with pytest.raises(ValueError, match="Temperature ranges must be continuous"):
            NASA9Coefficients(
                temperature_ranges=[
                    NASA9CoefficientSet(
                        temperature_range_min=200.0 * fl.u.K,
                        temperature_range_max=1000.0 * fl.u.K,
                        coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                    NASA9CoefficientSet(
                        temperature_range_min=1500.0 * fl.u.K,  # Gap: 1000-1500 K
                        temperature_range_max=6000.0 * fl.u.K,
                        coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                ]
            )


# =============================================================================
# ThermallyPerfectGas Tests
# =============================================================================


def _make_species(name: str, mass_fraction: float, t_min: float = 200.0, t_max: float = 6000.0):
    """Helper to create a FrozenSpecies with given mass fraction.

    Note: Must be called within a SI_unit_system context.
    """
    return FrozenSpecies(
        name=name,
        nasa_9_coefficients=NASA9Coefficients(
            temperature_ranges=[
                NASA9CoefficientSet(
                    temperature_range_min=t_min * fl.u.K,
                    temperature_range_max=t_max * fl.u.K,
                    coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                )
            ]
        ),
        mass_fraction=mass_fraction,
    )


def test_thermally_perfect_gas_single_species_valid():
    """Test creating ThermallyPerfectGas with a single species (mass_fraction=1.0)."""
    with SI_unit_system:
        tpg = ThermallyPerfectGas(species=[_make_species("N2", 1.0)])
    assert len(tpg.species) == 1
    assert tpg.species[0].mass_fraction == 1.0


def test_thermally_perfect_gas_multi_species_valid():
    """Test creating ThermallyPerfectGas with multiple species summing to 1.0."""
    with SI_unit_system:
        tpg = ThermallyPerfectGas(
            species=[
                _make_species("N2", 0.7555),
                _make_species("O2", 0.2316),
                _make_species("Ar", 0.0129),
            ]
        )
    assert len(tpg.species) == 3
    total = sum(s.mass_fraction for s in tpg.species)
    assert 0.999 <= total <= 1.001


def test_thermally_perfect_gas_mass_fractions_not_sum_to_one_raises():
    """Test that mass fractions not summing to 1.0 raise ValueError."""
    with SI_unit_system:
        with pytest.raises(ValueError, match="Mass fractions must sum to 1.0"):
            ThermallyPerfectGas(
                species=[
                    _make_species("N2", 0.5),
                    _make_species("O2", 0.3),
                    # Missing 0.2 to sum to 1.0
                ]
            )


def test_thermally_perfect_gas_mass_fractions_exceed_one_raises():
    """Test that mass fractions exceeding 1.0 raise ValueError."""
    with SI_unit_system:
        with pytest.raises(ValueError, match="Mass fractions must sum to 1.0"):
            ThermallyPerfectGas(
                species=[
                    _make_species("N2", 0.8),
                    _make_species("O2", 0.4),  # Sum = 1.2
                ]
            )


def test_thermally_perfect_gas_temperature_ranges_match_valid():
    """Test that species with matching temperature ranges are accepted."""
    with SI_unit_system:
        # Both species have same temperature range: 200-1000K, 1000-6000K
        n2 = FrozenSpecies(
            name="N2",
            nasa_9_coefficients=NASA9Coefficients(
                temperature_ranges=[
                    NASA9CoefficientSet(
                        temperature_range_min=200.0 * fl.u.K,
                        temperature_range_max=1000.0 * fl.u.K,
                        coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                    NASA9CoefficientSet(
                        temperature_range_min=1000.0 * fl.u.K,
                        temperature_range_max=6000.0 * fl.u.K,
                        coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                ]
            ),
            mass_fraction=0.75,
        )
        o2 = FrozenSpecies(
            name="O2",
            nasa_9_coefficients=NASA9Coefficients(
                temperature_ranges=[
                    NASA9CoefficientSet(
                        temperature_range_min=200.0 * fl.u.K,
                        temperature_range_max=1000.0 * fl.u.K,
                        coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                    NASA9CoefficientSet(
                        temperature_range_min=1000.0 * fl.u.K,
                        temperature_range_max=6000.0 * fl.u.K,
                        coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                ]
            ),
            mass_fraction=0.25,
        )
        tpg = ThermallyPerfectGas(species=[n2, o2])
    assert len(tpg.species) == 2


def test_thermally_perfect_gas_temperature_ranges_count_mismatch_raises():
    """Test that species with different number of temperature ranges raise ValueError."""
    with SI_unit_system:
        # N2 has 2 ranges, O2 has 1 range
        n2 = FrozenSpecies(
            name="N2",
            nasa_9_coefficients=NASA9Coefficients(
                temperature_ranges=[
                    NASA9CoefficientSet(
                        temperature_range_min=200.0 * fl.u.K,
                        temperature_range_max=1000.0 * fl.u.K,
                        coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                    NASA9CoefficientSet(
                        temperature_range_min=1000.0 * fl.u.K,
                        temperature_range_max=6000.0 * fl.u.K,
                        coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                ]
            ),
            mass_fraction=0.75,
        )
        o2 = FrozenSpecies(
            name="O2",
            nasa_9_coefficients=NASA9Coefficients(
                temperature_ranges=[
                    NASA9CoefficientSet(
                        temperature_range_min=200.0 * fl.u.K,
                        temperature_range_max=6000.0 * fl.u.K,
                        coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                ]
            ),
            mass_fraction=0.25,
        )
        with pytest.raises(ValueError, match="same number of temperature ranges"):
            ThermallyPerfectGas(species=[n2, o2])


def test_thermally_perfect_gas_temperature_ranges_boundary_mismatch_raises():
    """Test that species with mismatched temperature boundaries raise ValueError."""
    with SI_unit_system:
        # N2: 200-1000K, O2: 200-1200K (different boundary)
        n2 = FrozenSpecies(
            name="N2",
            nasa_9_coefficients=NASA9Coefficients(
                temperature_ranges=[
                    NASA9CoefficientSet(
                        temperature_range_min=200.0 * fl.u.K,
                        temperature_range_max=1000.0 * fl.u.K,
                        coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                ]
            ),
            mass_fraction=0.75,
        )
        o2 = FrozenSpecies(
            name="O2",
            nasa_9_coefficients=NASA9Coefficients(
                temperature_ranges=[
                    NASA9CoefficientSet(
                        temperature_range_min=200.0 * fl.u.K,
                        temperature_range_max=1200.0 * fl.u.K,  # Different max
                        coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                ]
            ),
            mass_fraction=0.25,
        )
        with pytest.raises(ValueError, match="boundaries mismatch"):
            ThermallyPerfectGas(species=[n2, o2])


# =============================================================================
# Air with ThermallyPerfectGas Tests
# =============================================================================


def test_air_with_thermally_perfect_gas_valid():
    """Test creating Air material with ThermallyPerfectGas."""
    with SI_unit_system:
        tpg = ThermallyPerfectGas(
            species=[
                _make_species("N2", 0.7555),
                _make_species("O2", 0.2316),
                _make_species("Ar", 0.0129),
            ]
        )
        air = fl.Air(thermally_perfect_gas=tpg)
    assert air.thermally_perfect_gas is not None
    assert len(air.thermally_perfect_gas.species) == 3


def test_air_with_custom_thermally_perfect_gas_valid():
    """Test creating Air material with custom thermally perfect gas coefficients."""
    with SI_unit_system:
        air = fl.Air(
            thermally_perfect_gas=ThermallyPerfectGas(
                species=[
                    FrozenSpecies(
                        name="Air",
                        nasa_9_coefficients=NASA9Coefficients(
                            temperature_ranges=[
                                NASA9CoefficientSet(
                                    temperature_range_min=200.0 * fl.u.K,
                                    temperature_range_max=6000.0 * fl.u.K,
                                    coefficients=[0.0, 0.0, 3.5, 1e-4, 0.0, 0.0, 0.0, 0.0, 0.0],
                                )
                            ]
                        ),
                        mass_fraction=1.0,
                    )
                ]
            )
        )
    assert air.thermally_perfect_gas is not None
    assert len(air.thermally_perfect_gas.species) == 1
    assert len(air.thermally_perfect_gas.species[0].nasa_9_coefficients.temperature_ranges) == 1


# =============================================================================
# CompressibleIsentropic Solver with CPG Validation Tests
# =============================================================================


def _is_constant_gamma_coefficients(coefficients):
    """Helper to check if coefficients represent constant gamma (only a2 non-zero)."""
    tolerance = 1e-10
    for i in range(9):
        if i == 2:
            continue  # Skip a2
        if abs(coefficients[i]) > tolerance:
            return False
    return True


def test_is_constant_gamma_coefficients_cpg():
    """Test that CPG coefficients (only a2 non-zero) are identified as constant gamma."""
    # Default CPG: gamma=1.4, a2=3.5
    cpg_coeffs = [0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert _is_constant_gamma_coefficients(cpg_coeffs) is True

    # Different constant gamma (e.g., gamma=1.3)
    cpg_coeffs_different = [0.0, 0.0, 4.333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert _is_constant_gamma_coefficients(cpg_coeffs_different) is True


def test_is_constant_gamma_coefficients_tpg():
    """Test that TPG coefficients (temperature-dependent terms) are not constant gamma."""
    # a3 non-zero (linear temperature dependence)
    tpg_a3 = [0.0, 0.0, 3.5, 1e-4, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert _is_constant_gamma_coefficients(tpg_a3) is False

    # a0 non-zero (inverse T^2 dependence)
    tpg_a0 = [1e5, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert _is_constant_gamma_coefficients(tpg_a0) is False

    # a1 non-zero (inverse T dependence)
    tpg_a1 = [0.0, 100.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert _is_constant_gamma_coefficients(tpg_a1) is False

    # a4 non-zero (T^2 dependence)
    tpg_a4 = [0.0, 0.0, 3.5, 0.0, 1e-7, 0.0, 0.0, 0.0, 0.0]
    assert _is_constant_gamma_coefficients(tpg_a4) is False

    # a5 non-zero (T^3 dependence)
    tpg_a5 = [0.0, 0.0, 3.5, 0.0, 0.0, 1e-9, 0.0, 0.0, 0.0]
    assert _is_constant_gamma_coefficients(tpg_a5) is False

    # a6 non-zero (T^4 dependence)
    tpg_a6 = [0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 1e-9, 0.0, 0.0]
    assert _is_constant_gamma_coefficients(tpg_a6) is False

    # a7 non-zero (enthalpy integration constant)
    tpg_a7 = [0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0]
    assert _is_constant_gamma_coefficients(tpg_a7) is False

    # a8 non-zero (entropy integration constant)
    tpg_a8 = [0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]
    assert _is_constant_gamma_coefficients(tpg_a8) is False


# =============================================================================
# Additional Validation Tests
# =============================================================================


def test_nasa9_coefficient_set_temperature_range_min_less_than_max():
    """Test that temperature_range_min must be less than temperature_range_max."""
    with SI_unit_system:
        with pytest.raises(ValueError, match="must be less than"):
            NASA9CoefficientSet(
                temperature_range_min=1000.0 * fl.u.K,
                temperature_range_max=200.0 * fl.u.K,  # max < min
                coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )


def test_nasa9_coefficient_set_temperature_range_min_equals_max_raises():
    """Test that temperature_range_min cannot equal temperature_range_max."""
    with SI_unit_system:
        with pytest.raises(ValueError, match="must be less than"):
            NASA9CoefficientSet(
                temperature_range_min=500.0 * fl.u.K,
                temperature_range_max=500.0 * fl.u.K,  # min == max
                coefficients=[0.0, 0.0, 3.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )


def test_thermally_perfect_gas_duplicate_species_names_raises():
    """Test that duplicate species names raise ValueError."""
    with SI_unit_system:
        with pytest.raises(ValueError, match="Species names must be unique"):
            ThermallyPerfectGas(
                species=[
                    _make_species("N2", 0.5),
                    _make_species("N2", 0.5),  # Duplicate name
                ]
            )


def test_thermally_perfect_gas_mass_fraction_renormalization():
    """Test that mass fractions within tolerance are renormalized to sum to exactly 1.0."""
    with SI_unit_system:
        # Mass fractions sum to 0.9995 (within 1e-3 tolerance)
        tpg = ThermallyPerfectGas(
            species=[
                _make_species("N2", 0.7553),
                _make_species("O2", 0.2315),
                _make_species("Ar", 0.0127),  # Sum = 0.9995
            ]
        )
    # After renormalization, mass fractions should sum to exactly 1.0
    total = sum(s.mass_fraction for s in tpg.species)
    assert total == pytest.approx(1.0, abs=1e-10)


# =============================================================================
# Gas Class Tests
# =============================================================================


def _make_simple_tpg(gamma=1.4):
    """Helper to create a simple single-species ThermallyPerfectGas."""
    cp_r = gamma / (gamma - 1)
    return ThermallyPerfectGas(
        species=[
            FrozenSpecies(
                name="TestGas",
                nasa_9_coefficients=NASA9Coefficients(
                    temperature_ranges=[
                        NASA9CoefficientSet(
                            temperature_range_min=200.0 * fl.u.K,
                            temperature_range_max=6000.0 * fl.u.K,
                            coefficients=[0.0, 0.0, cp_r, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ),
                    ]
                ),
                mass_fraction=1.0,
            )
        ]
    )


def test_gas_with_custom_properties():
    """Test creating Gas with custom gas properties (e.g., CO2)."""
    with SI_unit_system:
        gas = Gas(
            gas_constant=188.92 * fl.u.m**2 / fl.u.s**2 / fl.u.K,
            dynamic_viscosity=1.47e-5 * fl.u.Pa * fl.u.s,
            thermally_perfect_gas=_make_simple_tpg(gamma=1.3),
            prandtl_number=0.77,
        )
    assert gas.type == "gas"
    assert gas.gas_constant.to("m**2/s**2/K").v.item() == pytest.approx(188.92)
    assert gas.prandtl_number == 0.77
    assert gas.turbulent_prandtl_number == 0.9  # default


def test_gas_with_sutherland_viscosity():
    """Test creating Gas with Sutherland viscosity model."""
    with SI_unit_system:
        gas = Gas(
            gas_constant=188.92 * fl.u.m**2 / fl.u.s**2 / fl.u.K,
            dynamic_viscosity=Sutherland(
                reference_viscosity=1.47e-5 * fl.u.Pa * fl.u.s,
                reference_temperature=293.15 * fl.u.K,
                effective_temperature=240.0 * fl.u.K,
            ),
            thermally_perfect_gas=_make_simple_tpg(),
            prandtl_number=0.77,
        )
    assert isinstance(gas.dynamic_viscosity, Sutherland)


def test_gas_speed_of_sound():
    """Test that Gas computes speed of sound correctly."""
    with SI_unit_system:
        gas = Gas(
            gas_constant=287.0529 * fl.u.m**2 / fl.u.s**2 / fl.u.K,
            dynamic_viscosity=1.716e-5 * fl.u.Pa * fl.u.s,
            thermally_perfect_gas=_make_simple_tpg(gamma=1.4),
            prandtl_number=0.72,
        )
    sos = gas.get_speed_of_sound(288.15 * fl.u.K)
    # For air at 288.15 K: sqrt(1.4 * 287.0529 * 288.15) ≈ 340.29 m/s
    assert sos.to("m/s").v.item() == pytest.approx(340.29, abs=0.5)


def test_gas_get_pressure():
    """Test that Gas computes pressure correctly via ideal gas law."""
    with SI_unit_system:
        gas = Gas(
            gas_constant=287.0529 * fl.u.m**2 / fl.u.s**2 / fl.u.K,
            dynamic_viscosity=1.716e-5 * fl.u.Pa * fl.u.s,
            thermally_perfect_gas=_make_simple_tpg(),
            prandtl_number=0.72,
        )
    # P = rho * R * T = 1.225 * 287.0529 * 288.15 ≈ 101325 Pa
    pressure = gas.get_pressure(1.225 * fl.u.kg / fl.u.m**3, 288.15 * fl.u.K)
    assert pressure.to("Pa").v.item() == pytest.approx(101325, rel=0.01)


def test_air_inherits_from_gas():
    """Test that Air inherits from Gas and is recognized as Gas."""
    with SI_unit_system:
        air = fl.Air()
    assert isinstance(air, Gas)
    assert isinstance(air, fl.Air)
    assert air.type == "air"
    assert air.gas_constant.to("m**2/s**2/K").v.item() == pytest.approx(287.0529)


def test_air_and_gas_produce_same_results():
    """Test that Air and Gas with air properties produce identical physics results."""
    with SI_unit_system:
        air = fl.Air()
        gas = Gas(
            gas_constant=287.0529 * fl.u.m**2 / fl.u.s**2 / fl.u.K,
            dynamic_viscosity=Sutherland(
                reference_viscosity=1.716e-5 * fl.u.Pa * fl.u.s,
                reference_temperature=273.15 * fl.u.K,
                effective_temperature=110.4 * fl.u.K,
            ),
            thermally_perfect_gas=_make_simple_tpg(gamma=1.4),
            prandtl_number=0.72,
        )
    temp = 288.15 * fl.u.K
    assert air.get_speed_of_sound(temp).to("m/s").v.item() == pytest.approx(
        gas.get_speed_of_sound(temp).to("m/s").v.item(), rel=1e-6
    )
    density = 1.225 * fl.u.kg / fl.u.m**3
    assert air.get_pressure(density, temp).to("Pa").v.item() == pytest.approx(
        gas.get_pressure(density, temp).to("Pa").v.item(), rel=1e-6
    )


def test_gas_serialization_roundtrip():
    """Test that Gas serializes and deserializes correctly."""
    with SI_unit_system:
        gas = Gas(
            gas_constant=188.92 * fl.u.m**2 / fl.u.s**2 / fl.u.K,
            dynamic_viscosity=1.47e-5 * fl.u.Pa * fl.u.s,
            thermally_perfect_gas=_make_simple_tpg(),
            prandtl_number=0.77,
        )
    data = gas.model_dump(mode="json")
    assert data["type"] == "gas"
    assert data["gas_constant"]["value"] == pytest.approx(188.92)

    # Deserialize
    with SI_unit_system:
        gas2 = Gas(**data)
    assert gas2.type == "gas"
    assert gas2.gas_constant.to("m**2/s**2/K").v.item() == pytest.approx(188.92)
