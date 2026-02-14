import math
import re

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.volume_models import Fluid, Gravity
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.solver_translator import gravity_translator
from flow360.component.simulation.unit_system import SI_unit_system


class MockSimulationParams:
    """Mock class for testing gravity_translator"""

    def __init__(self, base_length_m=1.0, base_velocity_ms=340.0):
        self._base_length = base_length_m * u.m
        self._base_velocity = base_velocity_ms * u.m / u.s

    @property
    def base_length(self):
        return self._base_length

    @property
    def base_velocity(self):
        return self._base_velocity


# ============================================================================
# Gravity data class tests
# ============================================================================


def test_gravity_defaults():
    """Test Gravity with default values (Earth gravity, downward)."""
    gravity = Gravity()
    assert tuple(gravity.direction) == (0.0, 0.0, -1.0)
    assert math.isclose(gravity.magnitude.to("m/s**2").value, 9.81, rel_tol=1e-10)


def test_gravity_custom_values():
    """Test Gravity with custom direction and magnitude."""
    gravity = Gravity(
        direction=(1, 0, 0),
        magnitude=5.0 * u.m / u.s**2,
    )
    assert tuple(gravity.direction) == (1.0, 0.0, 0.0)
    assert math.isclose(gravity.magnitude.to("m/s**2").value, 5.0, rel_tol=1e-10)


def test_gravity_direction_normalization():
    """Test that direction is normalized automatically."""
    gravity = Gravity(
        direction=(0, 3, -4),  # magnitude is 5, will be normalized
    )
    # Expected: (0, 0.6, -0.8)
    assert math.isclose(gravity.direction[0], 0.0, abs_tol=1e-10)
    assert math.isclose(gravity.direction[1], 0.6, rel_tol=1e-10)
    assert math.isclose(gravity.direction[2], -0.8, rel_tol=1e-10)


def test_gravity_zero_direction_raises():
    """Test that zero direction vector raises an error."""
    with pytest.raises(ValueError, match=re.escape("Axis cannot be (0, 0, 0)")):
        Gravity(direction=(0, 0, 0))


def test_gravity_different_units():
    """Test Gravity with different acceleration units."""
    gravity = Gravity(
        direction=(0, 0, -1),
        magnitude=32.174 * u.ft / u.s**2,  # ~9.8 m/s^2
    )
    assert gravity.magnitude.to("m/s**2").value > 9.7
    assert gravity.magnitude.to("m/s**2").value < 9.9


def test_gravity_arbitrary_direction():
    """Test Gravity with arbitrary direction is normalized to unit vector."""
    gravity = Gravity(direction=(1, 1, 1))
    norm = math.sqrt(sum(d**2 for d in gravity.direction))
    assert math.isclose(norm, 1.0, rel_tol=1e-10)


def test_gravity_very_small_direction():
    """Test that very small but non-zero direction is normalized correctly."""
    gravity = Gravity(direction=(1e-10, 1e-10, 1e-10))
    norm = math.sqrt(sum(d**2 for d in gravity.direction))
    assert math.isclose(norm, 1.0, rel_tol=1e-10)


def test_gravity_negative_direction_components():
    """Test Gravity with all negative direction components."""
    gravity = Gravity(direction=(-1, -1, -1))
    expected = -1.0 / math.sqrt(3)
    assert math.isclose(gravity.direction[0], expected, rel_tol=1e-10)
    assert math.isclose(gravity.direction[1], expected, rel_tol=1e-10)
    assert math.isclose(gravity.direction[2], expected, rel_tol=1e-10)


def test_gravity_large_magnitude():
    """Test Gravity with large magnitude (e.g., Jupiter-like)."""
    gravity = Gravity(
        direction=(0, 0, -1),
        magnitude=24.79 * u.m / u.s**2,
    )
    assert gravity.magnitude.to("m/s**2").value > 24.0
    assert gravity.magnitude.to("m/s**2").value < 25.0


def test_gravity_small_magnitude():
    """Test Gravity with small magnitude (e.g., Moon-like)."""
    gravity = Gravity(
        direction=(0, 0, -1),
        magnitude=1.62 * u.m / u.s**2,
    )
    assert gravity.magnitude.to("m/s**2").value > 1.6
    assert gravity.magnitude.to("m/s**2").value < 1.7


# ============================================================================
# Fluid + Gravity integration tests
# ============================================================================


def test_fluid_without_gravity():
    """Test that Fluid defaults to no gravity."""
    fluid = Fluid()
    assert fluid.gravity is None


def test_fluid_with_default_gravity():
    """Test Fluid with default Earth gravity."""
    fluid = Fluid(gravity=Gravity())
    assert fluid.gravity is not None
    assert tuple(fluid.gravity.direction) == (0.0, 0.0, -1.0)
    assert math.isclose(fluid.gravity.magnitude.to("m/s**2").value, 9.81, rel_tol=1e-10)


def test_fluid_with_custom_gravity():
    """Test Fluid with custom gravity settings."""
    fluid = Fluid(
        gravity=Gravity(
            direction=(1, 0, 0),
            magnitude=5.0 * u.m / u.s**2,
        )
    )
    assert fluid.gravity is not None
    assert tuple(fluid.gravity.direction) == (1.0, 0.0, 0.0)
    assert math.isclose(fluid.gravity.magnitude.to("m/s**2").value, 5.0, rel_tol=1e-10)


def test_fluid_with_gravity_in_simulation_params():
    """Fluid with gravity should be accepted in SimulationParams."""
    fluid = Fluid(gravity=Gravity())
    with SI_unit_system:
        params = SimulationParams(models=[fluid])
    assert params


# ============================================================================
# Translator tests
# ============================================================================


def test_gravity_translator_nondimensionalization():
    """Test that gravity_translator correctly non-dimensionalizes.

    Non-dimensionalization: g* = g * L_ref / a_inf^2
    """
    gravity = Gravity(
        direction=(0, 0, -1),
        magnitude=9.81 * u.m / u.s**2,
    )

    mock_params = MockSimulationParams(base_length_m=1.0, base_velocity_ms=340.0)
    result = gravity_translator(gravity, mock_params)

    expected_nondim = 9.81 / (340.0**2)

    assert "gravityVector" in result
    assert len(result["gravityVector"]) == 3
    # direction is (0,0,-1), so gravityVector = (0, 0, -expected_nondim)
    assert math.isclose(result["gravityVector"][0], 0.0, abs_tol=1e-15)
    assert math.isclose(result["gravityVector"][1], 0.0, abs_tol=1e-15)
    assert math.isclose(result["gravityVector"][2], -expected_nondim, rel_tol=1e-5)


def test_gravity_translator_custom_direction():
    """Test that gravity_translator preserves direction correctly."""
    gravity = Gravity(
        direction=(1, 0, 0),
        magnitude=10.0 * u.m / u.s**2,
    )

    mock_params = MockSimulationParams(base_length_m=1.0, base_velocity_ms=100.0)
    result = gravity_translator(gravity, mock_params)

    expected_nondim = 10.0 / (100.0**2)

    assert "gravityVector" in result
    # direction is (1,0,0), so gravityVector = (expected_nondim, 0, 0)
    assert math.isclose(result["gravityVector"][0], expected_nondim, rel_tol=1e-5)
    assert math.isclose(result["gravityVector"][1], 0.0, abs_tol=1e-15)
    assert math.isclose(result["gravityVector"][2], 0.0, abs_tol=1e-15)
