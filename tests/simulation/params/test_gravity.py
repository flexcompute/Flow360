import math
import re

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.models.volume_models import Gravity
from flow360.component.simulation.primitives import GenericVolume
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.solver_translator import (
    gravity_entity_info_serializer,
    gravity_translator,
)
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.validation.validation_context import (
    ParamsValidationInfo,
    ValidationContext,
)


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


def test_gravity_creation_basic():
    """Test basic Gravity model creation with default units."""
    gravity = Gravity(
        direction=(0, 0, -1),
        magnitude=9.81 * u.m / u.s**2,
    )
    assert gravity.type == "Gravity"
    assert gravity.name == "Gravity"
    assert gravity.entities is None  # Default to all zones
    # Direction should be normalized (already is in this case)
    assert tuple(gravity.direction) == (0.0, 0.0, -1.0)


def test_gravity_direction_normalization():
    """Test that direction is normalized automatically."""
    gravity = Gravity(
        direction=(0, 3, -4),  # magnitude is 5, will be normalized
        magnitude=9.81 * u.m / u.s**2,
    )
    # Expected: (0, 0.6, -0.8)
    assert math.isclose(gravity.direction[0], 0.0, abs_tol=1e-10)
    assert math.isclose(gravity.direction[1], 0.6, rel_tol=1e-10)
    assert math.isclose(gravity.direction[2], -0.8, rel_tol=1e-10)


def test_gravity_zero_direction_raises():
    """Test that zero direction vector raises an error."""
    with pytest.raises(ValueError, match=re.escape("Axis cannot be (0, 0, 0)")):
        Gravity(
            direction=(0, 0, 0),
            magnitude=9.81 * u.m / u.s**2,
        )


def test_gravity_with_entities():
    """Test Gravity model with specific entities."""
    mock_context = ValidationContext(
        levels=None, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )

    with mock_context:
        gravity = Gravity(
            entities=[GenericVolume(name="fluid_zone")],
            direction=(0, 0, -1),
            magnitude=9.81 * u.m / u.s**2,
        )
        assert gravity.entities is not None
        assert len(gravity.entities.stored_entities) == 1


def test_gravity_different_units():
    """Test Gravity model with different acceleration units."""
    # Using ft/s^2
    gravity_imperial = Gravity(
        direction=(0, 0, -1),
        magnitude=32.174 * u.ft / u.s**2,  # ~9.8 m/s^2
    )
    # Verify magnitude is stored correctly
    assert gravity_imperial.magnitude.to("m/s**2").value > 9.7
    assert gravity_imperial.magnitude.to("m/s**2").value < 9.9


def test_gravity_custom_name():
    """Test Gravity model with custom name."""
    gravity = Gravity(
        name="Earth Gravity",
        direction=(0, 0, -1),
        magnitude=9.81 * u.m / u.s**2,
    )
    assert gravity.name == "Earth Gravity"


def test_gravity_arbitrary_direction():
    """Test Gravity model with arbitrary direction."""
    gravity = Gravity(
        direction=(1, 1, 1),
        magnitude=9.81 * u.m / u.s**2,
    )
    # Should be normalized to unit vector
    norm = math.sqrt(sum(d**2 for d in gravity.direction))
    assert math.isclose(norm, 1.0, rel_tol=1e-10)


def test_gravity_translator_nondimensionalization():
    """Test that gravity_translator correctly non-dimensionalizes the gravity vector.

    Non-dimensionalization: g* = g * L_ref / a_∞²
    """
    gravity = Gravity(
        direction=(0, 0, -1),
        magnitude=9.81 * u.m / u.s**2,
    )

    # With L_ref = 1 m and a_∞ = 340 m/s (speed of sound at sea level)
    mock_params = MockSimulationParams(base_length_m=1.0, base_velocity_ms=340.0)

    result = gravity_translator(gravity, mock_params)

    # Expected: g* = 9.81 * 1 / 340^2 = 9.81 / 115600 ≈ 8.49e-5
    expected_nondim = 9.81 / (340.0**2)

    assert "gravityVector" in result
    assert len(result["gravityVector"]) == 3
    assert math.isclose(result["gravityVector"][0], 0.0, abs_tol=1e-15)
    assert math.isclose(result["gravityVector"][1], 0.0, abs_tol=1e-15)
    assert math.isclose(result["gravityVector"][2], -expected_nondim, rel_tol=1e-5)


def test_gravity_translator_direction():
    """Test that gravity_translator preserves direction correctly."""
    gravity = Gravity(
        direction=(1, 0, 0),  # Gravity in +x direction
        magnitude=10.0 * u.m / u.s**2,
    )

    mock_params = MockSimulationParams(base_length_m=1.0, base_velocity_ms=100.0)
    result = gravity_translator(gravity, mock_params)

    # Expected: g* = 10 * 1 / 100^2 = 10 / 10000 = 0.001
    expected_nondim = 10.0 / (100.0**2)

    assert math.isclose(result["gravityVector"][0], expected_nondim, rel_tol=1e-5)
    assert math.isclose(result["gravityVector"][1], 0.0, abs_tol=1e-15)
    assert math.isclose(result["gravityVector"][2], 0.0, abs_tol=1e-15)


def test_gravity_entity_info_serializer_global():
    """Test gravity entity serializer for global gravity."""
    result = gravity_entity_info_serializer(None)
    assert result == {"zoneType": "global"}


def test_gravity_entity_info_serializer_zone():
    """Test gravity entity serializer for zone-specific gravity."""
    mock_context = ValidationContext(
        levels=None, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )

    with mock_context:
        volume = GenericVolume(name="test_zone")
        result = gravity_entity_info_serializer(volume)

        assert result["zoneType"] == "mesh"
        assert result["zoneName"] == volume.full_name


def test_gravity_very_small_direction():
    """Test that very small but non-zero direction is normalized correctly."""
    gravity = Gravity(
        direction=(1e-10, 1e-10, 1e-10),
        magnitude=9.81 * u.m / u.s**2,
    )
    norm = math.sqrt(sum(d**2 for d in gravity.direction))
    assert math.isclose(norm, 1.0, rel_tol=1e-10)


def test_gravity_negative_direction_components():
    """Test Gravity model with all negative direction components."""
    gravity = Gravity(
        direction=(-1, -1, -1),
        magnitude=9.81 * u.m / u.s**2,
    )
    # All components should be negative and equal after normalization
    expected = -1.0 / math.sqrt(3)
    assert math.isclose(gravity.direction[0], expected, rel_tol=1e-10)
    assert math.isclose(gravity.direction[1], expected, rel_tol=1e-10)
    assert math.isclose(gravity.direction[2], expected, rel_tol=1e-10)


def test_gravity_large_magnitude():
    """Test Gravity model with large magnitude (e.g., Jupiter-like)."""
    gravity = Gravity(
        direction=(0, 0, -1),
        magnitude=24.79 * u.m / u.s**2,  # Jupiter's surface gravity
    )
    assert gravity.magnitude.to("m/s**2").value > 24.0
    assert gravity.magnitude.to("m/s**2").value < 25.0


def test_gravity_small_magnitude():
    """Test Gravity model with small magnitude (e.g., Moon-like)."""
    gravity = Gravity(
        direction=(0, 0, -1),
        magnitude=1.62 * u.m / u.s**2,  # Moon's surface gravity
    )
    assert gravity.magnitude.to("m/s**2").value > 1.6
    assert gravity.magnitude.to("m/s**2").value < 1.7


def test_single_global_gravity_is_valid():
    """A single Gravity model with entities=None (global) should be accepted."""
    gravity = Gravity(
        direction=(0, 0, -1),
        magnitude=9.81 * u.m / u.s**2,
    )
    with SI_unit_system:
        params = SimulationParams(models=[gravity])
    assert params


def test_multiple_global_gravity_raises():
    """Two Gravity models with entities=None should raise a conflict error."""
    gravity1 = Gravity(
        direction=(0, 0, -1),
        magnitude=9.81 * u.m / u.s**2,
    )
    gravity2 = Gravity(
        direction=(1, 0, 0),
        magnitude=5.0 * u.m / u.s**2,
    )
    with SI_unit_system, pytest.raises(
        ValueError,
        match=re.escape(
            "Multiple Gravity models with unspecified entities (applying to all zones) "
            "are not allowed."
        ),
    ):
        SimulationParams(models=[gravity1, gravity2])


def test_global_gravity_with_zone_specific_raises():
    """A global Gravity (entities=None) mixed with zone-specific Gravity should raise."""
    mock_context = ValidationContext(
        levels=None, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )
    global_gravity = Gravity(
        direction=(0, 0, -1),
        magnitude=9.81 * u.m / u.s**2,
    )
    with mock_context:
        zone_gravity = Gravity(
            entities=[GenericVolume(name="zone1")],
            direction=(1, 0, 0),
            magnitude=5.0 * u.m / u.s**2,
        )
    with SI_unit_system, pytest.raises(
        ValueError,
        match=re.escape(
            "A Gravity model that applies to all zones (entities not specified) "
            "cannot coexist with other Gravity models."
        ),
    ):
        SimulationParams(models=[global_gravity, zone_gravity])


def test_multiple_zone_specific_gravity_is_valid():
    """Multiple Gravity models with distinct entities should be accepted."""
    mock_context = ValidationContext(
        levels=None, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )
    with mock_context:
        gravity1 = Gravity(
            entities=[GenericVolume(name="zone1")],
            direction=(0, 0, -1),
            magnitude=9.81 * u.m / u.s**2,
        )
        gravity2 = Gravity(
            entities=[GenericVolume(name="zone2")],
            direction=(1, 0, 0),
            magnitude=5.0 * u.m / u.s**2,
        )
    with SI_unit_system:
        params = SimulationParams(models=[gravity1, gravity2])
    assert params
