"""Tests for results_utils functions."""

import numpy as np
import pytest

import flow360 as fl
from flow360.component.results.results_utils import _get_lift_drag_direction
from flow360.component.simulation.framework.param_utils import AssetCache


def test_get_lift_drag_direction_with_generic_reference_condition():
    """Test that GenericReferenceCondition returns default lift/drag directions."""
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m,
                moment_length=1 * fl.u.m,
                area=1.0 * fl.u.m**2,
            ),
            operating_condition=fl.GenericReferenceCondition(
                velocity_magnitude=10 * fl.u.m / fl.u.s,
            ),
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )

    lift_dir, drag_dir = _get_lift_drag_direction(params)

    # Expected default directions for alpha=0, beta=0
    expected_lift_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    expected_drag_dir = np.array([1.0, 0.0, 0.0], dtype=float)

    assert np.allclose(lift_dir, expected_lift_dir, atol=1e-12)
    assert np.allclose(drag_dir, expected_drag_dir, atol=1e-12)


def test_get_lift_drag_direction_with_aerospace_condition():
    """Test that AerospaceCondition correctly calculates lift/drag directions."""
    alpha_deg = 10.0
    beta_deg = 5.0

    with fl.SI_unit_system:
        params = fl.SimulationParams(
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m,
                moment_length=1 * fl.u.m,
                area=1.0 * fl.u.m**2,
            ),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=10 * fl.u.m / fl.u.s,
                alpha=alpha_deg * fl.u.deg,
                beta=beta_deg * fl.u.deg,
            ),
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )

    lift_dir, drag_dir = _get_lift_drag_direction(params)

    # Compute expected values
    alpha_rad = np.deg2rad(alpha_deg)
    beta_rad = np.deg2rad(beta_deg)

    expected_u_inf = np.array(
        [
            np.cos(alpha_rad) * np.cos(beta_rad),
            -np.sin(beta_rad),
            np.sin(alpha_rad) * np.cos(beta_rad),
        ],
        dtype=float,
    )
    expected_lift_dir = np.array([-np.sin(alpha_rad), 0.0, np.cos(alpha_rad)], dtype=float)
    expected_drag_dir = expected_u_inf

    assert np.allclose(lift_dir, expected_lift_dir, atol=1e-12)
    assert np.allclose(drag_dir, expected_drag_dir, atol=1e-12)


def test_get_lift_drag_direction_with_aerospace_condition_zero_angles():
    """Test AerospaceCondition with alpha=0, beta=0 matches GenericReferenceCondition."""
    with fl.SI_unit_system:
        params_aerospace = fl.SimulationParams(
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m,
                moment_length=1 * fl.u.m,
                area=1.0 * fl.u.m**2,
            ),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=10 * fl.u.m / fl.u.s,
                alpha=0 * fl.u.deg,
                beta=0 * fl.u.deg,
            ),
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )

        params_generic = fl.SimulationParams(
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m,
                moment_length=1 * fl.u.m,
                area=1.0 * fl.u.m**2,
            ),
            operating_condition=fl.GenericReferenceCondition(
                velocity_magnitude=10 * fl.u.m / fl.u.s,
            ),
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )

    lift_dir_aerospace, drag_dir_aerospace = _get_lift_drag_direction(params_aerospace)
    lift_dir_generic, drag_dir_generic = _get_lift_drag_direction(params_generic)

    # Both should give the same result for alpha=0, beta=0
    assert np.allclose(lift_dir_aerospace, lift_dir_generic, atol=1e-12)
    assert np.allclose(drag_dir_aerospace, drag_dir_generic, atol=1e-12)


def test_get_lift_drag_direction_with_liquid_operating_condition():
    """Test that LiquidOperatingCondition correctly calculates lift/drag directions."""
    alpha_deg = 15.0
    beta_deg = -5.0

    with fl.SI_unit_system:
        params = fl.SimulationParams(
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m,
                moment_length=1 * fl.u.m,
                area=1.0 * fl.u.m**2,
            ),
            operating_condition=fl.LiquidOperatingCondition(
                velocity_magnitude=20 * fl.u.m / fl.u.s,
                reference_velocity_magnitude=20 * fl.u.m / fl.u.s,
                alpha=alpha_deg * fl.u.deg,
                beta=beta_deg * fl.u.deg,
            ),
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )

    lift_dir, drag_dir = _get_lift_drag_direction(params)

    # Compute expected values
    alpha_rad = np.deg2rad(alpha_deg)
    beta_rad = np.deg2rad(beta_deg)

    expected_u_inf = np.array(
        [
            np.cos(alpha_rad) * np.cos(beta_rad),
            -np.sin(beta_rad),
            np.sin(alpha_rad) * np.cos(beta_rad),
        ],
        dtype=float,
    )
    expected_lift_dir = np.array([-np.sin(alpha_rad), 0.0, np.cos(alpha_rad)], dtype=float)
    expected_drag_dir = expected_u_inf

    assert np.allclose(lift_dir, expected_lift_dir, atol=1e-12)
    assert np.allclose(drag_dir, expected_drag_dir, atol=1e-12)


def test_get_lift_drag_direction_lift_perpendicular_to_drag():
    """Test that lift direction is perpendicular to drag direction when beta=0."""
    alpha_deg = 25.0

    with fl.SI_unit_system:
        params = fl.SimulationParams(
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m,
                moment_length=1 * fl.u.m,
                area=1.0 * fl.u.m**2,
            ),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=10 * fl.u.m / fl.u.s,
                alpha=alpha_deg * fl.u.deg,
                beta=0 * fl.u.deg,
            ),
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )

    lift_dir, drag_dir = _get_lift_drag_direction(params)

    # When beta=0, lift should be perpendicular to drag
    dot_product = np.dot(lift_dir, drag_dir)
    assert np.isclose(dot_product, 0.0, atol=1e-12)


def test_get_lift_drag_direction_unit_vectors():
    """Test that returned directions are unit vectors."""
    with fl.SI_unit_system:
        # Test with GenericReferenceCondition
        params_generic = fl.SimulationParams(
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m,
                moment_length=1 * fl.u.m,
                area=1.0 * fl.u.m**2,
            ),
            operating_condition=fl.GenericReferenceCondition(
                velocity_magnitude=10 * fl.u.m / fl.u.s,
            ),
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )

        # Test with AerospaceCondition
        params_aerospace = fl.SimulationParams(
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m,
                moment_length=1 * fl.u.m,
                area=1.0 * fl.u.m**2,
            ),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=10 * fl.u.m / fl.u.s,
                alpha=30 * fl.u.deg,
                beta=15 * fl.u.deg,
            ),
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )

    # Check GenericReferenceCondition
    lift_dir, drag_dir = _get_lift_drag_direction(params_generic)
    assert np.isclose(np.linalg.norm(lift_dir), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(drag_dir), 1.0, atol=1e-12)

    # Check AerospaceCondition
    lift_dir, drag_dir = _get_lift_drag_direction(params_aerospace)
    assert np.isclose(np.linalg.norm(lift_dir), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(drag_dir), 1.0, atol=1e-12)
