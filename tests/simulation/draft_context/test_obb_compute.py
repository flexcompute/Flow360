"""Tests for OBB (Oriented Bounding Box) computation via PCA."""

import numpy as np
import pytest

from flow360.component.simulation.draft_context.obb.compute import (
    OBBResult,
    compute_obb,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _box_vertices(half_x, half_y, half_z, center=(0, 0, 0)):
    """Generate 8 corner vertices of an axis-aligned box."""
    cx, cy, cz = center
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return signs * np.array([half_x, half_y, half_z]) + np.array([cx, cy, cz])


def _cylinder_vertices(radius, half_height, axis="z", n_ring=64, n_height=10):
    """Generate a sampled cylinder surface for testing."""
    theta = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    heights = np.linspace(-half_height, half_height, n_height)
    rows = []
    for h in heights:
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.full_like(theta, h)
        if axis == "z":
            rows.append(np.column_stack([x, y, z]))
        elif axis == "x":
            rows.append(np.column_stack([z, x, y]))
        elif axis == "y":
            rows.append(np.column_stack([x, z, y]))
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# compute_obb tests
# ---------------------------------------------------------------------------


class TestComputeObb:
    def test_axis_aligned_box(self):
        """OBB of an axis-aligned box should recover the half-extents."""
        verts = _box_vertices(3, 2, 1)
        obb = compute_obb(verts)

        sorted_extents = np.sort(obb.extents)[::-1]
        np.testing.assert_allclose(sorted_extents, [3, 2, 1], atol=1e-10)

    def test_center_at_origin(self):
        verts = _box_vertices(1, 1, 1)
        obb = compute_obb(verts)
        np.testing.assert_allclose(obb.center, [0, 0, 0], atol=1e-10)

    def test_translated_box(self):
        verts = _box_vertices(2, 1, 1, center=(10, 20, 30))
        obb = compute_obb(verts)
        np.testing.assert_allclose(obb.center, [10, 20, 30], atol=1e-10)

    def test_axes_are_orthonormal(self):
        verts = _box_vertices(5, 3, 1)
        obb = compute_obb(verts)

        # Each axis should be unit length
        for i in range(3):
            np.testing.assert_allclose(np.linalg.norm(obb.axes[i]), 1.0, atol=1e-10)

        # Axes should be mutually orthogonal
        gram = obb.axes @ obb.axes.T
        np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)

    def test_right_handed_coordinate_system(self):
        verts = _box_vertices(5, 3, 1)
        obb = compute_obb(verts)
        det = np.linalg.det(obb.axes)
        assert det > 0, f"determinant should be positive (right-handed), got {det}"

    def test_rotated_box(self):
        """A 45-degree rotated box should still recover correct extents."""
        angle = np.pi / 4
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        verts = _box_vertices(4, 1, 1) @ rotation.T
        obb = compute_obb(verts)

        sorted_extents = np.sort(obb.extents)[::-1]
        np.testing.assert_allclose(sorted_extents, [4, 1, 1], atol=1e-10)


# ---------------------------------------------------------------------------
# OBBResult method tests
# ---------------------------------------------------------------------------


class TestObbResultRotationAxis:
    def test_hint_selects_closest_axis(self):
        """axis_of_rotation with a hint should pick the most-aligned OBB axis."""
        obb = OBBResult(
            center=np.zeros(3),
            axes=np.eye(3),  # x, y, z as rows
            extents=np.array([5.0, 1.0, 1.0]),
        )
        axis = obb.axis_of_rotation(rotation_axis_hint=np.array([1, 0, 0]))
        np.testing.assert_allclose(axis, [1, 0, 0], atol=1e-10)

    def test_hint_handles_non_unit_vector(self):
        obb = OBBResult(
            center=np.zeros(3),
            axes=np.eye(3),
            extents=np.array([5.0, 1.0, 1.0]),
        )
        axis = obb.axis_of_rotation(rotation_axis_hint=np.array([100, 0, 0]))
        np.testing.assert_allclose(axis, [1, 0, 0], atol=1e-10)

    def test_circularity_heuristic_picks_cylinder_axis(self):
        """Without a hint, the axis whose perpendicular extents are most equal wins."""
        # Cylinder-like: long along axis 0, circular cross-section (equal extents 1, 2)
        obb = OBBResult(
            center=np.zeros(3),
            axes=np.eye(3),
            extents=np.array([10.0, 2.0, 2.0]),
        )
        axis = obb.axis_of_rotation()
        np.testing.assert_allclose(axis, [1, 0, 0], atol=1e-10)

    def test_circularity_with_slight_asymmetry(self):
        """Even slightly unequal perpendicular extents should still pick the best axis."""
        obb = OBBResult(
            center=np.zeros(3),
            axes=np.eye(3),
            extents=np.array([10.0, 2.1, 1.9]),
        )
        axis = obb.axis_of_rotation()
        np.testing.assert_allclose(axis, [1, 0, 0], atol=1e-10)


class TestObbResultRadius:
    def test_radius_is_average_of_perpendicular_extents(self):
        obb = OBBResult(
            center=np.zeros(3),
            axes=np.eye(3),
            extents=np.array([10.0, 3.0, 5.0]),
        )
        # Rotation axis = index 0 (circularity: perpendicular extents 3, 5 ratio 0.6)
        # Compare with axis 1 (perp 10, 5, ratio 0.5) and axis 2 (perp 10, 3, ratio 0.3)
        # So axis 0 wins; radius = (3 + 5) / 2 = 4.0
        assert obb.radius() == pytest.approx(4.0)

    def test_radius_with_hint(self):
        obb = OBBResult(
            center=np.zeros(3),
            axes=np.eye(3),
            extents=np.array([10.0, 3.0, 5.0]),
        )
        # Force axis 2 as rotation axis; perpendicular extents are 10.0 and 3.0
        radius = obb.radius(rotation_axis_hint=np.array([0, 0, 1]))
        assert radius == pytest.approx(6.5)

    def test_radius_for_perfect_cylinder(self):
        verts = _cylinder_vertices(radius=5.0, half_height=20.0, axis="z")
        obb = compute_obb(verts)
        # The radius should be close to 5.0
        assert obb.radius(rotation_axis_hint=np.array([0, 0, 1])) == pytest.approx(5.0, abs=0.2)
