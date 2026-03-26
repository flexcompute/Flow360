"""Tests for OBB (Oriented Bounding Box) computation via PCA."""

import numpy as np
import pytest

from flow360.component.simulation.draft_context.obb.compute import (
    OBBResult,
    _select_rotation_axis_index,
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

        for i in range(3):
            np.testing.assert_allclose(np.linalg.norm(obb.axes[i]), 1.0, atol=1e-10)

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

    def test_result_has_all_fields(self):
        verts = _box_vertices(3, 2, 1)
        obb = compute_obb(verts)
        assert obb.center is not None
        assert obb.axes is not None
        assert obb.extents is not None
        assert obb.axis_of_rotation is not None
        assert obb.radius is not None


# ---------------------------------------------------------------------------
# Rotation axis selection tests
# ---------------------------------------------------------------------------


class TestRotationAxisSelection:
    def test_hint_selects_closest_axis(self):
        axes = np.eye(3)
        extents = np.array([5.0, 1.0, 1.0])
        idx = _select_rotation_axis_index(axes, extents, rotation_axis_hint=np.array([1, 0, 0]))
        assert idx == 0

    def test_hint_handles_non_unit_vector(self):
        axes = np.eye(3)
        extents = np.array([5.0, 1.0, 1.0])
        idx = _select_rotation_axis_index(axes, extents, rotation_axis_hint=np.array([100, 0, 0]))
        assert idx == 0

    def test_circularity_heuristic_picks_cylinder_axis(self):
        """Without a hint, the axis whose perpendicular extents are most equal wins."""
        axes = np.eye(3)
        extents = np.array([10.0, 2.0, 2.0])
        idx = _select_rotation_axis_index(axes, extents, rotation_axis_hint=None)
        assert idx == 0

    def test_circularity_with_slight_asymmetry(self):
        axes = np.eye(3)
        extents = np.array([10.0, 2.1, 1.9])
        idx = _select_rotation_axis_index(axes, extents, rotation_axis_hint=None)
        assert idx == 0


class TestComputeObbRotationAxis:
    def test_hint_baked_into_result(self):
        """rotation_axis_hint at compute_obb time is baked into the result fields."""
        verts = _cylinder_vertices(radius=5.0, half_height=20.0, axis="z")
        obb = compute_obb(verts, rotation_axis_hint=[0, 0, 1])
        assert abs(np.dot(obb.axis_of_rotation, [0, 0, 1])) > 0.99

    def test_default_circularity(self):
        """Without hint, circularity picks the cylinder axis."""
        verts = _cylinder_vertices(radius=5.0, half_height=20.0, axis="z")
        obb = compute_obb(verts)
        # Z-axis cylinder: rotation axis should align with Z
        assert abs(np.dot(obb.axis_of_rotation, [0, 0, 1])) > 0.99

    def test_radius_with_hint(self):
        verts = _box_vertices(10, 3, 5)
        obb = compute_obb(verts, rotation_axis_hint=[0, 0, 1])
        # Hint Z → axis 2 → perpendicular extents are 10 and 3 → radius = 6.5
        assert obb.radius == pytest.approx(6.5)

    def test_radius_default(self):
        verts = _box_vertices(10, 3, 5)
        obb = compute_obb(verts)
        # Circularity: axis 0 (perp 3,5 ratio 0.6) wins → radius = (3+5)/2 = 4.0
        assert obb.radius == pytest.approx(4.0)

    def test_radius_for_perfect_cylinder(self):
        verts = _cylinder_vertices(radius=5.0, half_height=20.0, axis="z")
        obb = compute_obb(verts, rotation_axis_hint=[0, 0, 1])
        assert obb.radius == pytest.approx(5.0, abs=0.2)
