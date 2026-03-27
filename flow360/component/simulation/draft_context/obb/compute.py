"""Oriented Bounding Box computation via PCA.

Computes an OBB from an (N, 3) vertex point cloud, with helpers to derive
rotation axis and radius for cylindrical geometry estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


def _select_rotation_axis_index(
    axes: np.ndarray,
    extents: np.ndarray,
    rotation_axis_hint: Optional[np.ndarray],
) -> int:
    """Determine which OBB axis is the rotation axis.

    If *rotation_axis_hint* is provided, picks the axis most aligned with it.
    Otherwise infers by circularity — the axis whose perpendicular cross-section
    has the most equal pair of extents.
    """
    if rotation_axis_hint is not None:
        hint = np.asarray(rotation_axis_hint, dtype=np.float64)
        dots = np.abs(axes @ hint)
        return int(np.argmax(dots))

    # Circularity heuristic: for each axis, ratio of the two perpendicular extents
    best_index = 0
    best_ratio = -1.0
    for i in range(3):
        others: List[float] = [extents[j] for j in range(3) if j != i]
        ratio = others[0] / others[1] if others[1] > others[0] else others[1] / others[0]
        if ratio > best_ratio:
            best_ratio = ratio
            best_index = i
    return best_index


@dataclass(frozen=True)
class OBBResult:
    """Oriented Bounding Box computed from a point cloud.

    All fields are properties — no method calls needed.

    Attributes:
        center: (3,) geometric center of the OBB.
        axes: (3, 3) principal axes as row vectors, descending by extent magnitude.
        extents: (3,) half-extents along each axis.
        axis_of_rotation: (3,) unit vector along the inferred rotation axis.
        radius: estimated cylinder radius perpendicular to the rotation axis.
    """

    center: np.ndarray
    axes: np.ndarray
    extents: np.ndarray
    axis_of_rotation: np.ndarray
    radius: float


def compute_obb(  # pylint:disable = too-many-locals
    vertices: np.ndarray,
    rotation_axis_hint: Optional[np.ndarray] = None,
) -> OBBResult:
    """Compute an oriented bounding box for an (N, 3) point cloud via PCA.

    Steps:
        1. PCA on the covariance matrix to find principal axes.
        2. Project points onto those axes to get half-extents.
        3. Re-center to the geometric center of the bounding box.
        4. Infer rotation axis from hint or circularity heuristic.

    Args:
        vertices: (N, 3) array of 3D positions.
        rotation_axis_hint: optional approximate rotation axis direction.
            If provided, the PCA axis most aligned with this hint is chosen.
            If None, the axis whose perpendicular cross-section is most circular is used.

    Returns:
        OBBResult with center, axes, extents, axis_of_rotation, and radius.
    """
    center = vertices.mean(axis=0)
    centered = vertices - center

    # PCA via eigendecomposition of covariance
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns ascending order; flip to descending (primary variance first)
    order = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, order]

    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    # Project onto principal axes to get half-extents
    projected = centered @ eigenvectors
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    extents = (maxs - mins) / 2.0

    # Re-center to geometric center of the OBB (not the centroid)
    obb_center = center + eigenvectors @ ((maxs + mins) / 2.0)

    # Axes as row vectors
    axes = eigenvectors.T

    # Derive rotation axis and radius
    rot_idx = _select_rotation_axis_index(axes, extents, rotation_axis_hint)
    rot_axis = axes[rot_idx].copy()
    perpendicular = [extents[j] for j in range(3) if j != rot_idx]
    radius = (perpendicular[0] + perpendicular[1]) / 2.0

    return OBBResult(
        center=obb_center,
        axes=axes,
        extents=extents,
        axis_of_rotation=rot_axis,
        radius=radius,
    )
