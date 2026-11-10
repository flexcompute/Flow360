"""Oriented Bounding Box computation via PCA.

Computes an OBB from an (N, 3) vertex point cloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import index
from typing import TYPE_CHECKING, List, Optional, SupportsIndex

import numpy as np
from unyt import unyt_quantity

from flow360.exceptions import Flow360ValueError
from flow360.log import log

from .geometry import (
    SurfaceSelection,
    collect_surface_face_ids,
    load_transformed_surface_vertices,
    require_geometry_root,
    require_project_length_unit,
    resolve_surface_entities,
)

if TYPE_CHECKING:
    from flow360.component.simulation.draft_context.context import DraftContext


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
        if np.linalg.norm(hint) <= np.finfo(float).eps:
            raise Flow360ValueError("rotation_axis_hint must be a non-zero vector.")
        dots = np.abs(axes @ hint)
        return int(np.argmax(dots))

    # Circularity heuristic: for each axis, ratio of the two perpendicular extents
    best_index = 0
    best_ratio = -1.0
    for i in range(3):
        others: List[float] = [extents[j] for j in range(3) if j != i]
        larger = max(others[0], others[1])
        ratio = min(others[0], others[1]) / larger if larger > 0 else 1.0
        if ratio > best_ratio:
            best_ratio = ratio
            best_index = i
    return best_index


@dataclass(frozen=True)
class RotationAxisAndRadius:
    """Rotation axis and averaged radius derived from an oriented bounding box.

    Attributes:
        axis_of_rotation (numpy.ndarray): Unit vector along the selected rotation axis,
            shape ``(3,)``. Always dimensionless.
        averaged_radius (unyt.unyt_quantity): Estimated cylinder radius, averaged from the
            two OBB half-extents perpendicular to the rotation axis. Carries the geometry's
            project length unit, inherited from the :class:`OBBResult` it was derived from.
    """

    axis_of_rotation: np.ndarray
    averaged_radius: unyt_quantity
    _averaged_radius_formula: str = field(
        default="(perpendicular_extent_1 + perpendicular_extent_2) / 2",
        repr=False,
        compare=False,
    )

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"axis_of_rotation={self.axis_of_rotation}, "
            f"averaged_radius={self.averaged_radius} "
            f"(calculated as {self._averaged_radius_formula})"
            ")"
        )

    __repr__ = __str__


@dataclass(frozen=True)
class OBBResult:
    """Oriented Bounding Box computed from a point cloud.

    OBBResult stores bounding-box geometry only. Use
    :meth:`get_rotation_axis_and_radius` when a cylinder-like rotation axis and
    averaged radius are needed.

    Attributes:
        center (numpy.ndarray): Geometric center of the OBB, shape ``(3,)``. Carries the
            geometry's project length unit when produced by :func:`oriented_bounding_box`.
        axes (numpy.ndarray): Principal axes as row vectors of shape ``(3, 3)``,
            ordered by descending extent magnitude. Always dimensionless.
        extents (numpy.ndarray): Half-extents along each axis, shape ``(3,)``. Carries the
            geometry's project length unit when produced by :func:`oriented_bounding_box`.
    """

    center: np.ndarray
    axes: np.ndarray
    extents: np.ndarray

    def get_rotation_axis_and_radius(
        self,
        axis_index: Optional[SupportsIndex] = None,
        rotation_axis_hint: Optional[np.ndarray] = None,
    ) -> RotationAxisAndRadius:
        """Derive a rotation axis and averaged radius from this oriented bounding box.

        Args:
            axis_index: Principal axis index to use as the rotation axis. Valid
                values are 0, 1, and 2.
            rotation_axis_hint: Approximate rotation-axis direction. The
                principal axis most aligned with this hint is selected.

        Returns:
            RotationAxisAndRadius with the selected axis and averaged
            perpendicular half-extent radius.

        Raises:
            Flow360ValueError: If both selection inputs are provided, if
                axis_index is invalid, or if rotation_axis_hint is zero.

        If neither axis_index nor rotation_axis_hint is provided, the rotation
        axis and averaged radius are guessed from the principal axis whose
        perpendicular bounding-box extents are most similar. A warning is
        emitted in that case because this is an inference, not a known
        geometric property.
        """
        if axis_index is not None and rotation_axis_hint is not None:
            raise Flow360ValueError("axis_index and rotation_axis_hint cannot both be provided.")

        if axis_index is not None:
            try:
                rotation_axis_index = index(axis_index)
            except TypeError as exc:
                raise Flow360ValueError("axis_index must be one of 0, 1, or 2.") from exc
            if rotation_axis_index not in range(3):
                raise Flow360ValueError("axis_index must be one of 0, 1, or 2.")
        else:
            if rotation_axis_hint is None:
                log.warning(
                    "No axis_index or rotation_axis_hint was provided. "
                    "The rotation axis and averaged radius are guessed by selecting the "
                    "principal axis whose perpendicular bounding-box extents are most similar."
                )
            rotation_axis_index = _select_rotation_axis_index(
                self.axes, self.extents, rotation_axis_hint
            )

        perpendicular = [self.extents[j] for j in range(3) if j != rotation_axis_index]
        averaged_radius = (perpendicular[0] + perpendicular[1]) / 2.0
        return RotationAxisAndRadius(
            axis_of_rotation=self.axes[rotation_axis_index].copy(),
            averaged_radius=averaged_radius,
            _averaged_radius_formula=f"({perpendicular[0]} + {perpendicular[1]}) / 2",
        )


def _compute_oriented_bounding_box(
    vertices: np.ndarray,
) -> OBBResult:  # pylint:disable = too-many-locals
    """Compute an oriented bounding box for an (N, 3) point cloud via PCA.

    Steps:
        1. PCA on the covariance matrix to find principal axes.
        2. Project points onto those axes to get half-extents.
        3. Re-center to the geometric center of the bounding box.

    Args:
        vertices: (N, 3) array of 3D positions.

    Returns:
        OBBResult with center, axes, and extents. Use
        OBBResult.get_rotation_axis_and_radius() to derive cylinder-like
        rotation metadata when needed.
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

    return OBBResult(
        center=obb_center,
        axes=axes,
        extents=extents,
    )


def oriented_bounding_box(
    draft: DraftContext,
    *,
    surfaces: SurfaceSelection,
) -> OBBResult:
    """Compute a PCA-oriented bounding box for selected geometry surfaces.

    Parameters
    ----------
    draft :
        Draft created from a Geometry resource.
    surfaces :
        Surfaces to bound. Accepts a single ``Surface``, a single ``SurfaceSelector``,
        a list mixing the two, or an ``EntityList[Surface]``. Duplicates are removed
        automatically.

    Returns
    -------
    OBBResult
        ``center`` and ``extents`` are expressed in the geometry's project length unit;
        ``axes`` stays dimensionless.

    Example
    -------
    >>> import flow360 as fl
    >>> geometry = fl.Geometry.from_cloud(id="...")
    >>> with fl.create_draft(new_run_from=geometry, face_grouping="faceId") as draft:
    ...     wheel = fl.SurfaceSelector(name="wheel_FL").match("*rim*FL*")
    ...     obb = fl.measure.oriented_bounding_box(draft, surfaces=wheel)
    ...     rotation = obb.get_rotation_axis_and_radius(rotation_axis_hint=(0, 1, 0))

    ====
    """
    operation = "oriented_bounding_box()"
    geometry_root = require_geometry_root(draft, operation=operation)
    length_unit = require_project_length_unit(geometry_root, operation=operation)
    resolved_surfaces = resolve_surface_entities(draft, surfaces, operation=operation)
    surface_face_ids = collect_surface_face_ids(resolved_surfaces, operation=operation)

    log.info("Computing Oriented Bounding Box (OBB)...")
    vertices = load_transformed_surface_vertices(draft, geometry_root, surface_face_ids)
    log.info(f"OBB: extracted {len(vertices)} vertices, computing PCA...")
    result = _compute_oriented_bounding_box(vertices)

    log.info("OBB computation complete.")
    return OBBResult(
        center=result.center * length_unit,
        axes=result.axes,
        extents=result.extents * length_unit,
    )
