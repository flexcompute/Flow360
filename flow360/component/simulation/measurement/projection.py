"""Projected silhouette area measurement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from unyt import unyt_quantity

from flow360.exceptions import Flow360ValueError
from flow360.log import log

from .geometry import (
    SurfaceSelection,
    collect_surface_face_ids,
    load_transformed_surface_triangles,
    require_geometry_root,
    require_project_length_unit,
    resolve_surface_entities,
)

if TYPE_CHECKING:
    from flow360.component.simulation.draft_context.context import DraftContext

ProjectionDirection = Literal["X", "Y", "Z"]
RenderQuality = Literal["low", "medium", "high", "ultra"]
# Half space to keep, as a sign and an axis, bounded by that axis' zero plane. Internal:
# the only caller is submission-time trimming against the global Y=0 symmetry plane.
HalfSpace = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

_PROJECTION_AXES = {"X": (1, 2), "Y": (0, 2), "Z": (0, 1)}
_RENDER_RESOLUTIONS = {"low": 512, "medium": 1024, "high": 2048, "ultra": 4096}
_MINIMUM_RASTER_DIMENSION = 256
# Sample points per rasterization call. Bounds the broadcast temporaries so neither a huge
# mesh nor a raster-spanning triangle turns one call into a multi-gigabyte allocation.
_MAX_BATCH_SAMPLES = 1 << 20
# Footprint size, in pixels, up to which batching wins. Batching reaches the raster through
# gathers and scatters, which cost several times more per pixel than a plain slice; it pays
# only while the per-triangle Python overhead it removes outweighs that. Measured crossover
# is a few thousand pixels, and the curve is shallow either side of it.
_MAX_BATCHED_FOOTPRINT = 2048


def projected_area(
    draft: DraftContext,
    *,
    surfaces: SurfaceSelection,
    direction: ProjectionDirection = "X",
    render_quality: RenderQuality = "medium",
) -> unyt_quantity:
    """Compute the rasterized projected silhouette area of selected surfaces.

    Parameters
    ----------
    draft :
        Draft created from a Geometry resource; supplies the tessellation and any
        coordinate-system transforms applied to the selected surfaces.
    surfaces :
        Surfaces to project. Accepts a single ``Surface``, a single
        ``SurfaceSelector``, a list mixing the two, or an ``EntityList[Surface]``
        (for example ``my_wall.entities``, to reuse a boundary condition's
        assignment). Duplicates are removed automatically -- there is no need to
        deduplicate before calling.
    direction :
        Global axis to project along.
    render_quality :
        Raster resolution preset, and therefore how accurate the result is. The area
        is measured by counting whole pixels of a rasterized silhouette, so the error
        is proportional to pixel size: each step up halves it, at roughly four times
        the computation. Raise this when a small discrepancy matters.

    Returns
    -------
    unyt_quantity
        Projected area expressed in the geometry's project length unit squared.
        Overlapping surfaces are counted once -- the result is the silhouette
        union, not a sum of per-surface areas.

    Example
    -------
    >>> import flow360 as fl
    >>> geometry = fl.Geometry.from_cloud(id="...")
    >>> with fl.create_draft(new_run_from=geometry, face_grouping="faceId") as draft:
    ...     wheels = fl.SurfaceSelector(name="wheels").match("*rim*")
    ...     body = fl.SurfaceSelector(name="body").match("*body*")
    ...     area = fl.measure.projected_area(draft, surfaces=[body, wheels], direction="X")

    ====
    """
    return _projected_area(
        draft,
        surfaces=surfaces,
        direction=direction,
        render_quality=render_quality,
        clip=None,
    )


def _projected_area(
    draft: DraftContext,
    *,
    surfaces: SurfaceSelection,
    direction: ProjectionDirection,
    render_quality: RenderQuality,
    clip: HalfSpace | None,
) -> unyt_quantity:
    """Measure a projected area, optionally restricted to one half space.

    The half-space restriction stays private because it is derived from finalized
    simulation settings at submission time rather than chosen by hand: a user cannot know
    whether their geometry needs trimming without knowing whether the mesher will trim it.
    """
    operation = "projected_area()"
    if direction not in _PROJECTION_AXES:
        raise Flow360ValueError(
            f"{operation} direction must be one of X, Y, or Z; got {direction!r}."
        )
    if render_quality not in _RENDER_RESOLUTIONS:
        raise Flow360ValueError(
            f"{operation} render_quality must be one of low, medium, high, or ultra; "
            f"got {render_quality!r}."
        )

    geometry_root = require_geometry_root(draft, operation=operation)
    length_unit = require_project_length_unit(geometry_root, operation=operation)
    resolved_surfaces = resolve_surface_entities(draft, surfaces, operation=operation)
    surface_face_ids = collect_surface_face_ids(resolved_surfaces, operation=operation)
    triangles = load_transformed_surface_triangles(draft, geometry_root, surface_face_ids)

    if clip is not None:
        selected_count = len(triangles)
        triangles = _clip_to_half_space(triangles, clip)
        # Always report the effect, not just that a restriction was requested: equal counts
        # mean the geometry was already inside the half space and nothing was removed.
        log.info(
            f"Trimmed the selected tessellation at {clip[1]}=0, keeping the {clip} side: "
            f"triangle count {selected_count} -> {len(triangles)}."
        )
        if len(triangles) == 0:
            raise Flow360ValueError(
                f"None of the selected surfaces lie on the {clip} side of the {clip[1]}=0 "
                "plane, so the projected area would be zero. Check that the selected "
                "surfaces belong to the half of the model being meshed."
            )

    area = _compute_projected_area(
        triangles,
        direction=direction,
        base_resolution=_RENDER_RESOLUTIONS[render_quality],
    )
    return area * length_unit**2


def _compute_projected_area(  # pylint: disable=too-many-locals
    triangles: np.ndarray,
    *,
    direction: ProjectionDirection,
    base_resolution: int,
) -> float:
    """Project triangles and estimate their silhouette union with a boolean raster.

    Whether a pixel centre falls inside a triangle is independent per (triangle, pixel)
    pair and the results are combined with OR, so triangles are rasterized in vectorized
    batches instead of one per Python iteration. Batches are grouped by pixel-footprint
    size, which is what lets one broadcast cover many triangles at once.

    Measures exactly the triangles it is given. To restrict a measurement to part of the
    geometry, pass it through :func:`_clip_to_half_space` first; the raster is then sized
    to what survives.
    """
    horizontal_axis, vertical_axis = _PROJECTION_AXES[direction]
    # Select the two projected axes straight into their final layout: going through
    # `asarray(..., float64)[..., axes]` would materialize a full-size upcast of all three
    # axes first, which on automotive meshes is hundreds of megabytes of pure waste.
    projected = np.empty((len(triangles), 3, 2), dtype=np.float64)
    projected[..., 0] = triangles[..., horizontal_axis]
    projected[..., 1] = triangles[..., vertical_axis]

    bounds_minimum = projected.min(axis=(0, 1))
    bounds_maximum = projected.max(axis=(0, 1))
    projected -= bounds_minimum
    projected_width, projected_height = bounds_maximum - bounds_minimum

    if projected_width <= 0 or projected_height <= 0:
        return 0.0

    raster_width, raster_height = _raster_dimensions(
        projected_width, projected_height, base_resolution
    )
    pixel_width = projected_width / raster_width
    pixel_height = projected_height / raster_height
    occupancy = np.zeros((raster_height, raster_width), dtype=bool)
    edge_tolerance = (
        np.finfo(np.float64).eps * max(projected_width, projected_height, 1.0) ** 2 * 32
    )

    corners = [(projected[:, index, 0], projected[:, index, 1]) for index in range(3)]
    twice_area = _edge_function(corners[0], corners[1], corners[2][0], corners[2][1])
    column_start, column_stop = _pixel_intervals(
        projected[:, :, 0].min(axis=1), projected[:, :, 0].max(axis=1), pixel_width, raster_width
    )
    row_start, row_stop = _pixel_intervals(
        projected[:, :, 1].min(axis=1), projected[:, :, 1].max(axis=1), pixel_height, raster_height
    )

    footprint_widths = column_stop - column_start
    footprint_heights = row_stop - row_start
    contributing = np.flatnonzero(
        (np.abs(twice_area) > edge_tolerance) & (footprint_widths > 0) & (footprint_heights > 0)
    )
    orientation = np.copysign(1.0, twice_area)

    for group in _footprint_groups(contributing, footprint_heights, footprint_widths, raster_width):
        footprint_height = int(footprint_heights[group[0]])
        footprint_width = int(footprint_widths[group[0]])
        # A footprint too tall to fit the sample cap on its own is split into row bands, so
        # no single evaluation allocates more than the cap regardless of triangle size.
        band_height = max(1, min(footprint_height, _MAX_BATCH_SAMPLES // footprint_width))

        if footprint_height * footprint_width <= _MAX_BATCHED_FOOTPRINT:
            mark = _mark_covered_batch
            batch_size = max(1, _MAX_BATCH_SAMPLES // (band_height * footprint_width))
        else:
            mark = _mark_covered_window
            batch_size = 1

        for batch_start in range(0, len(group), batch_size):
            batch = group[batch_start : batch_start + batch_size]
            vertices = projected[batch]
            batch_columns = column_start[batch]
            batch_orientation = orientation[batch]
            for band_start in range(0, footprint_height, band_height):
                mark(
                    occupancy,
                    vertices,
                    row_start[batch] + band_start,
                    batch_columns,
                    batch_orientation,
                    footprint=(
                        min(band_height, footprint_height - band_start),
                        footprint_width,
                    ),
                    pixel_width=pixel_width,
                    pixel_height=pixel_height,
                    edge_tolerance=edge_tolerance,
                )

    return float(np.count_nonzero(occupancy) * pixel_width * pixel_height)


def _plane_crossing(
    inside_point: np.ndarray,
    inside_depth: np.ndarray,
    outside_point: np.ndarray,
    outside_depth: np.ndarray,
) -> np.ndarray:
    """Return where each inside-to-outside segment crosses the plane.

    ``inside_depth`` is non-negative and ``outside_depth`` negative, so the denominator is
    always positive and the fraction lands in [0, 1).
    """
    fraction = inside_depth / (inside_depth - outside_depth)
    return inside_point + fraction[:, None] * (outside_point - inside_point)


def _clip_to_half_space(triangles: np.ndarray, half_space: HalfSpace) -> np.ndarray:
    """Return the part of `triangles` on the kept side of the half space's zero plane.

    Triangles crossing the plane are cut exactly, not kept or dropped whole. The intended
    plane is a symmetry plane, and a surface mesh crossing one carries a full line of
    triangles there; deciding those whole either way would bias the area by roughly a mesh
    cell along the entire cut.

    Only fragments with area survive. A triangle merely touching the plane from the
    discarded side cuts to a zero-area sliver, so keeping those would make the returned
    count claim retained geometry where there is none.
    """
    axis = "XYZ".index(half_space[1])
    sign = 1.0 if half_space[0] == "+" else -1.0
    # Non-negative depth means the vertex is in the half space being kept.
    depth = sign * triangles[:, :, axis].astype(np.float64)
    inside = depth >= 0.0
    inside_count = inside.sum(axis=1)

    kept = [triangles[inside_count == 3]]

    crossing_single = np.flatnonzero(inside_count == 1)
    if len(crossing_single) > 0:
        # One vertex survives, so the cut leaves a single smaller triangle.
        corner, corner_depth = _cyclic_corners(
            triangles, depth, crossing_single, np.argmax(inside[crossing_single], axis=1)
        )
        kept.append(
            np.stack(
                [
                    corner[0],
                    _plane_crossing(corner[0], corner_depth[0], corner[1], corner_depth[1]),
                    _plane_crossing(corner[0], corner_depth[0], corner[2], corner_depth[2]),
                ],
                axis=1,
            )
        )

    crossing_double = np.flatnonzero(inside_count == 2)
    if len(crossing_double) > 0:
        # One vertex is cut away, leaving a quadrilateral; fan it from the first survivor.
        # Rotating the outside vertex to the front puts the two survivors at 1 and 2, so
        # the quadrilateral in cyclic order is (1, 2, crossing on 2->0, crossing on 0->1).
        corner, corner_depth = _cyclic_corners(
            triangles, depth, crossing_double, np.argmin(inside[crossing_double], axis=1)
        )
        beyond_second = _plane_crossing(corner[2], corner_depth[2], corner[0], corner_depth[0])
        before_first = _plane_crossing(corner[1], corner_depth[1], corner[0], corner_depth[0])
        kept.append(np.stack([corner[1], corner[2], beyond_second], axis=1))
        kept.append(np.stack([corner[1], beyond_second, before_first], axis=1))

    return _with_area(np.concatenate(kept))


def _with_area(triangles: np.ndarray) -> np.ndarray:
    """Drop triangles whose three vertices are collinear, so contribute no area."""
    first_edge = triangles[:, 1] - triangles[:, 0]
    second_edge = triangles[:, 2] - triangles[:, 0]
    return triangles[np.any(np.cross(first_edge, second_edge) != 0, axis=1)]


def _cyclic_corners(
    triangles: np.ndarray, depth: np.ndarray, selected: np.ndarray, first: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Rotate each selected triangle so `first` leads, preserving winding order.

    Returns the three corners and their depths, so callers can name vertices by their
    position relative to the one that decides how the triangle is cut.
    """
    rotated = triangles[selected]
    rotated_depth = depth[selected]
    rows = np.arange(len(selected))
    corners = [rotated[rows, (first + offset) % 3] for offset in range(3)]
    depths = [rotated_depth[rows, (first + offset) % 3] for offset in range(3)]
    return corners, depths


def _footprint_groups(
    contributing: np.ndarray,
    footprint_heights: np.ndarray,
    footprint_widths: np.ndarray,
    raster_width: int,
) -> list[np.ndarray]:
    """Group triangle indices by identical pixel footprint, largest footprint first.

    Grouping is on the exact footprint rather than a rounded one so a batch needs no
    padding: every triangle in a group covers the same window shape, already clamped
    inside the raster. Largest first because those fill the raster fastest, which lets
    ``_mark_covered_pixels`` discard the many small triangles that land on ground already
    covered -- an automotive silhouette stacks body, underbody, suspension and hidden
    wheels along the same view direction.

    Every returned group is non-empty, so callers can read a group's footprint off its
    first member: ``np.split`` yields one empty group for empty input, which is dropped.
    """
    heights = footprint_heights[contributing]
    widths = footprint_widths[contributing]
    footprint_key = heights * (raster_width + 1) + widths
    order = np.lexsort((footprint_key, -(heights * widths)))

    grouped = contributing[order]
    sorted_key = footprint_key[order]
    groups = np.split(grouped, np.flatnonzero(np.diff(sorted_key)) + 1)
    return [group for group in groups if len(group) > 0]


def _covered_mask(
    corners: list[tuple[np.ndarray, np.ndarray]],
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    orientation: np.ndarray,
    edge_tolerance: float,
) -> np.ndarray:
    """Return which sample points lie inside the triangles described by `corners`.

    Shapes are the caller's business: pass scalars per corner for one triangle, or
    leading-axis arrays to test a whole batch at once. Both broadcast against the sample
    grids the same way, which keeps one copy of the containment arithmetic.
    """
    covered = (
        orientation * _edge_function(corners[0], corners[1], sample_x, sample_y) >= -edge_tolerance
    )
    covered &= (
        orientation * _edge_function(corners[1], corners[2], sample_x, sample_y) >= -edge_tolerance
    )
    covered &= (
        orientation * _edge_function(corners[2], corners[0], sample_x, sample_y) >= -edge_tolerance
    )
    return covered


def _mark_covered_window(  # pylint: disable=too-many-arguments,too-many-locals
    occupancy: np.ndarray,
    vertices: np.ndarray,
    row_start: np.ndarray,
    column_start: np.ndarray,
    orientation: np.ndarray,
    *,
    footprint: tuple[int, int],
    pixel_width: float,
    pixel_height: float,
    edge_tolerance: float,
) -> None:
    """Rasterize a single triangle into a contiguous raster window.

    Used for footprints too large to batch, where slicing beats the gather and scatter
    that make batching worthwhile for small ones.
    """
    footprint_height, footprint_width = footprint
    first_row, first_column = int(row_start[0]), int(column_start[0])
    window = occupancy[
        first_row : first_row + footprint_height,
        first_column : first_column + footprint_width,
    ]
    if window.all():
        return

    sample_x = ((first_column + np.arange(footprint_width) + 0.5) * pixel_width)[None, :]
    sample_y = ((first_row + np.arange(footprint_height) + 0.5) * pixel_height)[:, None]
    corners = [(vertices[0, index, 0], vertices[0, index, 1]) for index in range(3)]

    window |= _covered_mask(corners, sample_x, sample_y, orientation[0], edge_tolerance)


def _mark_covered_batch(  # pylint: disable=too-many-arguments,too-many-locals
    occupancy: np.ndarray,
    vertices: np.ndarray,
    row_start: np.ndarray,
    column_start: np.ndarray,
    orientation: np.ndarray,
    *,
    footprint: tuple[int, int],
    pixel_width: float,
    pixel_height: float,
    edge_tolerance: float,
) -> None:
    """Set every pixel whose centre lies inside one of a batch of same-footprint triangles.

    ``vertices`` is ``(batch, 3, 2)`` in raster coordinates. Writes only ever set True, so
    the scatter tolerates the duplicate indices that overlapping triangles produce.
    """
    footprint_height, footprint_width = footprint
    rows = row_start[:, None] + np.arange(footprint_height)
    columns = column_start[:, None] + np.arange(footprint_width)

    # A triangle whose whole footprint is already covered can only set pixels that are
    # set, so drop it before evaluating any edge function.
    pending = ~occupancy[rows[:, :, None], columns[:, None, :]].all(axis=(1, 2))
    if not pending.any():
        return
    if not pending.all():
        vertices, rows, columns = vertices[pending], rows[pending], columns[pending]
        orientation = orientation[pending]

    sample_x = ((columns + 0.5) * pixel_width)[:, None, :]
    sample_y = ((rows + 0.5) * pixel_height)[:, :, None]
    corners = [
        (vertices[:, index, 0][:, None, None], vertices[:, index, 1][:, None, None])
        for index in range(3)
    ]

    covered = _covered_mask(corners, sample_x, sample_y, orientation[:, None, None], edge_tolerance)
    batch_index, row_index, column_index = np.nonzero(covered)
    occupancy[rows[batch_index, row_index], columns[batch_index, column_index]] = True


def _raster_dimensions(width: float, height: float, base_resolution: int) -> tuple[int, int]:
    """Match the Web user interface's quality and aspect-ratio raster sizing."""
    if width >= height:
        raster_width = base_resolution
        raster_height = int(np.floor(base_resolution * height / width + 0.5))
    else:
        raster_height = base_resolution
        raster_width = int(np.floor(base_resolution * width / height + 0.5))
    return (
        max(_MINIMUM_RASTER_DIMENSION, raster_width),
        max(_MINIMUM_RASTER_DIMENSION, raster_height),
    )


def _pixel_intervals(
    minimum: np.ndarray, maximum: np.ndarray, pixel_size: float, pixel_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-triangle half-open pixel-centre intervals intersecting each bound."""
    start = np.maximum(np.ceil(minimum / pixel_size - 0.5).astype(np.int64), 0)
    stop = np.minimum(np.floor(maximum / pixel_size - 0.5).astype(np.int64) + 1, pixel_count)
    return start, stop


def _edge_function(
    start: np.ndarray, end: np.ndarray, sample_x: np.ndarray | float, sample_y: np.ndarray | float
) -> np.ndarray | float:
    """Return the signed two-dimensional edge function at sample positions."""
    return (end[0] - start[0]) * (sample_y - start[1]) - (end[1] - start[1]) * (sample_x - start[0])
