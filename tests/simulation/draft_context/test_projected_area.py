"""Tests for projected silhouette area measurement."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import unyt
from flow360_schema.models.entities.surface_entities import Surface
from flow360_schema.models.reference_geometry import ProjectedArea, ReferenceGeometry
from flow360_schema.models.simulation.simulation_params import SimulationParams

import flow360 as fl
from flow360.component.project_utils import _compute_projected_reference_area
from flow360.component.simulation.measurement.projection import (
    _RENDER_RESOLUTIONS,
    _clip_to_half_space,
    _compute_projected_area,
    _projected_area,
    _raster_dimensions,
)
from flow360.exceptions import Flow360RuntimeError, Flow360ValueError


def _rectangle_triangles(width: float = 2.0, height: float = 1.0) -> np.ndarray:
    """Return two triangles covering an axis-aligned rectangle in the XY plane."""
    return np.array(
        [
            [[0, 0, 0], [width, 0, 0], [width, height, 0]],
            [[0, 0, 0], [width, height, 0], [0, height, 0]],
        ],
        dtype=np.float64,
    )


def test_axis_aligned_rectangle_area():
    area = _compute_projected_area(_rectangle_triangles(), direction="Z", base_resolution=512)

    assert area == pytest.approx(2.0)


def test_rotated_rectangle_area():
    triangles = _rectangle_triangles()
    angle = np.deg2rad(37)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    triangles = triangles @ rotation.T

    area = _compute_projected_area(triangles, direction="Z", base_resolution=512)

    assert area == pytest.approx(2.0, rel=0.01)


def test_overlapping_triangles_are_counted_once():
    triangles = _rectangle_triangles()
    duplicated = np.concatenate([triangles, triangles], axis=0)

    area = _compute_projected_area(duplicated, direction="Z", base_resolution=512)

    assert area == pytest.approx(2.0)


def test_edge_on_rectangle_has_zero_area():
    area = _compute_projected_area(_rectangle_triangles(), direction="X", base_resolution=512)

    assert area == 0.0


def _sphere_shell(count: int, *, seed: int = 20260806) -> np.ndarray:
    """Triangles scattered on a sphere: overlapping footprints of many different sizes."""
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3, 3))
    points /= np.linalg.norm(points, axis=2, keepdims=True)
    return (points * 0.5).astype(np.float32)


def _tessellated_sphere(subdivisions: int, radius: float = 0.5) -> np.ndarray:
    """A UV-sphere tessellation, standing in for a real surface mesh.

    Unlike randomly scattered triangles, this has the property that matters for the
    rasterizer: many small triangles, each covering a handful of pixels, with front and
    back faces overlapping along the view direction. Silhouette area is exactly a disc.
    """
    theta, phi = np.meshgrid(
        np.linspace(0.0, np.pi, subdivisions + 1),
        np.linspace(0.0, 2.0 * np.pi, 2 * subdivisions + 1),
        indexing="ij",
    )
    points = np.stack(
        [
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ],
        axis=-1,
    )
    corner = (points[:-1, :-1], points[:-1, 1:], points[1:, 1:], points[1:, :-1])
    quads = [
        np.stack([corner[0], corner[1], corner[2]], axis=-2),
        np.stack([corner[0], corner[2], corner[3]], axis=-2),
    ]
    return np.concatenate([quad.reshape(-1, 3, 3) for quad in quads]).astype(np.float32)


def test_area_is_invariant_to_triangle_order():
    """Triangles are rasterized in batches grouped by footprint size, so input order
    decides which triangles share a batch. A scatter that dropped overlapping writes
    within a batch would show up here as an order-dependent area."""
    triangles = _sphere_shell(400)
    shuffled = triangles[np.random.default_rng(7).permutation(len(triangles))]

    assert _compute_projected_area(
        triangles, direction="Z", base_resolution=512
    ) == _compute_projected_area(shuffled, direction="Z", base_resolution=512)


def test_large_and_tiny_footprints_mix():
    """A raster-spanning triangle batches alone; sub-pixel ones batch in bulk. Both must
    reach the raster, so triangles placed outside the big one have to add area."""
    spanning = np.array([[[-1, -1, 0], [1, -1, 0], [0, 1, 0]]], dtype=np.float32)
    # Near the top-left corner, which the spanning triangle's apex leaves uncovered.
    tiny = _sphere_shell(200) * 0.01 + np.array([-0.9, 0.9, 0.0], dtype=np.float32)

    spanning_only = _compute_projected_area(spanning, direction="Z", base_resolution=512)
    together = _compute_projected_area(
        np.concatenate([spanning, tiny], axis=0), direction="Z", base_resolution=512
    )

    assert together > spanning_only


def test_degenerate_triangles_are_ignored():
    """Collapsed triangles must not reach the raster; the tolerance test drops them."""
    triangles = _rectangle_triangles()
    collapsed = np.array([[[0, 0, 0], [2, 1, 0], [2, 1, 0]]], dtype=np.float64)

    area = _compute_projected_area(
        np.concatenate([triangles, collapsed], axis=0), direction="Z", base_resolution=512
    )

    assert area == pytest.approx(2.0)


def test_dense_mesh_rasterizes_within_time_budget():
    """Guards the batching. Cost here is dominated by per-triangle overhead rather than
    by raster size, which is what the previous one-triangle-per-iteration loop paid in
    full; the budget is loose enough for a loaded machine but far below that."""
    import time

    triangles = _tessellated_sphere(224)
    assert len(triangles) > 200_000

    start = time.perf_counter()
    area = _compute_projected_area(triangles, direction="Z", base_resolution=1024)
    elapsed = time.perf_counter() - start

    assert area == pytest.approx(np.pi * 0.25, rel=0.01)
    assert elapsed < 10.0, f"{len(triangles)} triangles took {elapsed:.1f}s"


def _straddling_triangle() -> np.ndarray:
    """One triangle crossing Y=0, cut into a trapezoid of area 0.75 and a tip of 0.25.

    Vertices (0,-1), (1,-1), (0,1) in the XY plane: total area 1. Keeping y<=0 leaves a
    trapezoid with parallel sides of 1 and 0.5 over a height of 1; keeping y>=0 leaves a
    triangle with base 0.5 and height 1. The two exercise both crossing cases -- two
    vertices surviving, and one.
    """
    return np.array([[[0, -1, 0], [1, -1, 0], [0, 1, 0]]], dtype=np.float64)


def _clipped_area(triangles, half_space, *, base_resolution=1024):
    """Projected area of the part of `triangles` inside `half_space`, along Z."""
    return _compute_projected_area(
        _clip_to_half_space(triangles, half_space),
        direction="Z",
        base_resolution=base_resolution,
    )


def test_clip_cuts_straddling_triangles_exactly():
    """Both crossing cases must cut, not round the triangle to one side."""
    triangle = _straddling_triangle()

    assert _clipped_area(triangle, "-Y", base_resolution=512) == pytest.approx(0.75, rel=5e-3)
    assert _clipped_area(triangle, "+Y", base_resolution=512) == pytest.approx(0.25, rel=5e-3)


def test_clip_drops_triangles_that_only_touch_the_plane():
    """A triangle on the discarded side with a vertex exactly on the plane cuts to a
    zero-area sliver. Keeping it would let the retained count claim geometry that
    contributes nothing, so the emptiness check downstream could not be trusted."""
    touching = np.array(
        [
            [[0, 0, 0], [1, 1, 0], [2, 1, 0]],
            [[0, 0, 1], [1, 2, 1], [2, 2, 1]],
        ],
        dtype=np.float64,
    )

    assert len(_clip_to_half_space(touching, "-Y")) == 0
    assert len(_clip_to_half_space(touching, "+Y")) == 2


def test_all_degenerate_triangles_over_a_two_dimensional_span_measure_zero():
    """Two slivers, each collapsed but pointing along different axes, span a 2D box while
    contributing nothing. Nothing reaches the raster, and reporting zero must not depend
    on the projected bounds having collapsed too."""
    slivers = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 2, 0]],
        ],
        dtype=np.float64,
    )

    assert _compute_projected_area(slivers, direction="Z", base_resolution=512) == 0.0


def test_complementary_clips_partition_the_silhouette():
    """Projected along Z, the two Y half spaces land in disjoint parts of the raster, so
    their areas have to add back up to the unclipped one."""
    triangles = _tessellated_sphere(48)

    unclipped = _compute_projected_area(triangles, direction="Z", base_resolution=1024)
    negative = _clipped_area(triangles, "-Y")
    positive = _clipped_area(triangles, "+Y")

    assert negative + positive == pytest.approx(unclipped, rel=2e-3)
    assert negative == pytest.approx(positive, rel=2e-3)


def test_clip_halves_a_symmetric_mesh():
    """A sphere is symmetric about Y=0, so keeping one side halves the silhouette."""
    assert _clipped_area(_tessellated_sphere(48), "-Y") == pytest.approx(np.pi * 0.25 / 2, rel=5e-3)


def test_clip_is_a_no_op_on_geometry_already_inside_the_half_space():
    """Why submission trims rather than scaling by a factor: on an already-half mesh the
    correct correction is none, and clipping gets that for free."""
    half = _tessellated_sphere(48)
    half = half[np.all(half[:, :, 1] <= 0.0, axis=1)]

    unclipped = _compute_projected_area(half, direction="Z", base_resolution=1024)

    assert _clipped_area(half, "-Y") == unclipped


def test_clip_on_the_projection_axis_selects_facing_geometry():
    """Clipping is a half-space filter in three dimensions, so it also works on the axis
    being projected along -- there it keeps the near or far side."""
    assert _clipped_area(_tessellated_sphere(48), "-Z") == pytest.approx(np.pi * 0.25, rel=1e-2)


def test_clip_can_remove_every_triangle():
    """Clipping reports an empty result rather than deciding it is fatal; the caller that
    asked for the restriction owns that call, and can say why it was applied."""
    triangles = _tessellated_sphere(24)
    positive_y_only = triangles[np.all(triangles[:, :, 1] > 0.01, axis=1)]

    assert len(_clip_to_half_space(positive_y_only, "-Y")) == 0


@pytest.mark.parametrize(
    ("quality", "resolution"),
    [("low", 512), ("medium", 1024), ("high", 2048), ("ultra", 4096)],
)
def test_render_quality_sets_long_raster_dimension(quality, resolution):
    raster_width, raster_height = _raster_dimensions(3.0, 1.0, _RENDER_RESOLUTIONS[quality])

    assert raster_width == resolution
    assert raster_height == max(256, int(np.floor(resolution / 3 + 0.5)))


@pytest.mark.parametrize(
    ("direction", "triangles"),
    [
        ("X", _rectangle_triangles()[..., [2, 0, 1]]),
        ("Y", _rectangle_triangles()[..., [0, 2, 1]]),
        ("Z", _rectangle_triangles()),
    ],
)
def test_projection_directions(direction, triangles):
    area = _compute_projected_area(triangles, direction=direction, base_resolution=512)

    assert area == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("direction", "x", "direction must be one of X, Y, or Z"),
        ("render_quality", "maximum", "render_quality must be one of"),
    ],
)
def test_public_facade_rejects_invalid_settings(keyword, value, message):
    arguments = {
        "draft": MagicMock(),
        "surfaces": Surface(
            name="surface",
            private_attribute_id="surface-id",
        ),
        keyword: value,
    }

    with pytest.raises(Flow360ValueError, match=message):
        fl.measure.projected_area(**arguments)


def test_public_facade_requires_geometry_root():
    draft = MagicMock()
    draft._geometry_root = None

    with pytest.raises(Flow360RuntimeError, match="projected_area.*requires.*Geometry"):
        fl.measure.projected_area(
            draft,
            surfaces=Surface(
                name="surface",
                private_attribute_id="surface-id",
                private_attribute_sub_components=["face-id"],
            ),
        )


def test_public_facade_rejects_surface_without_face_identifiers():
    draft = MagicMock()
    draft._geometry_root = MagicMock()

    with pytest.raises(Flow360ValueError, match="tessellation face identifiers"):
        fl.measure.projected_area(
            draft,
            surfaces=Surface(
                name="surface",
                private_attribute_id="surface-id",
            ),
        )


def test_submission_resolver_updates_computed_result_and_logs(monkeypatch):
    surface = Surface(
        name="surface",
        private_attribute_id="surface-id",
        private_attribute_sub_components=["face-id"],
    )
    with fl.SI_unit_system:
        params = SimulationParams(
            reference_geometry=ReferenceGeometry(
                area=ProjectedArea(
                    surfaces=[surface],
                    direction="Y",
                    render_quality="high",
                )
            )
        )
    active_draft = MagicMock()
    compute = MagicMock(return_value=3.25 * unyt.m**2)
    info = MagicMock()
    monkeypatch.setattr("flow360.component.project_utils._projected_area", compute)
    monkeypatch.setattr("flow360.component.project_utils.log.info", info)

    _compute_projected_reference_area(params, active_draft)

    area = params.reference_geometry.area
    assert isinstance(area, ProjectedArea)
    assert area.computed is not None
    assert area.computed.to_value("m**2") == pytest.approx(3.25)
    serialized_area = params.model_dump(mode="json", exclude_none=True)["reference_geometry"][
        "area"
    ]
    assert serialized_area["type_name"] == "projected_area"
    assert serialized_area["computed"] == {"value": 3.25}
    assert serialized_area["direction"] == "Y"
    assert serialized_area["render_quality"] == "high"
    assert serialized_area["surfaces"]["stored_entities"] == [
        {"type": "Surface", "ids": ["surface-id"]}
    ]
    compute.assert_called_once_with(
        active_draft,
        surfaces=area.surfaces,
        direction="Y",
        render_quality="high",
        clip=None,
    )
    info.assert_called_once_with("Automatically computed projected reference area: 3.25 m**2.")


def test_submission_resolver_requires_active_draft():
    with fl.SI_unit_system:
        params = SimulationParams(
            reference_geometry=ReferenceGeometry(
                area=ProjectedArea(
                    surfaces=[
                        Surface(
                            name="surface",
                            private_attribute_id="surface-id",
                            private_attribute_sub_components=["face-id"],
                        )
                    ]
                )
            )
        )

    with pytest.raises(Flow360RuntimeError, match="requires an active draft"):
        _compute_projected_reference_area(params, None)


def _draft_returning(monkeypatch, triangles):
    """A draft whose tessellation load yields `triangles`, skipping any cloud access."""
    draft = MagicMock()
    draft._geometry_root._project_length_unit = 1.0 * unyt.m
    monkeypatch.setattr(
        "flow360.component.simulation.measurement.projection" ".load_transformed_surface_triangles",
        lambda *_args, **_kwargs: triangles,
    )
    return draft


def _measure_clipped(draft, half_space):
    return _projected_area(
        draft,
        surfaces=Surface(
            name="surface",
            private_attribute_id="surface-id",
            private_attribute_sub_components=["face-id"],
        ),
        direction="Z",
        render_quality="low",
        clip=half_space,
    )


def test_trimming_reports_its_effect_on_the_tessellation(monkeypatch):
    """Report the effect, not just the intent: an unchanged count is how a user sees that
    the geometry was already half, the case where a hand-written 0.5 factor is wrong."""
    draft = _draft_returning(monkeypatch, _straddling_triangle())
    messages = []
    monkeypatch.setattr(
        "flow360.component.simulation.measurement.projection.log.info", messages.append
    )

    _measure_clipped(draft, "-Y")

    assert any(
        "keeping the -Y side: triangle count 1 -> 2" in message for message in messages
    ), messages


def test_trimming_away_every_selected_surface_raises(monkeypatch):
    """A zero reference area would silently poison every force coefficient."""
    draft = _draft_returning(
        monkeypatch, _straddling_triangle() + np.array([0.0, 5.0, 0.0], dtype=np.float64)
    )

    with pytest.raises(Flow360ValueError, match="None of the selected surfaces lie on the -Y"):
        _measure_clipped(draft, "-Y")


def _params_with_farfield(farfield):
    """Projected-area recipe plus one farfield volume zone."""
    with fl.SI_unit_system:
        return SimulationParams(
            meshing=fl.MeshingParams(volume_zones=[farfield]),
            reference_geometry=ReferenceGeometry(
                area=ProjectedArea(
                    surfaces=[
                        Surface(
                            name="surface",
                            private_attribute_id="surface-id",
                            private_attribute_sub_components=["face-id"],
                        )
                    ]
                )
            ),
        )


@pytest.mark.parametrize(
    "farfield_type", [fl.UserDefinedFarfield, fl.AutomatedFarfield], ids=["user", "automated"]
)
@pytest.mark.parametrize(
    ("domain_type", "expected_clip"),
    [
        ("half_body_negative_y", "-Y"),
        ("half_body_positive_y", "+Y"),
        ("full_body", None),
        (None, None),
    ],
)
def test_submission_resolver_derives_clip_from_domain_type(
    monkeypatch, farfield_type, domain_type, expected_clip
):
    """`domain_type` is declared on the farfield base class, so every farfield type must
    drive the restriction; anything but a half body leaves the geometry alone."""
    params = _params_with_farfield(farfield_type(name="zone", domain_type=domain_type))
    compute = MagicMock(return_value=3.25 * unyt.m**2)
    monkeypatch.setattr("flow360.component.project_utils._projected_area", compute)

    _compute_projected_reference_area(params, MagicMock())

    assert compute.call_args.kwargs["clip"] == expected_clip


def test_submission_resolver_reports_why_the_area_was_restricted(monkeypatch):
    """The user set a domain_type, never a half space, so that is what the log has to name
    -- otherwise a halved reference area looks unexplained."""
    params = _params_with_farfield(
        fl.UserDefinedFarfield(name="zone", domain_type="half_body_negative_y")
    )
    monkeypatch.setattr(
        "flow360.component.project_utils._projected_area",
        MagicMock(return_value=3.25 * unyt.m**2),
    )
    messages = []
    monkeypatch.setattr("flow360.component.project_utils.log.info", messages.append)

    _compute_projected_reference_area(params, MagicMock())

    assert any(
        "domain_type=half_body_negative_y" in message and "-Y side only" in message
        for message in messages
    ), messages


def test_submission_resolver_ignores_legacy_automatic_settings(monkeypatch):
    with fl.SI_unit_system:
        params = SimulationParams(
            reference_geometry=ReferenceGeometry(
                area=2.0 * unyt.m**2,
                private_attribute_area_settings={"automatically": True},
            )
        )
    compute = MagicMock()
    info = MagicMock()
    monkeypatch.setattr("flow360.component.project_utils._projected_area", compute)
    monkeypatch.setattr("flow360.component.project_utils.log.info", info)

    _compute_projected_reference_area(params, None)

    assert params.reference_geometry.area.to_value("m**2") == pytest.approx(2.0)
    compute.assert_not_called()
    info.assert_not_called()
