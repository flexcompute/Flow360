"""End-to-end tests for DraftContext.compute_obb() pipeline.

Uses local geo-1 (hollow cylinder) test data with a mocked download layer.
Verifies the full pipeline: Surface → face_ids → TessellationFileLoader → UVF parser → PCA → OBBResult.
Also tests the SurfaceSelector and List[Surface] input routes.
"""

from pathlib import Path
from unittest.mock import MagicMock

import flow360_schema.models.simulation.units as u
import numpy as np
import pytest
import unyt
from flow360_schema.framework.entity.entity_operation import CoordinateSystem
from flow360_schema.framework.entity.entity_selector import SurfaceSelector
from flow360_schema.models.entities.geometry_entities import GeometryBodyGroup
from flow360_schema.models.entities.surface_entities import Surface

import flow360 as fl
from flow360.cloud.file_cache import CloudFileCache
from flow360.component.simulation.measurement.bounding_box import OBBResult
from flow360.component.simulation.measurement.bounding_box import (
    _compute_oriented_bounding_box as compute_obb,
)
from flow360.component.simulation.measurement.tessellation_loader import (
    TessellationFileLoader,
)

GEO_1_DIR = Path(__file__).resolve().parents[2] / "data" / "tessellation" / "geo-1"
GEOMETRY_ID = "geo-1-test"

# POC reference values for geo-1 (all 6 faces)
POC_ALL_FACES = {
    "center": [1.2756629078265226e-10, 1.0117358560906686e-10, 1.4181101660255997e-09],
    "axes": [
        [-0.07556681447845126, -0.0003246319295740485, 0.9971406877485693],
        [0.9971377618421228, -0.0024688959043392446, 0.0755657889623094],
        [0.0024373054921518525, 0.9999968995787557, 0.0005102693544478121],
    ],
    "extents": [0.04999054650361766, 0.050011838526111047, 0.01012446986866631],
    "rotation_axis_index": 2,
    "averaged_radius": 0.05000119251486435,
    "num_vertices": 45156,
}

ALL_FACE_IDS = [
    "body00001_face00001",
    "body00001_face00002",
    "body00001_face00003",
    "body00001_face00004",
    "body00001_face00005",
    "body00001_face00006",
]


@pytest.fixture()
def local_geometry_root(tmp_path, monkeypatch):
    """Geometry root that reads geo-1 tessellation data from disk."""
    mock_resource = MagicMock()

    def fake_download(file_path, *, to_file, log_error=True, verbose=True):
        # Map cloud path to local test data
        local_file = GEO_1_DIR / file_path
        assert local_file.exists(), f"Test data missing: {local_file}"
        with open(local_file, "rb") as src, open(to_file, "wb") as dst:
            dst.write(src.read())
        return to_file

    mock_resource._download_file = lambda file_path, to_file, **kwargs: fake_download(
        file_path, to_file=to_file, **kwargs
    )

    cache = CloudFileCache(cache_root=tmp_path / "cache", max_size_bytes=100 * 1024 * 1024)
    monkeypatch.setattr(
        "flow360.component.simulation.measurement.geometry.get_shared_cloud_file_cache",
        lambda: cache,
    )

    geometry_root = MagicMock()
    geometry_root.id = GEOMETRY_ID
    geometry_root._webapi = mock_resource
    # A real Geometry resource always carries this; measurement now hard-requires it.
    geometry_root._project_length_unit = 1.0 * unyt.m
    return geometry_root


@pytest.fixture()
def local_geometry_resources(local_geometry_root):
    """Geometry resource map for direct TessellationFileLoader tests."""
    return {local_geometry_root.id: local_geometry_root._webapi}


@pytest.fixture()
def local_tessellation_loader(tmp_path, local_geometry_resources):
    """TessellationFileLoader that reads from local geo-1 test data instead of cloud."""

    cache = CloudFileCache(cache_root=tmp_path / "cache", max_size_bytes=100 * 1024 * 1024)
    return TessellationFileLoader(
        geometry_resources=local_geometry_resources,
        cloud_cache=cache,
    )


def _make_surface(name, face_ids):
    """Create a minimal Surface entity for testing."""
    return Surface(
        name=name,
        private_attribute_id=name,
        private_attribute_sub_components=face_ids,
    )


def _assert_obb_matches(actual, expected):
    np.testing.assert_allclose(actual.center, expected.center, atol=1e-9)
    np.testing.assert_allclose(
        sorted(actual.extents.tolist()),
        sorted(expected.extents.tolist()),
        atol=1e-9,
    )
    axis_alignment = np.abs(actual.axes @ expected.axes.T)
    assert np.all(axis_alignment.max(axis=0) > 0.999)


class TestTessellationLoaderE2E:
    """Test TessellationFileLoader with real geo-1 data."""

    def test_load_all_vertices(self, local_tessellation_loader):
        vertices = local_tessellation_loader.load_vertices(ALL_FACE_IDS)
        assert vertices.shape == (POC_ALL_FACES["num_vertices"], 3)
        assert vertices.dtype == np.float32

    def test_load_triangles_preserves_face_ownership(self, local_tessellation_loader):
        face_ids = [ALL_FACE_IDS[1], ALL_FACE_IDS[0]]
        triangles = local_tessellation_loader.load_triangles(face_ids)

        assert list(triangles) == face_ids
        assert all(value.shape[1:] == (3, 3) for value in triangles.values())
        np.testing.assert_array_equal(
            np.concatenate(list(triangles.values()), axis=0).reshape(-1, 3),
            local_tessellation_loader.load_vertices(face_ids),
        )

    def test_obb_matches_poc_reference(self, local_tessellation_loader):
        vertices = local_tessellation_loader.load_vertices(ALL_FACE_IDS)
        result = compute_obb(vertices)

        np.testing.assert_allclose(result.center, POC_ALL_FACES["center"], atol=1e-6)
        np.testing.assert_allclose(
            sorted(result.extents.tolist(), reverse=True),
            sorted(POC_ALL_FACES["extents"], reverse=True),
            atol=1e-6,
        )

        # Axes alignment (sign-agnostic)
        poc_axes = np.array(POC_ALL_FACES["axes"])
        dots = np.abs(result.axes @ poc_axes.T)
        assert np.all(dots.max(axis=0) > 0.999)

    def test_averaged_radius_matches_poc(self, local_tessellation_loader):
        vertices = local_tessellation_loader.load_vertices(ALL_FACE_IDS)
        result = compute_obb(vertices)
        rotation_info = result.get_rotation_axis_and_radius(
            axis_index=POC_ALL_FACES["rotation_axis_index"]
        )
        assert abs(rotation_info.averaged_radius - POC_ALL_FACES["averaged_radius"]) < 1e-6

    def test_rotation_axis_matches_poc(self, local_tessellation_loader):
        vertices = local_tessellation_loader.load_vertices(ALL_FACE_IDS)
        result = compute_obb(vertices)
        rotation_info = result.get_rotation_axis_and_radius(
            axis_index=POC_ALL_FACES["rotation_axis_index"]
        )
        poc_rot_axis = POC_ALL_FACES["axes"][POC_ALL_FACES["rotation_axis_index"]]
        dot = abs(np.dot(rotation_info.axis_of_rotation, poc_rot_axis))
        assert dot > 0.999

    def test_caching_returns_same_result(self, local_tessellation_loader):
        """Second call should hit L1 cache and return identical results."""
        v1 = local_tessellation_loader.load_vertices(ALL_FACE_IDS)
        v2 = local_tessellation_loader.load_vertices(ALL_FACE_IDS)
        np.testing.assert_array_equal(v1, v2)

    def test_subset_faces(self, local_tessellation_loader):
        """Loading a subset of faces returns fewer vertices."""
        v_all = local_tessellation_loader.load_vertices(ALL_FACE_IDS)
        v_two = local_tessellation_loader.load_vertices(ALL_FACE_IDS[:2])
        assert len(v_two) < len(v_all)

    def test_disk_cache_hit(self, local_tessellation_loader, tmp_path):
        """After L2 cache is populated, a new loader instance can read from disk."""
        local_tessellation_loader.load_vertices(ALL_FACE_IDS)

        # Create a second loader pointing at the same disk cache
        mock_resource = MagicMock()
        mock_resource._download_file = MagicMock(
            side_effect=AssertionError("Should not download — disk cache should serve")
        )
        cache2 = CloudFileCache(cache_root=tmp_path / "cache", max_size_bytes=100 * 1024 * 1024)
        loader2 = TessellationFileLoader(
            geometry_resources={GEOMETRY_ID: mock_resource},
            cloud_cache=cache2,
        )
        v2 = loader2.load_vertices(ALL_FACE_IDS)
        assert v2.shape[0] == POC_ALL_FACES["num_vertices"]


class TestOrientedBoundingBox:
    """Test the public measurement facade with a Geometry-root draft."""

    @pytest.fixture()
    def draft_with_surfaces(self, local_geometry_root):
        """Create a DraftContext with mock surfaces and a Geometry root."""
        from flow360_schema.models.entity_info import GeometryEntityInfo

        from flow360.component.simulation.draft_context.context import DraftContext

        body_group = GeometryBodyGroup(
            name="body00001",
            private_attribute_id="body00001",
            private_attribute_tag_key="groupByFile",
            private_attribute_sub_components=["body00001"],
        )

        entity_info = GeometryEntityInfo(
            face_ids=ALL_FACE_IDS,
            face_attribute_names=["faceId"],
            face_group_tag="faceId",
            body_ids=["body00001"],
            body_attribute_names=["groupByFile"],
            body_group_tag="groupByFile",
            grouped_faces=[
                [_make_surface(fid, [fid]) for fid in ALL_FACE_IDS],
            ],
            grouped_bodies=[[body_group]],
            bodies_face_edge_ids={
                "body00001": {"face_ids": ALL_FACE_IDS},
            },
        )

        return DraftContext(
            entity_info=entity_info,
            geometry_root=local_geometry_root,
        )

    def test_with_surface_list(self, draft_with_surfaces):
        """Test passing a list of Surface entities."""
        surfaces = [draft_with_surfaces.surfaces[fid] for fid in ALL_FACE_IDS]
        result = fl.measure.oriented_bounding_box(draft_with_surfaces, surfaces=surfaces)

        assert isinstance(result, OBBResult)
        np.testing.assert_allclose(result.center.to_value("m"), POC_ALL_FACES["center"], atol=1e-6)

    def test_with_single_surface(self, draft_with_surfaces):
        """Test passing a single Surface entity."""
        surface = draft_with_surfaces.surfaces[ALL_FACE_IDS[0]]
        result = fl.measure.oriented_bounding_box(draft_with_surfaces, surfaces=surface)
        assert isinstance(result, OBBResult)
        assert result.center is not None

    def test_with_selector(self, draft_with_surfaces):
        """Test passing a SurfaceSelector with glob pattern."""
        selector = SurfaceSelector(name="test").match("body00001_face0000*")
        result = fl.measure.oriented_bounding_box(draft_with_surfaces, surfaces=selector)
        assert isinstance(result, OBBResult)
        # Glob matches all 6 faces (face00001-face00006)
        np.testing.assert_allclose(result.center.to_value("m"), POC_ALL_FACES["center"], atol=1e-6)

    def test_with_selector_list(self, draft_with_surfaces):
        """A list of selectors is accepted; their matches are unioned."""
        first_half = SurfaceSelector(name="first").match("body00001_face0000[123]")
        second_half = SurfaceSelector(name="second").match("body00001_face0000[456]")
        result = fl.measure.oriented_bounding_box(
            draft_with_surfaces, surfaces=[first_half, second_half]
        )
        np.testing.assert_allclose(result.center.to_value("m"), POC_ALL_FACES["center"], atol=1e-6)

    def test_with_mixed_surface_and_selector_list(self, draft_with_surfaces):
        """Explicit entities and selectors can be mixed in one list."""
        explicit = [draft_with_surfaces.surfaces[fid] for fid in ALL_FACE_IDS[:3]]
        selector = SurfaceSelector(name="rest").match("body00001_face0000[456]")
        result = fl.measure.oriented_bounding_box(
            draft_with_surfaces, surfaces=explicit + [selector]
        )
        np.testing.assert_allclose(result.center.to_value("m"), POC_ALL_FACES["center"], atol=1e-6)

    def test_duplicated_selection_is_deduplicated(self, draft_with_surfaces):
        """Repeating the same surfaces -- listed twice or via overlapping selectors --
        must not change the result, and callers need not deduplicate first."""
        surfaces = [draft_with_surfaces.surfaces[fid] for fid in ALL_FACE_IDS]
        overlapping_selector = SurfaceSelector(name="all").match("body00001_face0000*")

        reference = fl.measure.projected_area(draft_with_surfaces, surfaces=surfaces)
        duplicated = fl.measure.projected_area(
            draft_with_surfaces, surfaces=surfaces + surfaces + [overlapping_selector]
        )
        assert duplicated.to_value("m**2") == pytest.approx(reference.to_value("m**2"))

    def test_with_entity_list_from_boundary_condition(self, draft_with_surfaces):
        """An EntityList -- e.g. a Wall's `entities` -- can be reused verbatim."""
        surfaces = [draft_with_surfaces.surfaces[fid] for fid in ALL_FACE_IDS]
        wall = fl.Wall(name="wall", surfaces=surfaces)
        result = fl.measure.oriented_bounding_box(draft_with_surfaces, surfaces=wall.entities)
        np.testing.assert_allclose(result.center.to_value("m"), POC_ALL_FACES["center"], atol=1e-6)

    def test_rejects_non_surface_selector(self, draft_with_surfaces):
        """Passing an EdgeSelector or BodyGroupSelector should raise immediately."""
        from flow360_schema.framework.entity.entity_selector import EdgeSelector

        with pytest.raises(Exception, match="SurfaceSelector"):
            fl.measure.oriented_bounding_box(
                draft_with_surfaces,
                surfaces=EdgeSelector(name="bad").match("*"),
            )

    def test_without_geometry_root_raises(self, draft_with_surfaces):
        """Measurement requires a Geometry-root draft."""
        from flow360.component.simulation.draft_context.context import DraftContext

        draft_without_geometry_root = DraftContext(entity_info=draft_with_surfaces._entity_info)
        surfaces = [draft_without_geometry_root.surfaces[ALL_FACE_IDS[0]]]
        with pytest.raises(Exception, match="oriented_bounding_box.*requires.*Geometry"):
            fl.measure.oriented_bounding_box(draft_without_geometry_root, surfaces=surfaces)

    def test_missing_length_unit_raises(self, draft_with_surfaces, local_geometry_root):
        """A Geometry without a project length unit is a hard error, not a bare float."""
        local_geometry_root._project_length_unit = None
        surfaces = [draft_with_surfaces.surfaces[fid] for fid in ALL_FACE_IDS]

        with pytest.raises(Exception, match="project length unit"):
            fl.measure.oriented_bounding_box(draft_with_surfaces, surfaces=surfaces)
        with pytest.raises(Exception, match="project length unit"):
            fl.measure.projected_area(draft_with_surfaces, surfaces=surfaces)

    def test_with_length_unit(self, draft_with_surfaces):
        """Center and extents carry the geometry's length unit; axes stay dimensionless."""
        surfaces = [draft_with_surfaces.surfaces[fid] for fid in ALL_FACE_IDS]
        result = fl.measure.oriented_bounding_box(draft_with_surfaces, surfaces=surfaces)

        # Center and extents should carry units
        assert isinstance(result.center, unyt.unyt_array)
        assert isinstance(result.extents, unyt.unyt_array)
        assert str(result.center.units) == "m"
        assert str(result.extents.units) == "m"
        assert not hasattr(result, "radius")
        assert not hasattr(result, "axis_of_rotation")

        rotation_info = result.get_rotation_axis_and_radius(
            axis_index=POC_ALL_FACES["rotation_axis_index"]
        )
        assert isinstance(rotation_info.averaged_radius, unyt.unyt_quantity)
        assert str(rotation_info.averaged_radius.units) == "m"

        # Axes and derived axis_of_rotation should remain dimensionless numpy
        assert not isinstance(result.axes, unyt.unyt_array)
        assert not isinstance(rotation_info.axis_of_rotation, unyt.unyt_array)

    def test_half_body_trimming_through_real_tessellation(self, draft_with_surfaces):
        """geo-1 is a cylinder centred on Y=0, so trimming to one side must halve the
        silhouette. Exercises the trim against real UVF data rather than synthetic
        triangles: the cut runs through the middle of the tessellation."""
        from flow360.component.simulation.measurement.projection import _projected_area

        surfaces = [draft_with_surfaces.surfaces[face_id] for face_id in ALL_FACE_IDS]
        arguments = {
            "surfaces": surfaces,
            "direction": "X",
            "render_quality": "medium",
        }

        full = _projected_area(draft_with_surfaces, clip=None, **arguments)
        negative = _projected_area(draft_with_surfaces, clip="-Y", **arguments)
        positive = _projected_area(draft_with_surfaces, clip="+Y", **arguments)

        assert full.to_value("m**2") == pytest.approx(0.002, rel=1e-6)
        assert negative.to_value("m**2") == pytest.approx(0.001, rel=1e-3)
        assert positive.to_value("m**2") == pytest.approx(0.001, rel=1e-3)

    def test_projected_area_with_coordinate_rotation(self, draft_with_surfaces):
        """Regress projected area through real UVF data and coordinate transforms.

        The coordinate system is assigned to the body group that owns the faces, which
        is the only assignment either interface can produce: a coordinate system never
        attaches to an individual surface.
        """
        surfaces = [draft_with_surfaces.surfaces[face_id] for face_id in ALL_FACE_IDS]
        recipe = fl.ProjectedArea(
            surfaces=surfaces,
            direction="X",
            render_quality="medium",
        )

        unrotated = fl.measure.projected_area(
            draft_with_surfaces,
            surfaces=recipe.surfaces,
            direction=recipe.direction,
            render_quality=recipe.render_quality,
        )

        assert isinstance(unrotated, unyt.unyt_quantity)
        assert unrotated.to_value("m**2") == pytest.approx(0.002, rel=1e-6)

        draft_with_surfaces.coordinate_systems.assign(
            entities=draft_with_surfaces.body_groups["body00001"],
            coordinate_system=CoordinateSystem(
                name="Coordinate System 2",
                axis_of_rotation=(0, 1, 1),
                angle_of_rotation=15 * u.degree,
            ),
        )
        rotated = fl.measure.projected_area(
            draft_with_surfaces,
            surfaces=recipe.surfaces,
            direction=recipe.direction,
            render_quality=recipe.render_quality,
        )

        assert rotated.to_value("m**2") == pytest.approx(
            0.00340332933085,
            rel=1e-5,
        )

    def test_transforms_are_resolved_per_body_group(self, local_geometry_root):
        """Each surface must take the transform of its own owning body group.

        Production setups assign separate coordinate systems to separate body groups --
        wheels and suspension on one, the vehicle body on another -- so resolving a
        single transform for the whole draft would move both alike.
        """
        from flow360_schema.models.entity_info import GeometryEntityInfo

        from flow360.component.simulation.draft_context.context import DraftContext
        from flow360.component.simulation.measurement.geometry import (
            _resolve_surface_transforms,
        )

        wheel_face = "body00001_face00001"
        body_face = "body00002_face00001"
        draft = DraftContext(
            entity_info=GeometryEntityInfo(
                face_ids=[wheel_face, body_face],
                face_attribute_names=["faceId"],
                face_group_tag="faceId",
                body_ids=["body00001", "body00002"],
                body_attribute_names=["groupByFile"],
                body_group_tag="groupByFile",
                grouped_faces=[
                    [_make_surface(wheel_face, [wheel_face]), _make_surface(body_face, [body_face])]
                ],
                grouped_bodies=[
                    [
                        GeometryBodyGroup(
                            name=body_id,
                            private_attribute_id=body_id,
                            private_attribute_tag_key="groupByFile",
                            private_attribute_sub_components=[body_id],
                        )
                        for body_id in ("body00001", "body00002")
                    ]
                ],
                bodies_face_edge_ids={
                    "body00001": {"face_ids": [wheel_face]},
                    "body00002": {"face_ids": [body_face]},
                },
            ),
            geometry_root=local_geometry_root,
        )

        wheel_system = CoordinateSystem(
            name="wheels", axis_of_rotation=(0, 1, 0), angle_of_rotation=20 * u.degree
        )
        body_system = CoordinateSystem(name="body", translation=(0.0, 0.0, 0.4) * u.m)
        draft.coordinate_systems.assign(
            entities=draft.body_groups["body00001"], coordinate_system=wheel_system
        )
        draft.coordinate_systems.assign(
            entities=draft.body_groups["body00002"], coordinate_system=body_system
        )

        transforms = _resolve_surface_transforms(draft)

        assert set(transforms) == {wheel_face, body_face}
        # pylint: disable=protected-access
        get_matrix = draft.coordinate_systems._get_coordinate_system_matrix
        np.testing.assert_allclose(
            transforms[wheel_face], get_matrix(coordinate_system=wheel_system)
        )
        np.testing.assert_allclose(transforms[body_face], get_matrix(coordinate_system=body_system))

    def test_applies_composed_rotation_and_translation(
        self,
        draft_with_surfaces,
        local_tessellation_loader,
    ):
        surfaces = [draft_with_surfaces.surfaces[fid] for fid in ALL_FACE_IDS]
        parent = draft_with_surfaces.coordinate_systems.add(
            coordinate_system=CoordinateSystem(
                name="translated",
                translation=(1.5, -2.0, 0.25) * u.m,
            )
        )
        child = draft_with_surfaces.coordinate_systems.add(
            coordinate_system=CoordinateSystem(
                name="rotated",
                axis_of_rotation=(1, 1, 0),
                angle_of_rotation=37 * u.degree,
            ),
            parent=parent,
        )
        draft_with_surfaces.coordinate_systems.assign(
            entities=draft_with_surfaces.body_groups["body00001"],
            coordinate_system=child,
        )

        matrix = draft_with_surfaces.coordinate_systems._get_coordinate_system_matrix(
            coordinate_system=child
        )
        vertices = local_tessellation_loader.load_vertices(ALL_FACE_IDS)
        expected = compute_obb(vertices @ matrix[:, :3].T + matrix[:, 3])
        actual = fl.measure.oriented_bounding_box(
            draft_with_surfaces,
            surfaces=surfaces,
        )

        _assert_obb_matches(actual, expected)

    def test_applies_non_uniform_scale(
        self,
        draft_with_surfaces,
        local_tessellation_loader,
    ):
        surfaces = [draft_with_surfaces.surfaces[fid] for fid in ALL_FACE_IDS]
        coordinate_system = CoordinateSystem(
            name="scaled",
            scale=(2.0, 0.5, 3.0),
        )
        draft_with_surfaces.coordinate_systems.assign(
            entities=draft_with_surfaces.body_groups["body00001"],
            coordinate_system=coordinate_system,
        )

        matrix = coordinate_system.get_transformation_matrix()
        vertices = local_tessellation_loader.load_vertices(ALL_FACE_IDS)
        expected = compute_obb(vertices @ matrix[:, :3].T + matrix[:, 3])
        actual = fl.measure.oriented_bounding_box(
            draft_with_surfaces,
            surfaces=surfaces,
        )

        _assert_obb_matches(actual, expected)

    def test_rejects_non_surface_view(self, draft_with_surfaces):
        with pytest.raises(Exception, match="Surface view"):
            fl.measure.oriented_bounding_box(
                draft_with_surfaces,
                surfaces=draft_with_surfaces.body_groups,
            )

    def test_rejects_mirrored_surface(self, draft_with_surfaces):
        from flow360_schema.models.entities.surface_entities import MirroredSurface

        mirrored_surface = MirroredSurface(
            name="mirrored",
            surface_id="source-surface",
            mirror_plane_id="mirror-plane",
        )
        with pytest.raises(Exception, match="MirroredSurface"):
            fl.measure.oriented_bounding_box(draft_with_surfaces, surfaces=[mirrored_surface])

    def test_rejects_surface_without_face_identifiers(self, draft_with_surfaces):
        with pytest.raises(Exception, match="tessellation face identifiers"):
            fl.measure.oriented_bounding_box(
                draft_with_surfaces,
                surfaces=Surface(
                    name="face-less",
                    private_attribute_id="face-less",
                ),
            )

    def test_draft_compute_obb_warns_and_delegates(self, draft_with_surfaces):
        surfaces = [draft_with_surfaces.surfaces[fid] for fid in ALL_FACE_IDS]
        with pytest.warns(DeprecationWarning, match="draft.compute_obb"):
            result = draft_with_surfaces.compute_obb(surfaces)

        assert isinstance(result, OBBResult)
