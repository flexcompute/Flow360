"""End-to-end tests for DraftContext.compute_obb() pipeline.

Uses local geo-1 (hollow cylinder) test data with a mocked download layer.
Verifies the full pipeline: Surface → face_ids → TessellationFileLoader → UVF parser → PCA → OBBResult.
Also tests the SurfaceSelector and List[Surface] input routes.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from flow360.cloud.file_cache import CloudFileCache
from flow360.component.simulation.draft_context.obb.compute import (
    OBBResult,
    compute_obb,
)
from flow360.component.simulation.draft_context.obb.tessellation_loader import (
    TessellationFileLoader,
)
from flow360.component.simulation.draft_context.obb.uvf_parser import parse_manifest
from flow360.component.simulation.framework.entity_selector import SurfaceSelector
from flow360.component.simulation.primitives import GeometryBodyGroup, Surface

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
    "radius": 0.05000119251486435,
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
def local_tessellation_loader(tmp_path):
    """TessellationFileLoader that reads from local geo-1 test data instead of cloud."""
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
    return TessellationFileLoader(
        geometry_resources={GEOMETRY_ID: mock_resource},
        cloud_cache=cache,
    )


def _make_surface(name, face_ids):
    """Create a minimal Surface entity for testing."""
    return Surface(
        name=name,
        private_attribute_sub_components=face_ids,
    )


class TestTessellationLoaderE2E:
    """Test TessellationFileLoader with real geo-1 data."""

    def test_load_all_vertices(self, local_tessellation_loader):
        vertices = local_tessellation_loader.load_vertices(ALL_FACE_IDS)
        assert vertices.shape == (POC_ALL_FACES["num_vertices"], 3)
        assert vertices.dtype == np.float32

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

    def test_radius_matches_poc(self, local_tessellation_loader):
        vertices = local_tessellation_loader.load_vertices(ALL_FACE_IDS)
        result = compute_obb(vertices)
        assert abs(result.radius - POC_ALL_FACES["radius"]) < 1e-6

    def test_rotation_axis_matches_poc(self, local_tessellation_loader):
        vertices = local_tessellation_loader.load_vertices(ALL_FACE_IDS)
        result = compute_obb(vertices)
        poc_rot_axis = POC_ALL_FACES["axes"][POC_ALL_FACES["rotation_axis_index"]]
        dot = abs(np.dot(result.axis_of_rotation, poc_rot_axis))
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


class TestDraftContextComputeObb:
    """Test DraftContext.compute_obb() with mocked tessellation loader."""

    @pytest.fixture()
    def draft_with_surfaces(self, local_tessellation_loader):
        """Create a DraftContext with mock surfaces and a real tessellation loader."""
        from flow360.component.simulation.draft_context.context import DraftContext
        from flow360.component.simulation.entity_info import GeometryEntityInfo

        body_group = GeometryBodyGroup(
            name="body00001",
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
            tessellation_loader=local_tessellation_loader,
        )

    def test_compute_obb_with_surface_list(self, draft_with_surfaces):
        """Test passing a list of Surface entities."""
        surfaces = [draft_with_surfaces.surfaces[fid] for fid in ALL_FACE_IDS]
        result = draft_with_surfaces.compute_obb(surfaces)

        assert isinstance(result, OBBResult)
        np.testing.assert_allclose(result.center, POC_ALL_FACES["center"], atol=1e-6)

    def test_compute_obb_with_single_surface(self, draft_with_surfaces):
        """Test passing a single Surface entity."""
        surface = draft_with_surfaces.surfaces[ALL_FACE_IDS[0]]
        result = draft_with_surfaces.compute_obb(surface)
        assert isinstance(result, OBBResult)
        assert result.center is not None

    def test_compute_obb_with_selector(self, draft_with_surfaces):
        """Test passing a SurfaceSelector with glob pattern."""
        selector = SurfaceSelector(name="test").match("body00001_face0000*")
        result = draft_with_surfaces.compute_obb(selector)
        assert isinstance(result, OBBResult)
        # Glob matches all 6 faces (face00001-face00006)
        np.testing.assert_allclose(result.center, POC_ALL_FACES["center"], atol=1e-6)

    def test_compute_obb_no_loader_raises(self, draft_with_surfaces):
        """DraftContext without tessellation loader should raise on compute_obb."""
        from flow360.component.simulation.draft_context.context import DraftContext

        # Create a new draft using the same entity_info but WITHOUT a loader
        draft_no_loader = DraftContext(
            entity_info=draft_with_surfaces._entity_info,
            tessellation_loader=None,
        )
        surfaces = [draft_no_loader.surfaces[ALL_FACE_IDS[0]]]
        with pytest.raises(Exception, match="compute_obb.*requires.*Geometry"):
            draft_no_loader.compute_obb(surfaces)

    def test_compute_obb_with_length_unit(self, draft_with_surfaces, local_tessellation_loader):
        """When length_unit is provided, center/extents/radius have units."""
        import unyt

        from flow360.component.simulation.draft_context.context import DraftContext

        length_unit = 1.0 * unyt.m
        draft = DraftContext(
            entity_info=draft_with_surfaces._entity_info,
            tessellation_loader=local_tessellation_loader,
            length_unit=length_unit,
        )

        surfaces = [draft.surfaces[fid] for fid in ALL_FACE_IDS]
        result = draft.compute_obb(surfaces)

        # Center and extents should carry units
        assert isinstance(result.center, unyt.unyt_array)
        assert isinstance(result.extents, unyt.unyt_array)
        assert str(result.center.units) == "m"
        assert str(result.extents.units) == "m"

        # Radius should also carry units
        assert isinstance(result.radius, unyt.unyt_quantity)

        # Axes and axis_of_rotation should remain dimensionless numpy
        assert not isinstance(result.axes, unyt.unyt_array)
        assert not isinstance(result.axis_of_rotation, unyt.unyt_array)
