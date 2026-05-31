"""Tests for UVF manifest parsing and vertex extraction."""

import json

import numpy as np
import pytest

from flow360.component.simulation.draft_context.obb.uvf_parser import (
    _find_section,
    extract_face_vertices,
    parse_manifest,
    resolve_buffers,
)

# ---------------------------------------------------------------------------
# Helpers to build manifest / binary fixtures
# ---------------------------------------------------------------------------


def _make_position_buffer(vertices: np.ndarray) -> bytes:
    """Encode an (N,3) float32 array to raw bytes."""
    return vertices.astype(np.float32).tobytes()


def _make_index_buffer(indices: np.ndarray) -> bytes:
    """Encode a uint32 index array to raw bytes."""
    return indices.astype(np.uint32).tobytes()


def _make_solid_entry(
    solid_id: str,
    bin_path: str,
    position_offset: int,
    position_length: int,
    index_offset: int = None,
    index_length: int = None,
    lod: bool = False,
    lod_default: int = 0,
):
    """Build a manifest solid entry."""
    sections = [{"name": "position", "offset": position_offset, "length": position_length}]
    if index_offset is not None:
        sections.append({"name": "indices", "offset": index_offset, "length": index_length})

    buffer_payload = {"path": bin_path, "sections": sections}
    if lod:
        buffers = {"type": "lod", "default": lod_default, "levels": [buffer_payload]}
    else:
        buffers = {**buffer_payload, "type": "plain"}

    return {
        "id": solid_id,
        "resources": {"buffers": buffers},
    }


def _make_face_entry(face_id: str, parent_id: str, start_index: int, end_index: int):
    """Build a manifest face entry."""
    return {
        "id": face_id,
        "attributions": {"packedParentId": parent_id},
        "properties": {
            "bufferLocations": {
                "indices": [{"startIndex": start_index, "endIndex": end_index}],
            }
        },
    }


# ---------------------------------------------------------------------------
# parse_manifest
# ---------------------------------------------------------------------------


class TestParseManifest:
    def test_parses_json_string(self):
        data = json.dumps([{"id": "a"}, {"id": "b"}])
        result = parse_manifest(data)
        assert len(result) == 2
        assert result[0]["id"] == "a"

    def test_parses_bytes(self):
        data = json.dumps([{"id": "x"}]).encode("utf-8")
        result = parse_manifest(data)
        assert result[0]["id"] == "x"


# ---------------------------------------------------------------------------
# resolve_buffers
# ---------------------------------------------------------------------------


class TestResolveBuffers:
    def test_plain_returns_as_is(self):
        solid = {"resources": {"buffers": {"type": "plain", "path": "a.bin", "sections": []}}}
        result = resolve_buffers(solid, lod_level=None)
        assert result["path"] == "a.bin"

    def test_lod_uses_specified_level(self):
        solid = {
            "resources": {
                "buffers": {
                    "type": "lod",
                    "default": 0,
                    "levels": [
                        {"path": "lod0.bin", "sections": []},
                        {"path": "lod1.bin", "sections": []},
                    ],
                }
            }
        }
        result = resolve_buffers(solid, lod_level=1)
        assert result["path"] == "lod1.bin"

    def test_lod_falls_back_to_default(self):
        solid = {
            "resources": {
                "buffers": {
                    "type": "lod",
                    "default": 1,
                    "levels": [
                        {"path": "lod0.bin", "sections": []},
                        {"path": "lod1.bin", "sections": []},
                    ],
                }
            }
        }
        result = resolve_buffers(solid, lod_level=None)
        assert result["path"] == "lod1.bin"


# ---------------------------------------------------------------------------
# _find_section
# ---------------------------------------------------------------------------


class TestFindSection:
    def test_finds_existing_section(self):
        sections = [{"name": "position", "offset": 0}, {"name": "indices", "offset": 100}]
        assert _find_section(sections, "position")["offset"] == 0

    def test_returns_none_for_missing(self):
        sections = [{"name": "position", "offset": 0}]
        assert _find_section(sections, "normals") is None


# ---------------------------------------------------------------------------
# extract_face_vertices — indexed geometry
# ---------------------------------------------------------------------------


class TestExtractFaceVerticesIndexed:
    @pytest.fixture()
    def indexed_scene(self):
        """A minimal indexed scene: 1 solid, 2 faces, shared vertex pool."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1]],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 2, 1, 3, 0, 1, 4], dtype=np.uint32)

        pos_bytes = _make_position_buffer(vertices)
        idx_bytes = _make_index_buffer(indices)
        bin_data = idx_bytes + pos_bytes

        solid = _make_solid_entry(
            solid_id="solid_1",
            bin_path="mesh.bin",
            position_offset=len(idx_bytes),
            position_length=len(pos_bytes),
            index_offset=0,
            index_length=len(idx_bytes),
        )
        # face_a: indices 0..6 (two triangles), face_b: indices 6..9 (one triangle)
        face_a = _make_face_entry("face_a", "solid_1", start_index=0, end_index=6)
        face_b = _make_face_entry("face_b", "solid_1", start_index=6, end_index=9)

        manifest = [solid, face_a, face_b]
        bin_data_map = {"mesh.bin": bin_data}
        return manifest, bin_data_map

    def test_single_face(self, indexed_scene):
        manifest, bin_data_map = indexed_scene
        verts = extract_face_vertices(manifest, ["face_a"], bin_data_map)
        assert verts.shape == (6, 3)

    def test_multiple_faces(self, indexed_scene):
        manifest, bin_data_map = indexed_scene
        verts = extract_face_vertices(manifest, ["face_a", "face_b"], bin_data_map)
        assert verts.shape == (9, 3)

    def test_vertex_values(self, indexed_scene):
        manifest, bin_data_map = indexed_scene
        verts = extract_face_vertices(manifest, ["face_b"], bin_data_map)
        # face_b indices are [0, 1, 4] in the original index array at positions 6..9
        expected = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 1]], dtype=np.float32)
        np.testing.assert_array_equal(verts, expected)


# ---------------------------------------------------------------------------
# extract_face_vertices — unindexed geometry
# ---------------------------------------------------------------------------


class TestExtractFaceVerticesUnindexed:
    @pytest.fixture()
    def unindexed_scene(self):
        """A scene with no index section — buffer locations are float32 element offsets."""
        # 4 triangles worth of vertices (12 vertices, 36 floats)
        vertices = np.arange(36, dtype=np.float32).reshape(12, 3)
        pos_bytes = _make_position_buffer(vertices)

        solid = _make_solid_entry(
            solid_id="solid_1",
            bin_path="mesh.bin",
            position_offset=0,
            position_length=len(pos_bytes),
        )
        # face covers elements 0..18 (6 vertices = 2 triangles)
        face = _make_face_entry("face_1", "solid_1", start_index=0, end_index=18)

        manifest = [solid, face]
        bin_data_map = {"mesh.bin": pos_bytes}
        return manifest, bin_data_map

    def test_extracts_correct_vertex_count(self, unindexed_scene):
        manifest, bin_data_map = unindexed_scene
        verts = extract_face_vertices(manifest, ["face_1"], bin_data_map)
        assert verts.shape == (6, 3)

    def test_vertex_values(self, unindexed_scene):
        manifest, bin_data_map = unindexed_scene
        verts = extract_face_vertices(manifest, ["face_1"], bin_data_map)
        expected = np.arange(18, dtype=np.float32).reshape(6, 3)
        np.testing.assert_array_equal(verts, expected)


# ---------------------------------------------------------------------------
# extract_face_vertices — LOD support
# ---------------------------------------------------------------------------


class TestExtractFaceVerticesLod:
    def test_lod_level_selects_correct_buffer(self):
        """When using LOD buffers, the specified level should be used."""
        verts_lod0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        verts_lod1 = np.array([[9, 9, 9], [8, 8, 8], [7, 7, 7]], dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)

        idx_bytes = _make_index_buffer(indices)
        pos0_bytes = _make_position_buffer(verts_lod0)
        pos1_bytes = _make_position_buffer(verts_lod1)

        bin0 = idx_bytes + pos0_bytes
        bin1 = idx_bytes + pos1_bytes

        solid = {
            "id": "solid_1",
            "resources": {
                "buffers": {
                    "type": "lod",
                    "default": 0,
                    "levels": [
                        {
                            "path": "lod0.bin",
                            "sections": [
                                {"name": "indices", "offset": 0, "length": len(idx_bytes)},
                                {
                                    "name": "position",
                                    "offset": len(idx_bytes),
                                    "length": len(pos0_bytes),
                                },
                            ],
                        },
                        {
                            "path": "lod1.bin",
                            "sections": [
                                {"name": "indices", "offset": 0, "length": len(idx_bytes)},
                                {
                                    "name": "position",
                                    "offset": len(idx_bytes),
                                    "length": len(pos1_bytes),
                                },
                            ],
                        },
                    ],
                }
            },
        }
        face = _make_face_entry("face_1", "solid_1", start_index=0, end_index=3)
        manifest = [solid, face]
        bin_data_map = {"lod0.bin": bin0, "lod1.bin": bin1}

        verts = extract_face_vertices(manifest, ["face_1"], bin_data_map, lod_level=1)
        np.testing.assert_array_equal(verts, verts_lod1)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestExtractFaceVerticesErrors:
    def test_missing_face_id_raises_key_error(self):
        manifest = [
            {
                "id": "solid_1",
                "resources": {"buffers": {"type": "plain", "path": "a.bin", "sections": []}},
            }
        ]
        with pytest.raises(KeyError, match="nonexistent"):
            extract_face_vertices(manifest, ["nonexistent"], {})

    def test_missing_position_section_raises_value_error(self):
        solid = _make_solid_entry("solid_1", "mesh.bin", position_offset=0, position_length=0)
        # Remove the position section
        solid["resources"]["buffers"]["sections"] = []
        face = _make_face_entry("face_1", "solid_1", start_index=0, end_index=3)
        manifest = [solid, face]

        with pytest.raises(ValueError, match="no position section"):
            extract_face_vertices(manifest, ["face_1"], {"mesh.bin": b"\x00" * 100})
