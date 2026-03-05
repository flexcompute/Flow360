"""
Tests for the geometry tree and face grouping API.

Uses the airplane RC geometry (194 faces, 7 colors) as the test model.

Covers:
- TreeBackend: versioned JSON loading, node traversal, filtering
- Geometry.from_local_tree(): loading from local hierarchical metadata JSON
- Tree navigation: root_node, children, descendants, faces (with filters)
- Node attribute access (including FaceGroup.face_ids)
- Face group management: create_face_group, get_face_group, list_groups, clear_groups
- save_groups_to_file and _build_face_grouping_config: export to JSON
- Set operations on NodeSet and Geometry
- Filter pattern matching (glob)
"""

import json
import os

import pytest

from flow360.component.geometry import Geometry
from flow360.component.geometry_tree import TreeBackend
from flow360.component.geometry_tree.face_group import FaceGroup
from flow360.component.geometry_tree.filters import matches_pattern
from flow360.exceptions import Flow360ValueError

TREE_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/geometry_tree")
AIRPLANE_JSON_PATH = os.path.join(TREE_DATA_DIR, "airplane_rc_geometry_tree.json")

AIRPLANE_COLOR_EXPECTED = [
    ("255,0,255", 40),
    ("0,0,255", 37),
    ("0,255,0", 37),
    ("0,255,255", 27),
    ("255,255,0", 27),
    ("", 24),
    ("0,0,0", 2),
]


@pytest.fixture
def airplane_json():
    with open(AIRPLANE_JSON_PATH, "r") as f:
        return json.load(f)


@pytest.fixture
def airplane_tree(airplane_json):
    tree = TreeBackend()
    tree.load_from_json(airplane_json)
    return tree


@pytest.fixture
def airplane_geometry():
    return Geometry.from_local_tree(AIRPLANE_JSON_PATH)


# ================================================================
# TreeBackend tests
# ================================================================


class TestTreeBackend:
    def test_load_from_json_versioned(self, airplane_tree):
        assert airplane_tree.get_root() is not None
        # ModelFile + 2 Assembly + Part + Body + 3 BodyCollection
        # + ShellCollection + Shell + 194 Faces = 204
        assert airplane_tree.node_count() == 204

    def test_load_from_json_requires_version_key(self):
        """Unversioned data (no 'tree' key) should raise KeyError."""
        backend = TreeBackend()
        raw_tree = {"name": "root", "type": "ModelFile", "children": []}
        with pytest.raises(KeyError):
            backend.load_from_json(raw_tree)

    def test_load_from_file(self):
        backend = TreeBackend()
        root_id = backend.load_from_file(AIRPLANE_JSON_PATH)
        assert root_id is not None
        assert backend.node_count() == 204

    def test_root_node_attributes(self, airplane_tree):
        root_attrs = airplane_tree.get_node_attrs(airplane_tree.get_root())
        assert root_attrs["name"] == "Solid-Body-RC-Plane_v2024_colored"
        assert root_attrs["type"] == "ModelFile"

    def test_get_children(self, airplane_tree):
        children = airplane_tree.get_children(airplane_tree.get_root())
        assert len(children) == 1  # single Assembly child

    def test_get_descendants(self, airplane_tree):
        descendants = airplane_tree.get_descendants(airplane_tree.get_root())
        assert len(descendants) == 203  # 204 total - 1 root

    def test_get_parent(self, airplane_tree):
        root = airplane_tree.get_root()
        children = airplane_tree.get_children(root)
        assert airplane_tree.get_parent(children[0]) == root
        assert airplane_tree.get_parent(root) is None

    def test_get_all_faces(self, airplane_tree):
        assert len(airplane_tree.get_all_faces()) == 194

    def test_filter_nodes_by_type(self, airplane_tree):
        all_nodes = airplane_tree.get_all_nodes()
        assert len(airplane_tree.filter_nodes(all_nodes, type="Face")) == 194
        assert len(airplane_tree.filter_nodes(all_nodes, type="Shell")) == 1
        assert len(airplane_tree.filter_nodes(all_nodes, type="BodyCollection")) == 3

    def test_nonexistent_node(self, airplane_tree):
        assert airplane_tree.get_node_attrs("nonexistent") == {}
        assert airplane_tree.get_children("nonexistent") == []
        assert airplane_tree.get_parent("nonexistent") is None
        assert airplane_tree.get_descendants("nonexistent") == set()


# ================================================================
# Geometry.from_local_tree() tests
# ================================================================


class TestGeometryFromLocalTree:
    def test_loads_tree_with_correct_face_count(self, airplane_geometry):
        assert len(airplane_geometry.faces()) == 194

    def test_repr(self, airplane_geometry):
        assert "194 faces" in repr(airplane_geometry)

    def test_root_node_raises_without_tree(self):
        geo = Geometry(id=None)
        with pytest.raises(Flow360ValueError):
            geo.root_node()


# ================================================================
# Tree navigation tests
# ================================================================


class TestTreeNavigation:
    def test_root_node(self, airplane_geometry):
        root = airplane_geometry.root_node()
        assert len(root) == 1
        node = next(iter(root))
        assert node.name == "Solid-Body-RC-Plane_v2024_colored"
        assert node.type == "ModelFile"

    def test_children(self, airplane_geometry):
        children = airplane_geometry.children()
        assert len(children) == 1

    def test_descendants(self, airplane_geometry):
        assert len(airplane_geometry.descendants()) == 203

    def test_descendants_with_type_filter(self, airplane_geometry):
        assert len(airplane_geometry.descendants(type="Part")) == 1
        assert len(airplane_geometry.descendants(type="Body")) == 1
        assert len(airplane_geometry.descendants(type="Shell")) == 1
        assert len(airplane_geometry.descendants(type="Assembly")) == 2
        assert len(airplane_geometry.descendants(type="BodyCollection")) == 3

    def test_faces_no_filter(self, airplane_geometry):
        assert len(airplane_geometry.faces()) == 194

    @pytest.mark.parametrize("color,expected_count", AIRPLANE_COLOR_EXPECTED)
    def test_faces_filter_by_color(self, airplane_geometry, color, expected_count):
        assert len(airplane_geometry.faces(colorRGB=color)) == expected_count

    def test_faces_filter_no_match(self, airplane_geometry):
        assert len(airplane_geometry.faces(colorRGB="999,999,999")) == 0

    def test_all_color_filters_sum_to_total(self, airplane_geometry):
        total = sum(
            len(airplane_geometry.faces(colorRGB=color))
            for color, _ in AIRPLANE_COLOR_EXPECTED
        )
        assert total == 194

    def test_shell_has_all_faces_as_children(self, airplane_geometry):
        shell = next(iter(airplane_geometry.descendants(type="Shell")))
        children = shell.children()
        assert len(children) == 194
        for child in children:
            assert child.is_face()

    def test_body_collections_are_empty(self, airplane_geometry):
        """The 3 BodyCollections (Sketch1/2/3) have no face descendants."""
        body_collections = airplane_geometry.descendants(type="BodyCollection")
        assert len(body_collections) == 3
        for bc in body_collections:
            assert len(bc.faces()) == 0


# ================================================================
# Node attribute tests
# ================================================================


class TestNodeAttributes:
    def test_face_node_properties(self, airplane_geometry):
        magenta_faces = airplane_geometry.faces(colorRGB="255,0,255")
        node = next(iter(magenta_faces))
        assert node.type == "Face"
        assert node.color == "255,0,255"
        assert node.uuid is not None
        assert node.is_face()

    def test_non_face_node(self, airplane_geometry):
        node = next(iter(airplane_geometry.descendants(type="Part")))
        assert node.type == "Part"
        assert not node.is_face()

    def test_node_equality(self, airplane_geometry):
        faces1 = list(airplane_geometry.faces(colorRGB="0,0,0"))
        faces2 = list(airplane_geometry.faces(colorRGB="0,0,0"))
        assert len(faces1) == 2
        # Same query yields same nodes
        assert set(n.node_id for n in faces1) == set(n.node_id for n in faces2)

    def test_face_uuids_are_unique(self, airplane_geometry):
        uuids = [node.uuid for node in airplane_geometry.faces()]
        assert len(uuids) == len(set(uuids))


# ================================================================
# Face group management tests
# ================================================================


class TestFaceGroupManagement:
    def test_create_face_group(self, airplane_geometry):
        group = airplane_geometry.create_face_group(
            "magenta", airplane_geometry.faces(colorRGB="255,0,255")
        )
        assert isinstance(group, FaceGroup)
        assert group.name == "magenta"
        assert group.face_count() == 40

    def test_face_group_face_ids_returns_copy(self, airplane_geometry):
        group = airplane_geometry.create_face_group(
            "blue", airplane_geometry.faces(colorRGB="0,0,255")
        )
        ids = group.face_ids
        assert len(ids) == 37
        # Mutating the returned set should not affect the group
        ids.clear()
        assert group.face_count() == 37

    def test_list_groups(self, airplane_geometry):
        airplane_geometry.create_face_group("magenta", airplane_geometry.faces(colorRGB="255,0,255"))
        airplane_geometry.create_face_group("blue", airplane_geometry.faces(colorRGB="0,0,255"))
        assert airplane_geometry.list_groups() == ["magenta", "blue"]

    def test_get_face_group(self, airplane_geometry):
        airplane_geometry.create_face_group("green", airplane_geometry.faces(colorRGB="0,255,0"))
        group = airplane_geometry.get_face_group("green")
        assert group.name == "green"
        assert group.face_count() == 37

    def test_get_face_group_not_found(self, airplane_geometry):
        with pytest.raises(KeyError):
            airplane_geometry.get_face_group("nonexistent")

    def test_duplicate_group_name_raises(self, airplane_geometry):
        airplane_geometry.create_face_group("magenta", airplane_geometry.faces(colorRGB="255,0,255"))
        with pytest.raises(ValueError, match="already exists"):
            airplane_geometry.create_face_group("magenta", airplane_geometry.faces(colorRGB="0,0,255"))

    def test_clear_groups(self, airplane_geometry):
        airplane_geometry.create_face_group("magenta", airplane_geometry.faces(colorRGB="255,0,255"))
        airplane_geometry.create_face_group("blue", airplane_geometry.faces(colorRGB="0,0,255"))
        airplane_geometry.clear_groups()
        assert airplane_geometry.list_groups() == []

    def test_exclusive_face_ownership(self, airplane_geometry):
        """Assigning faces to a new group removes them from the old group."""
        all_group = airplane_geometry.create_face_group("all", airplane_geometry.faces())
        assert all_group.face_count() == 194

        airplane_geometry.create_face_group("magenta", airplane_geometry.faces(colorRGB="255,0,255"))
        assert all_group.face_count() == 154  # 194 - 40

        airplane_geometry.create_face_group("blue", airplane_geometry.faces(colorRGB="0,0,255"))
        assert all_group.face_count() == 117  # 154 - 37

    @pytest.mark.parametrize("color,expected_count", AIRPLANE_COLOR_EXPECTED)
    def test_group_face_count_per_color(self, airplane_geometry, color, expected_count):
        name = color if color else "uncolored"
        group = airplane_geometry.create_face_group(name, airplane_geometry.faces(colorRGB=color))
        assert group.face_count() == expected_count

    def test_group_all_colors(self, airplane_geometry):
        for color, _ in AIRPLANE_COLOR_EXPECTED:
            name = color if color else "uncolored"
            airplane_geometry.create_face_group(name, airplane_geometry.faces(colorRGB=color))
        assert len(airplane_geometry.list_groups()) == 7


# ================================================================
# save_groups_to_file and _build_face_grouping_config tests
# ================================================================


class TestSaveGroups:
    def test_build_face_grouping_config(self, airplane_geometry):
        airplane_geometry.create_face_group("magenta", airplane_geometry.faces(colorRGB="255,0,255"))
        airplane_geometry.create_face_group("blue", airplane_geometry.faces(colorRGB="0,0,255"))

        config = airplane_geometry._build_face_grouping_config()
        assert len(config) == 40 + 37  # 77 UUID entries
        assert set(config.values()) == {"magenta", "blue"}
        # Verify a specific face UUID maps to the correct group
        magenta_uuid = next(iter(airplane_geometry.faces(colorRGB="255,0,255"))).uuid
        blue_uuid = next(iter(airplane_geometry.faces(colorRGB="0,0,255"))).uuid
        assert config[magenta_uuid] == "magenta"
        assert config[blue_uuid] == "blue"

    def test_save_and_load(self, airplane_geometry, tmp_path):
        airplane_geometry.create_face_group("magenta", airplane_geometry.faces(colorRGB="255,0,255"))
        airplane_geometry.create_face_group("blue", airplane_geometry.faces(colorRGB="0,0,255"))
        airplane_geometry.create_face_group("green", airplane_geometry.faces(colorRGB="0,255,0"))

        output_path = str(tmp_path / "face_grouping.json")
        airplane_geometry.save_groups_to_file(output_path)
        with open(output_path, "r") as f:
            data = json.load(f)

        assert len(data) == 40 + 37 + 37  # 114 faces across 3 groups
        assert set(data.values()) == {"magenta", "blue", "green"}
        # Keys should be valid UUIDs (contain hyphens)
        for key in data:
            assert "-" in key


# ================================================================
# Set operation tests
# ================================================================


class TestSetOperations:
    def test_nodeset_union(self, airplane_geometry):
        magenta = airplane_geometry.faces(colorRGB="255,0,255")
        blue = airplane_geometry.faces(colorRGB="0,0,255")
        assert len(magenta | blue) == 77  # 40 + 37

    def test_nodeset_intersection(self, airplane_geometry):
        all_faces = airplane_geometry.faces()
        magenta = airplane_geometry.faces(colorRGB="255,0,255")
        assert len(all_faces & magenta) == 40

    def test_nodeset_intersection_disjoint(self, airplane_geometry):
        magenta = airplane_geometry.faces(colorRGB="255,0,255")
        blue = airplane_geometry.faces(colorRGB="0,0,255")
        assert len(magenta & blue) == 0

    def test_nodeset_difference(self, airplane_geometry):
        all_faces = airplane_geometry.faces()
        magenta = airplane_geometry.faces(colorRGB="255,0,255")
        assert len(all_faces - magenta) == 154

    def test_subtract_multiple_colors(self, airplane_geometry):
        remaining = (
            airplane_geometry.faces()
            - airplane_geometry.faces(colorRGB="255,0,255")
            - airplane_geometry.faces(colorRGB="0,0,255")
        )
        assert len(remaining) == 117  # 194 - 40 - 37

    def test_nodeset_subtract_face_group(self, airplane_geometry):
        group = airplane_geometry.create_face_group(
            "magenta", airplane_geometry.faces(colorRGB="255,0,255")
        )
        assert len(airplane_geometry.faces() - group) == 154

    def test_geometry_subtract_face_group(self, airplane_geometry):
        group = airplane_geometry.create_face_group(
            "magenta", airplane_geometry.faces(colorRGB="255,0,255")
        )
        assert len(airplane_geometry - group) == 154

    def test_geometry_subtract_nodeset(self, airplane_geometry):
        magenta = airplane_geometry.faces(colorRGB="255,0,255")
        assert len(airplane_geometry - magenta) == 154

    def test_nodeset_is_empty(self, airplane_geometry):
        empty = airplane_geometry.faces(colorRGB="999,999,999")
        assert empty.is_empty()
        assert not bool(empty)

        faces = airplane_geometry.faces()
        assert not faces.is_empty()
        assert bool(faces)

    def test_nodeset_contains(self, airplane_geometry):
        faces = airplane_geometry.faces()
        node = next(iter(faces))
        assert node in faces

    def test_nodeset_equality(self, airplane_geometry):
        faces1 = airplane_geometry.faces(colorRGB="255,0,255")
        faces2 = airplane_geometry.faces(colorRGB="255,0,255")
        assert faces1 == faces2


# ================================================================
# Filter pattern matching tests
# ================================================================


class TestFilterPatterns:
    def test_exact_match(self):
        assert matches_pattern("hello", "hello")
        assert matches_pattern("Hello", "hello")  # case insensitive
        assert not matches_pattern("hello", "world")

    def test_wildcard_star(self):
        assert matches_pattern("top_surface", "top*")
        assert matches_pattern("top_surface", "*surface")
        assert matches_pattern("top_surface", "*_*")
        assert not matches_pattern("top_surface", "bottom*")

    def test_wildcard_question(self):
        assert matches_pattern("abc", "a?c")
        assert not matches_pattern("abbc", "a?c")

    def test_none_value(self):
        assert not matches_pattern(None, "anything")

    def test_filter_by_type_glob(self, airplane_geometry):
        """Glob filter works through the tree navigation API."""
        result = airplane_geometry.descendants(type="Body*")
        # Body + 3 BodyCollection = 4
        assert len(result) == 4
