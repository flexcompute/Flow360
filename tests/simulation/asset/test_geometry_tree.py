"""
Tests for the geometry tree and face grouping API.

Covers:
- TreeBackend: versioned JSON loading
- Geometry.from_local_tree(): loading from local hierarchical metadata JSON
- Tree navigation: root_node, children, descendants, faces (with filters)
- Face group management: create_face_group, get_face_group, list_groups, clear_groups
- save_groups_to_file and _build_face_grouping_config: export to JSON
- Set operations on NodeSet and Geometry
- Node attribute access (including FaceGroup.face_ids)
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
CYLINDER_JSON_PATH = os.path.join(TREE_DATA_DIR, "cylinder_geometry_tree.json")
AIRPLANE_JSON_PATH = os.path.join(TREE_DATA_DIR, "airplane_rc_geometry_tree.json")


@pytest.fixture
def cylinder_json():
    with open(CYLINDER_JSON_PATH, "r") as f:
        return json.load(f)


@pytest.fixture
def cylinder_backend(cylinder_json):
    backend = TreeBackend()
    backend.load_from_json(cylinder_json)
    return backend


@pytest.fixture
def cylinder_geometry():
    return Geometry.from_local_tree(CYLINDER_JSON_PATH)


@pytest.fixture
def airplane_geometry():
    return Geometry.from_local_tree(AIRPLANE_JSON_PATH)


# ================================================================
# TreeBackend tests
# ================================================================


class TestTreeBackend:
    def test_load_from_json_versioned(self, cylinder_backend):
        assert cylinder_backend.get_root() is not None
        # ModelFile + 2 Assembly + Part + Body + ShellCollection + Shell + 3 Faces = 10
        assert cylinder_backend.node_count() == 10

    def test_load_from_json_requires_version_key(self):
        """Unversioned data (no 'tree' key) should raise KeyError."""
        backend = TreeBackend()
        raw_tree = {"name": "root", "type": "ModelFile", "children": []}
        with pytest.raises(KeyError):
            backend.load_from_json(raw_tree)

    def test_load_from_file(self):
        backend = TreeBackend()
        root_id = backend.load_from_file(CYLINDER_JSON_PATH)
        assert root_id is not None
        assert backend.node_count() == 10

    def test_root_node_attributes(self, cylinder_backend):
        root_attrs = cylinder_backend.get_node_attrs(cylinder_backend.get_root())
        assert root_attrs["name"] == "test_cylinder_colored"
        assert root_attrs["type"] == "ModelFile"

    def test_get_children(self, cylinder_backend):
        children = cylinder_backend.get_children(cylinder_backend.get_root())
        assert len(children) == 1  # single Assembly child

    def test_get_descendants(self, cylinder_backend):
        descendants = cylinder_backend.get_descendants(cylinder_backend.get_root())
        # 2 Assembly + Part + Body + ShellCollection + Shell + 3 Faces = 9
        assert len(descendants) == 9

    def test_get_parent(self, cylinder_backend):
        root = cylinder_backend.get_root()
        children = cylinder_backend.get_children(root)
        assert cylinder_backend.get_parent(children[0]) == root
        assert cylinder_backend.get_parent(root) is None

    def test_get_all_faces(self, cylinder_backend):
        faces = cylinder_backend.get_all_faces()
        assert len(faces) == 3

    def test_filter_nodes_by_type(self, cylinder_backend):
        all_nodes = cylinder_backend.get_all_nodes()
        faces = cylinder_backend.filter_nodes(all_nodes, type="Face")
        assert len(faces) == 3

    def test_nonexistent_node(self, cylinder_backend):
        assert cylinder_backend.get_node_attrs("nonexistent") == {}
        assert cylinder_backend.get_children("nonexistent") == []
        assert cylinder_backend.get_parent("nonexistent") is None
        assert cylinder_backend.get_descendants("nonexistent") == set()


# ================================================================
# Geometry.from_local_tree() tests
# ================================================================


class TestGeometryFromLocalTree:
    def test_loads_tree_with_correct_face_count(self, cylinder_geometry):
        assert len(cylinder_geometry.faces()) == 3

    def test_repr(self, cylinder_geometry):
        assert "3 faces" in repr(cylinder_geometry)


# ================================================================
# Tree navigation tests
# ================================================================


class TestTreeNavigation:
    def test_root_node(self, cylinder_geometry):
        root = cylinder_geometry.root_node()
        assert len(root) == 1
        node = next(iter(root))
        assert node.name == "test_cylinder_colored"
        assert node.type == "ModelFile"

    def test_children(self, cylinder_geometry):
        children = cylinder_geometry.children()
        assert len(children) == 1

    def test_descendants(self, cylinder_geometry):
        descendants = cylinder_geometry.descendants()
        assert len(descendants) == 9

    def test_descendants_with_type_filter(self, cylinder_geometry):
        assert len(cylinder_geometry.descendants(type="Part")) == 1
        assert len(cylinder_geometry.descendants(type="Shell")) == 1

    def test_faces_no_filter(self, cylinder_geometry):
        assert len(cylinder_geometry.faces()) == 3

    def test_faces_filter_by_color(self, cylinder_geometry):
        assert len(cylinder_geometry.faces(colorRGB="255,0,0")) == 1
        assert len(cylinder_geometry.faces(colorRGB="0,0,255")) == 1
        assert len(cylinder_geometry.faces(colorRGB="255,255,0")) == 1

    def test_faces_filter_by_name(self, cylinder_geometry):
        assert len(cylinder_geometry.faces(name="side_surface")) == 1

    def test_faces_filter_no_match(self, cylinder_geometry):
        assert len(cylinder_geometry.faces(colorRGB="999,999,999")) == 0

    def test_root_node_raises_without_backend(self):
        geo = Geometry(id=None)
        with pytest.raises(Flow360ValueError):
            geo.root_node()


# ================================================================
# Node attribute tests
# ================================================================


class TestNodeAttributes:
    def test_face_node_properties(self, cylinder_geometry):
        red_faces = cylinder_geometry.faces(colorRGB="255,0,0")
        node = next(iter(red_faces))
        assert node.name == "top_surface"
        assert node.type == "Face"
        assert node.color == "255,0,0"
        assert node.uuid is not None
        assert node.is_face()

    def test_non_face_node(self, cylinder_geometry):
        node = next(iter(cylinder_geometry.descendants(type="Part")))
        assert node.type == "Part"
        assert not node.is_face()

    def test_node_equality(self, cylinder_geometry):
        faces1 = list(cylinder_geometry.faces(colorRGB="255,0,0"))
        faces2 = list(cylinder_geometry.faces(colorRGB="255,0,0"))
        assert faces1[0] == faces2[0]

    def test_node_children_navigation(self, cylinder_geometry):
        shell = next(iter(cylinder_geometry.descendants(type="Shell")))
        children = shell.children()
        assert len(children) == 3
        for child in children:
            assert child.is_face()


# ================================================================
# Face group management tests
# ================================================================


class TestFaceGroupManagement:
    def test_create_face_group(self, cylinder_geometry):
        group = cylinder_geometry.create_face_group("top", cylinder_geometry.faces(colorRGB="255,0,0"))
        assert isinstance(group, FaceGroup)
        assert group.name == "top"
        assert group.face_count() == 1

    def test_face_group_face_ids_returns_copy(self, cylinder_geometry):
        group = cylinder_geometry.create_face_group("top", cylinder_geometry.faces(colorRGB="255,0,0"))
        ids = group.face_ids
        assert len(ids) == 1
        # Mutating the returned set should not affect the group
        ids.clear()
        assert group.face_count() == 1

    def test_list_groups(self, cylinder_geometry):
        cylinder_geometry.create_face_group("top", cylinder_geometry.faces(colorRGB="255,0,0"))
        cylinder_geometry.create_face_group("side", cylinder_geometry.faces(colorRGB="0,0,255"))
        assert cylinder_geometry.list_groups() == ["top", "side"]

    def test_get_face_group(self, cylinder_geometry):
        cylinder_geometry.create_face_group("top", cylinder_geometry.faces(colorRGB="255,0,0"))
        group = cylinder_geometry.get_face_group("top")
        assert group.name == "top"

    def test_get_face_group_not_found(self, cylinder_geometry):
        with pytest.raises(KeyError):
            cylinder_geometry.get_face_group("nonexistent")

    def test_duplicate_group_name_raises(self, cylinder_geometry):
        cylinder_geometry.create_face_group("top", cylinder_geometry.faces(colorRGB="255,0,0"))
        with pytest.raises(ValueError, match="already exists"):
            cylinder_geometry.create_face_group("top", cylinder_geometry.faces(colorRGB="0,0,255"))

    def test_clear_groups(self, cylinder_geometry):
        cylinder_geometry.create_face_group("top", cylinder_geometry.faces(colorRGB="255,0,0"))
        cylinder_geometry.create_face_group("side", cylinder_geometry.faces(colorRGB="0,0,255"))
        cylinder_geometry.clear_groups()
        assert cylinder_geometry.list_groups() == []

    def test_exclusive_face_ownership(self, cylinder_geometry):
        """A face reassigned to a new group is removed from the old one."""
        group_all = cylinder_geometry.create_face_group("all", cylinder_geometry.faces())
        assert group_all.face_count() == 3

        cylinder_geometry.create_face_group("top", cylinder_geometry.faces(colorRGB="255,0,0"))
        assert group_all.face_count() == 2


# ================================================================
# save_groups_to_file and _build_face_grouping_config tests
# ================================================================


class TestSaveGroups:
    def test_build_face_grouping_config(self, cylinder_geometry):
        cylinder_geometry.create_face_group("top", cylinder_geometry.faces(colorRGB="255,0,0"))
        cylinder_geometry.create_face_group("side", cylinder_geometry.faces(colorRGB="0,0,255"))

        config = cylinder_geometry._build_face_grouping_config()
        assert len(config) == 2
        assert set(config.values()) == {"top", "side"}
        # Keys are the _Flow360UUIDs from the tree data
        top_uuid = next(iter(cylinder_geometry.faces(colorRGB="255,0,0"))).uuid
        side_uuid = next(iter(cylinder_geometry.faces(colorRGB="0,0,255"))).uuid
        assert config[top_uuid] == "top"
        assert config[side_uuid] == "side"

    def test_save_and_load(self, cylinder_geometry, tmp_path):
        cylinder_geometry.create_face_group("top", cylinder_geometry.faces(colorRGB="255,0,0"))
        cylinder_geometry.create_face_group("side", cylinder_geometry.faces(colorRGB="0,0,255"))
        cylinder_geometry.create_face_group("bottom", cylinder_geometry.faces(colorRGB="255,255,0"))

        output_path = str(tmp_path / "face_grouping.json")
        cylinder_geometry.save_groups_to_file(output_path)
        with open(output_path, "r") as f:
            data = json.load(f)

        assert len(data) == 3
        assert set(data.values()) == {"top", "side", "bottom"}
        # Keys should be valid UUIDs (contain hyphens)
        for key in data:
            assert "-" in key


# ================================================================
# Set operation tests
# ================================================================


class TestSetOperations:
    def test_nodeset_union(self, cylinder_geometry):
        red = cylinder_geometry.faces(colorRGB="255,0,0")
        blue = cylinder_geometry.faces(colorRGB="0,0,255")
        assert len(red | blue) == 2

    def test_nodeset_intersection(self, cylinder_geometry):
        all_faces = cylinder_geometry.faces()
        red = cylinder_geometry.faces(colorRGB="255,0,0")
        assert len(all_faces & red) == 1

    def test_nodeset_difference(self, cylinder_geometry):
        all_faces = cylinder_geometry.faces()
        red = cylinder_geometry.faces(colorRGB="255,0,0")
        assert len(all_faces - red) == 2

    def test_nodeset_subtract_face_group(self, cylinder_geometry):
        group = cylinder_geometry.create_face_group("top", cylinder_geometry.faces(colorRGB="255,0,0"))
        assert len(cylinder_geometry.faces() - group) == 2

    def test_geometry_subtract_face_group(self, cylinder_geometry):
        group = cylinder_geometry.create_face_group("top", cylinder_geometry.faces(colorRGB="255,0,0"))
        assert len(cylinder_geometry - group) == 2

    def test_geometry_subtract_nodeset(self, cylinder_geometry):
        red = cylinder_geometry.faces(colorRGB="255,0,0")
        assert len(cylinder_geometry - red) == 2

    def test_nodeset_is_empty(self, cylinder_geometry):
        empty = cylinder_geometry.faces(colorRGB="999,999,999")
        assert empty.is_empty()
        assert not bool(empty)

        faces = cylinder_geometry.faces()
        assert not faces.is_empty()
        assert bool(faces)

    def test_nodeset_contains(self, cylinder_geometry):
        faces = cylinder_geometry.faces()
        node = next(iter(faces))
        assert node in faces

    def test_nodeset_equality(self, cylinder_geometry):
        faces1 = cylinder_geometry.faces(colorRGB="255,0,0")
        faces2 = cylinder_geometry.faces(colorRGB="255,0,0")
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

    def test_faces_with_name_glob(self, cylinder_geometry):
        """Glob filter works through the tree navigation API."""
        result = cylinder_geometry.faces(name="*surface*")
        assert len(result) == 3  # all three faces have "surface" in name


# ================================================================
# Airplane geometry tests (complex model: 194 faces, 7 colors)
# ================================================================


AIRPLANE_COLOR_EXPECTED = [
    ("255,0,255", 40),
    ("0,0,255", 37),
    ("0,255,0", 37),
    ("0,255,255", 27),
    ("255,255,0", 27),
    ("", 24),
    ("0,0,0", 2),
]


class TestAirplaneTreeStructure:
    """Verify tree loading and node counts on a complex airplane model."""

    def test_face_count(self, airplane_geometry):
        assert len(airplane_geometry.faces()) == 194

    def test_node_type_counts(self, airplane_geometry):
        assert len(airplane_geometry.descendants(type="Part")) == 1
        assert len(airplane_geometry.descendants(type="Body")) == 1
        assert len(airplane_geometry.descendants(type="Shell")) == 1
        assert len(airplane_geometry.descendants(type="Assembly")) == 2
        assert len(airplane_geometry.descendants(type="BodyCollection")) == 3

    def test_root_node(self, airplane_geometry):
        node = next(iter(airplane_geometry.root_node()))
        assert node.name == "Solid-Body-RC-Plane_v2024_colored"
        assert node.type == "ModelFile"

    def test_repr(self, airplane_geometry):
        assert "194 faces" in repr(airplane_geometry)


class TestAirplaneColorFiltering:
    """Filter faces by color on the airplane model (7 distinct colors)."""

    @pytest.mark.parametrize("color,expected_count", AIRPLANE_COLOR_EXPECTED)
    def test_face_count_per_color(self, airplane_geometry, color, expected_count):
        assert len(airplane_geometry.faces(colorRGB=color)) == expected_count

    def test_all_colors_sum_to_total(self, airplane_geometry):
        total = sum(
            len(airplane_geometry.faces(colorRGB=color))
            for color, _ in AIRPLANE_COLOR_EXPECTED
        )
        assert total == 194


class TestAirplaneFaceGrouping:
    """Face grouping on a complex model with many faces and colors."""

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

    def test_save_groups_to_file(self, airplane_geometry, tmp_path):
        airplane_geometry.create_face_group("magenta", airplane_geometry.faces(colorRGB="255,0,255"))
        airplane_geometry.create_face_group("blue", airplane_geometry.faces(colorRGB="0,0,255"))
        airplane_geometry.create_face_group("green", airplane_geometry.faces(colorRGB="0,255,0"))

        output_path = str(tmp_path / "face_grouping.json")
        airplane_geometry.save_groups_to_file(output_path)
        with open(output_path, "r") as f:
            data = json.load(f)

        assert len(data) == 40 + 37 + 37  # 114 faces across 3 groups
        assert set(data.values()) == {"magenta", "blue", "green"}

    def test_exclusive_ownership_across_groups(self, airplane_geometry):
        """Assigning faces to a new group removes them from the old group."""
        all_group = airplane_geometry.create_face_group("all", airplane_geometry.faces())
        assert all_group.face_count() == 194

        airplane_geometry.create_face_group("magenta", airplane_geometry.faces(colorRGB="255,0,255"))
        assert all_group.face_count() == 154  # 194 - 40

        airplane_geometry.create_face_group("blue", airplane_geometry.faces(colorRGB="0,0,255"))
        assert all_group.face_count() == 117  # 154 - 37


class TestAirplaneSetOperations:
    """Set operations on a complex model."""

    def test_union_of_color_groups(self, airplane_geometry):
        magenta = airplane_geometry.faces(colorRGB="255,0,255")
        blue = airplane_geometry.faces(colorRGB="0,0,255")
        assert len(magenta | blue) == 77  # 40 + 37

    def test_subtract_multiple_colors(self, airplane_geometry):
        remaining = airplane_geometry.faces() - airplane_geometry.faces(colorRGB="255,0,255") - airplane_geometry.faces(colorRGB="0,0,255")
        assert len(remaining) == 117  # 194 - 40 - 37

    def test_intersection_is_disjoint_for_different_colors(self, airplane_geometry):
        magenta = airplane_geometry.faces(colorRGB="255,0,255")
        blue = airplane_geometry.faces(colorRGB="0,0,255")
        assert len(magenta & blue) == 0

    def test_geometry_subtract_face_group(self, airplane_geometry):
        group = airplane_geometry.create_face_group("magenta", airplane_geometry.faces(colorRGB="255,0,255"))
        assert len(airplane_geometry - group) == 154


class TestAirplaneTreeNavigation:
    """Tree navigation on a complex model with deeper hierarchy."""

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

    def test_face_uuids_are_unique(self, airplane_geometry):
        uuids = [node.uuid for node in airplane_geometry.faces()]
        assert len(uuids) == len(set(uuids))
