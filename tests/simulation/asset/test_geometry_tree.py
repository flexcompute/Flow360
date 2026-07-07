"""
Tests for the geometry tree and face grouping API.
"""

import json
import os

import pytest

from flow360.component.geometry import Geometry
from flow360.component.geometry_tree import NodeType, TreeBackend
from flow360.component.geometry_tree.face_group import FaceGroup
from flow360.component.geometry_tree.filters import matches_pattern
from flow360.exceptions import Flow360ValueError

TREE_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data/geometry_tree")
AIRPLANE_JSON_PATH = os.path.join(TREE_DATA_DIR, "airplane_rc_geometry_tree.json")
AIRPLANE_TOTAL_FACES = 194

DRIVAER_JSON_PATH = os.path.join(TREE_DATA_DIR, "drivaer_geometry_tree.json")
DRIVAER_TOTAL_FACES = 74
DRIVAER_TOTAL_NODES = 233

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


@pytest.fixture
def drivaer_tree():
    with open(DRIVAER_JSON_PATH, "r") as f:
        json_data = json.load(f)
    tree = TreeBackend()
    tree.load_from_json(json_data)
    return tree


@pytest.fixture
def drivaer_geometry():
    return Geometry.from_local_tree(DRIVAER_JSON_PATH)


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

    def test_load_from_json_missing_uuid(self):
        """Nodes without _Flow360UUID should raise ValueError."""
        backend = TreeBackend()
        tree_data = {
            "version": "1.0",
            "tree": {"name": "root", "type": "ModelFile", "children": []},
        }
        with pytest.raises(ValueError, match="missing.*_Flow360UUID"):
            backend.load_from_json(tree_data)

    def test_load_from_json_duplicate_uuid(self):
        """Duplicate _Flow360UUID should raise ValueError."""
        backend = TreeBackend()
        tree_data = {
            "version": "1.0",
            "tree": {
                "name": "root",
                "type": "ModelFile",
                "attributes": {"_Flow360UUID": "same-uuid"},
                "children": [
                    {
                        "name": "child",
                        "type": "Face",
                        "attributes": {"_Flow360UUID": "same-uuid"},
                    }
                ],
            },
        }
        with pytest.raises(ValueError, match="Duplicate.*_Flow360UUID"):
            backend.load_from_json(tree_data)

    def test_load_from_json_unknown_type(self):
        """Unknown node type should raise ValueError."""
        backend = TreeBackend()
        tree_data = {
            "version": "1.0",
            "tree": {
                "name": "root",
                "type": "UnknownType",
                "attributes": {"_Flow360UUID": "uuid-1"},
            },
        }
        with pytest.raises(ValueError, match="unknown type.*UnknownType"):
            backend.load_from_json(tree_data)

    def test_load_from_file(self):
        backend = TreeBackend()
        root_id = backend.load_from_file(AIRPLANE_JSON_PATH)
        assert root_id is not None
        assert backend.node_count() == 204

    def test_root_node_attributes(self, airplane_tree):
        root_attrs = airplane_tree.get_node_attrs(airplane_tree.get_root())
        assert root_attrs["name"] == "Solid-Body-RC-Plane_v2024_colored"
        assert root_attrs["type"] == NodeType.MODEL_FILE

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
        assert len(airplane_tree.get_all_faces()) == AIRPLANE_TOTAL_FACES

    def test_filter_nodes_by_type(self, airplane_tree):
        all_nodes = airplane_tree.get_all_nodes()
        assert (
            len(airplane_tree.filter_nodes(all_nodes, type=NodeType.FACE)) == AIRPLANE_TOTAL_FACES
        )
        assert len(airplane_tree.filter_nodes(all_nodes, type=NodeType.SHELL)) == 1
        assert len(airplane_tree.filter_nodes(all_nodes, type=NodeType.BODY_COLLECTION)) == 3

    def test_nonexistent_node(self, airplane_tree):
        assert airplane_tree.get_node_attrs("nonexistent") == {}
        assert airplane_tree.get_children("nonexistent") == []
        assert airplane_tree.get_parent("nonexistent") is None
        assert airplane_tree.get_descendants("nonexistent") == set()


# ================================================================
# Tree navigation tests
# ================================================================


class TestTreeNavigation:
    def test_root_node(self, airplane_geometry):
        root = airplane_geometry.root_node()
        assert root.name == "Solid-Body-RC-Plane_v2024_colored"
        assert root.type == NodeType.MODEL_FILE

    def test_children(self, airplane_geometry):
        children = airplane_geometry.children()
        assert len(children) == 1

    def test_descendants(self, airplane_geometry):
        assert len(airplane_geometry.descendants()) == 203

    def test_descendants_with_type_filter(self, airplane_geometry):
        assert len(airplane_geometry.descendants(type=NodeType.FACE)) == AIRPLANE_TOTAL_FACES
        assert len(airplane_geometry.descendants(type=NodeType.SHELL_COLLECTION)) == 1
        assert len(airplane_geometry.descendants(type=NodeType.PART)) == 1
        assert len(airplane_geometry.descendants(type=NodeType.BODY)) == 1
        assert len(airplane_geometry.descendants(type=NodeType.SHELL)) == 1
        assert len(airplane_geometry.descendants(type=NodeType.ASSEMBLY)) == 2
        assert len(airplane_geometry.descendants(type=NodeType.BODY_COLLECTION)) == 3

    def test_faces_no_filter(self, airplane_geometry):
        assert len(airplane_geometry.faces()) == AIRPLANE_TOTAL_FACES

    @pytest.mark.parametrize("color,expected_count", AIRPLANE_COLOR_EXPECTED)
    def test_faces_filter_by_color(self, airplane_geometry, color, expected_count):
        assert len(airplane_geometry.faces(colorRGB=color)) == expected_count

    def test_faces_filter_no_match(self, airplane_geometry):
        assert len(airplane_geometry.faces(colorRGB="999,999,999")) == 0

    def test_all_color_filters_sum_to_total(self, airplane_geometry):
        total = sum(
            len(airplane_geometry.faces(colorRGB=color)) for color, _ in AIRPLANE_COLOR_EXPECTED
        )
        assert total == AIRPLANE_TOTAL_FACES


# ================================================================
# Node attribute tests
# ================================================================


class TestNodeAttributes:
    def test_face_node_properties(self, airplane_geometry):
        magenta_faces = airplane_geometry.faces(colorRGB="255,0,255")
        node = next(iter(magenta_faces))
        assert node.type == NodeType.FACE
        assert node.color == "255,0,255"
        assert node.is_face()

    def test_non_face_node(self, airplane_geometry):
        node = next(iter(airplane_geometry.descendants(type=NodeType.PART)))
        assert node.type == NodeType.PART
        assert not node.is_face()

    def test_node_equality(self, airplane_geometry):
        faces1 = airplane_geometry.faces(colorRGB="0,0,0")
        faces2 = airplane_geometry.faces(colorRGB="0,0,0")
        assert len(faces1) == 2
        # Same query yields equal NodeSets
        assert faces1 == faces2


# ================================================================
# Face group management tests
# ================================================================


class TestFaceGroupManagement:
    def test_create_face_group(self, airplane_geometry):
        group = airplane_geometry.create_face_group(
            name="magenta", selection=airplane_geometry.faces(colorRGB="255,0,255")
        )
        assert isinstance(group, FaceGroup)
        assert group.name == "magenta"
        assert group.face_count() == 40

    def test_list_groups(self, airplane_geometry):
        airplane_geometry.create_face_group(
            name="magenta", selection=airplane_geometry.faces(colorRGB="255,0,255")
        )
        airplane_geometry.create_face_group(
            name="blue", selection=airplane_geometry.faces(colorRGB="0,0,255")
        )
        assert airplane_geometry.list_groups() == ["magenta", "blue"]

    def test_get_face_group(self, airplane_geometry):
        airplane_geometry.create_face_group(
            name="green", selection=airplane_geometry.faces(colorRGB="0,255,0")
        )
        group = airplane_geometry.get_face_group("green")
        assert group.name == "green"
        assert group.face_count() == 37

    def test_get_face_group_not_found(self, airplane_geometry):
        with pytest.raises(KeyError):
            airplane_geometry.get_face_group("nonexistent")

    def test_duplicate_group_name_raises(self, airplane_geometry):
        airplane_geometry.create_face_group(
            name="magenta", selection=airplane_geometry.faces(colorRGB="255,0,255")
        )
        with pytest.raises(ValueError, match="already exists"):
            airplane_geometry.create_face_group(
                name="magenta", selection=airplane_geometry.faces(colorRGB="0,0,255")
            )

    def test_clear_groups(self, airplane_geometry):
        airplane_geometry.create_face_group(
            name="magenta", selection=airplane_geometry.faces(colorRGB="255,0,255")
        )
        airplane_geometry.create_face_group(
            name="blue", selection=airplane_geometry.faces(colorRGB="0,0,255")
        )
        airplane_geometry.clear_groups()
        assert airplane_geometry.list_groups() == []

    def test_exclusive_face_ownership(self, airplane_geometry):
        """Assigning faces to a new group removes them from the old group."""
        all_group = airplane_geometry.create_face_group(
            name="all", selection=airplane_geometry.faces()
        )
        assert all_group.face_count() == AIRPLANE_TOTAL_FACES

        airplane_geometry.create_face_group(
            name="magenta", selection=airplane_geometry.faces(colorRGB="255,0,255")
        )
        assert all_group.face_count() == AIRPLANE_TOTAL_FACES - 40

        airplane_geometry.create_face_group(
            name="blue", selection=airplane_geometry.faces(colorRGB="0,0,255")
        )
        assert all_group.face_count() == AIRPLANE_TOTAL_FACES - 40 - 37

    @pytest.mark.parametrize("color,expected_count", AIRPLANE_COLOR_EXPECTED)
    def test_group_face_count_per_color(self, airplane_geometry, color, expected_count):
        name = color if color else "uncolored"
        group = airplane_geometry.create_face_group(
            name=name, selection=airplane_geometry.faces(colorRGB=color)
        )
        assert group.face_count() == expected_count

    def test_group_all_colors(self, airplane_geometry):
        for color, _ in AIRPLANE_COLOR_EXPECTED:
            name = color if color else "uncolored"
            airplane_geometry.create_face_group(
                name=name, selection=airplane_geometry.faces(colorRGB=color)
            )
        assert len(airplane_geometry.list_groups()) == 7


# ================================================================
# export_face_grouping_config and _build_face_grouping_config tests
# ================================================================


class TestSaveGroups:
    def test_build_face_grouping_config(self, airplane_geometry):
        airplane_geometry.create_face_group(
            name="magenta", selection=airplane_geometry.faces(colorRGB="255,0,255")
        )
        airplane_geometry.create_face_group(
            name="blue", selection=airplane_geometry.faces(colorRGB="0,0,255")
        )

        config = airplane_geometry._build_face_grouping_config()
        assert config["version"] == "1.0"
        mapping = config["face_group_mapping"]
        assert len(mapping) == 40 + 37  # 77 UUID entries
        assert set(mapping.values()) == {"magenta", "blue"}

    def test_save_and_load(self, airplane_geometry, tmp_path):
        airplane_geometry.create_face_group(
            name="magenta", selection=airplane_geometry.faces(colorRGB="255,0,255")
        )
        airplane_geometry.create_face_group(
            name="blue", selection=airplane_geometry.faces(colorRGB="0,0,255")
        )
        airplane_geometry.create_face_group(
            name="green", selection=airplane_geometry.faces(colorRGB="0,255,0")
        )

        output_path = str(tmp_path / "face_grouping.json")
        airplane_geometry.export_face_grouping_config(output_path)
        with open(output_path, "r") as f:
            data = json.load(f)

        assert data["version"] == "1.0"
        mapping = data["face_group_mapping"]
        assert len(mapping) == 40 + 37 + 37  # 114 faces across 3 groups
        assert set(mapping.values()) == {"magenta", "blue", "green"}
        # Keys should be valid UUIDs (contain hyphens)
        for key in mapping:
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
        assert len(all_faces - magenta) == AIRPLANE_TOTAL_FACES - 40

    def test_subtract_multiple_colors(self, airplane_geometry):
        remaining = (
            airplane_geometry.faces()
            - airplane_geometry.faces(colorRGB="255,0,255")
            - airplane_geometry.faces(colorRGB="0,0,255")
        )
        assert len(remaining) == AIRPLANE_TOTAL_FACES - 40 - 37

    def test_nodeset_subtract_face_group(self, airplane_geometry):
        group = airplane_geometry.create_face_group(
            name="magenta", selection=airplane_geometry.faces(colorRGB="255,0,255")
        )
        assert len(airplane_geometry.faces() - group) == AIRPLANE_TOTAL_FACES - 40

    def test_geometry_subtract_face_group(self, airplane_geometry):
        group = airplane_geometry.create_face_group(
            name="magenta", selection=airplane_geometry.faces(colorRGB="255,0,255")
        )
        assert len(airplane_geometry - group) == AIRPLANE_TOTAL_FACES - 40

    def test_geometry_subtract_nodeset_raises(self, airplane_geometry):
        magenta = airplane_geometry.faces(colorRGB="255,0,255")
        with pytest.raises(Flow360ValueError):
            _ = airplane_geometry - magenta

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
# DrivAer geometry tests (large model: 9645 faces, 6240 bodies)
# ================================================================


class TestDrivAer:
    def test_tree_loading(self, drivaer_tree):
        assert drivaer_tree.node_count() == DRIVAER_TOTAL_NODES
        assert len(drivaer_tree.get_all_faces()) == DRIVAER_TOTAL_FACES

    def test_root_node(self, drivaer_geometry):
        root = drivaer_geometry.root_node()
        assert root.name == "ANSA_Assembled_DrivAer_Input"
        assert root.type == NodeType.MODEL_FILE

    def test_faces(self, drivaer_geometry):
        assert len(drivaer_geometry.faces()) == DRIVAER_TOTAL_FACES

    def test_type_counts(self, drivaer_geometry):
        assert len(drivaer_geometry.descendants(type=NodeType.FACE)) == DRIVAER_TOTAL_FACES
        assert len(drivaer_geometry.descendants(type=NodeType.BODY)) == 38
        assert len(drivaer_geometry.descendants(type=NodeType.SHELL)) == 38
        assert len(drivaer_geometry.descendants(type=NodeType.SHELL_COLLECTION)) == 38
        assert len(drivaer_geometry.descendants(type=NodeType.PART)) == 15
        assert len(drivaer_geometry.descendants(type=NodeType.ASSEMBLY)) == 29

    def test_all_faces_uncolored(self, drivaer_geometry):
        """DrivAer model has no color assignments."""
        assert len(drivaer_geometry.faces(colorRGB="")) == DRIVAER_TOTAL_FACES

    def test_face_grouping_by_assembly_name(self, drivaer_geometry):
        """Group faces by Assembly name and verify counts."""
        expected_faces = {
            "09_engineAndGearbox": 3,
            "21_Chassis": 9,
            "22_EngineBaySeals": 3,
            "Body": 3,
            "ExhaustSystem_new": 3,
            "Mirrors": 6,
            "RearEnd_Sedan": 3,
            "Slip_Ground": 4,
            "Underbody_new": 4,
            "WT_Sides": 7,
            "WT_ground_NoSlip": 2,
            "Wheels_Front": 9,
            "Wheels_Rear": 9,
            "tyre_plinth": 9,
        }

        for name, count in expected_faces.items():
            faces = drivaer_geometry.descendants(type=NodeType.ASSEMBLY, name=name).faces()
            group = drivaer_geometry.create_face_group(name=name, selection=faces)
            assert group.face_count() == count

        assert len(drivaer_geometry.list_groups()) == 14

        # Subtracting all groups from geometry should leave 0 faces
        remaining = drivaer_geometry
        for name in expected_faces:
            remaining = remaining - drivaer_geometry.get_face_group(name)
        assert len(remaining) == 0

    def test_face_grouping_by_children_navigation(self, drivaer_geometry):
        """Group faces using chained .children() navigation."""
        expected = {
            "Body": 3,
            "Wheels_Front": 9,
            "Wheels_Rear": 9,
            "Underbody_new": 4,
            "Mirrors": 6,
        }

        for name, count in expected.items():
            group = drivaer_geometry.create_face_group(
                name=name,
                selection=drivaer_geometry.children().children(name=name),
            )
            assert group.face_count() == count

        # Subtracting these groups leaves the remaining faces
        remaining = drivaer_geometry
        for name in expected:
            remaining = remaining - drivaer_geometry.get_face_group(name)
        assert len(remaining) == DRIVAER_TOTAL_FACES - sum(expected.values())


# ================================================================
# NodeType enum tests
# ================================================================


class TestNodeType:
    def test_str_enum_value(self):
        """NodeType values are the strings exported by C++."""
        assert NodeType.MODEL_FILE == "ModelFile"
        assert NodeType.FACE == "Face"
        assert NodeType.BODY_COLLECTION == "BodyCollection"

    def test_unknown_type_rejected(self):
        """Unknown type string raises ValueError on load."""
        backend = TreeBackend()
        tree_data = {
            "version": "1.0",
            "tree": {
                "name": "root",
                "type": "Bogus",
                "attributes": {"_Flow360UUID": "uuid-1"},
            },
        }
        with pytest.raises(ValueError, match="unknown type"):
            backend.load_from_json(tree_data)


# ================================================================
# Filter pattern matching tests (for non-type attributes)
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

    def test_filter_by_name_glob(self, airplane_geometry):
        """Glob filter works for name attribute."""
        result = airplane_geometry.descendants(name="*Default*")
        assert len(result) > 0
