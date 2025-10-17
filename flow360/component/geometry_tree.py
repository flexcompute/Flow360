"""
Tree-based geometry grouping functionality for Geometry models
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union


class NodeType(Enum):
    """Geometry tree node types"""

    ModelFile = "ModelFile"
    ProductOccurrence = "ProductOccurrence"
    PartDefinition = "PartDefinition"
    FRMFeatureBasedEntity = "FRMFeatureBasedEntity"
    FRMFeatureParameter = "FRMFeatureParameter"
    FRMFeature = "FRMFeature"
    FRMFeatureLinkedItem = "FRMFeatureLinkedItem"
    RiBrepModel = "RiBrepModel"
    RiSet = "RiSet"
    TopoConnex = "TopoConnex"
    TopoShell = "TopoShell"
    TopoFace = "TopoFace"


class TreeNode:
    """Represents a node in the Geometry hierarchy tree"""

    def __init__(
        self,
        node_type: str,
        name: str = "",
        attributes: Dict[str, str] = None,
        color: str = "",
        uuid: str = "",
        children: List[TreeNode] = None,
    ):
        self.type = node_type
        self.name = name
        self.attributes = attributes or {}
        self.color = color
        self.uuid = uuid
        self.children = children or []
        self.parent: Optional[TreeNode] = None

        # Set parent for children
        for child in self.children:
            child.parent = self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TreeNode:
        """Create TreeNode from dictionary"""
        children = [cls.from_dict(child) for child in data.get("children", [])]
        node = cls(
            node_type=data.get("type", ""),
            name=data.get("name", ""),
            attributes=data.get("attributes", {}),
            color=data.get("color", ""),
            uuid=data.get("UUID", ""),
            children=children,
        )
        return node

    def get_path(self) -> List[TreeNode]:
        """Get the path from root to this node"""
        path = []
        current = self
        while current is not None:
            path.insert(0, current)
            current = current.parent
        return path

    def get_all_faces(self) -> List[TreeNode]:
        """Get all TopoFace nodes under this node"""
        faces = []
        if self.type == "TopoFace":
            faces.append(self)
        for child in self.children:
            faces.extend(child.get_all_faces())
        return faces

    def find_nodes(self, filter_func: Callable[[TreeNode], bool]) -> List[TreeNode]:
        """Find all nodes matching the filter function"""
        matches = []
        if filter_func(self):
            matches.append(self)
        for child in self.children:
            matches.extend(child.find_nodes(filter_func))
        return matches

    def __repr__(self):
        return f"TreeNode(type={self.type}, name={self.name})"


class FilterExpression:
    """Base class for filter expressions"""

    def __call__(self, node: TreeNode) -> bool:
        raise NotImplementedError

    def __and__(self, other: FilterExpression) -> AndExpression:
        return AndExpression(self, other)

    def __or__(self, other: FilterExpression) -> OrExpression:
        return OrExpression(self, other)

    def __invert__(self) -> NotExpression:
        return NotExpression(self)


class AndExpression(FilterExpression):
    """Logical AND of two filter expressions"""

    def __init__(self, left: FilterExpression, right: FilterExpression):
        self.left = left
        self.right = right

    def __call__(self, node: TreeNode) -> bool:
        return self.left(node) and self.right(node)


class OrExpression(FilterExpression):
    """Logical OR of two filter expressions"""

    def __init__(self, left: FilterExpression, right: FilterExpression):
        self.left = left
        self.right = right

    def __call__(self, node: TreeNode) -> bool:
        return self.left(node) or self.right(node)


class NotExpression(FilterExpression):
    """Logical NOT of a filter expression"""

    def __init__(self, expr: FilterExpression):
        self.expr = expr

    def __call__(self, node: TreeNode) -> bool:
        return not self.expr(node)


class TypeQuery:
    """Query builder for node type"""

    def __eq__(self, node_type: Union[NodeType, str]) -> FilterExpression:
        if isinstance(node_type, NodeType):
            type_str = node_type.value
        else:
            type_str = node_type

        class TypeFilter(FilterExpression):
            def __call__(self, node: TreeNode) -> bool:
                return node.type == type_str

        return TypeFilter()


class NameQuery:
    """Query builder for node name"""

    def contains(self, substring: str) -> FilterExpression:
        class NameContainsFilter(FilterExpression):
            def __call__(self, node: TreeNode) -> bool:
                return substring.lower() in node.name.lower()

        return NameContainsFilter()

    def __eq__(self, name: str) -> FilterExpression:
        class NameEqualsFilter(FilterExpression):
            def __call__(self, node: TreeNode) -> bool:
                return node.name == name

        return NameEqualsFilter()

    def startswith(self, prefix: str) -> FilterExpression:
        class NameStartsWithFilter(FilterExpression):
            def __call__(self, node: TreeNode) -> bool:
                return node.name.startswith(prefix)

        return NameStartsWithFilter()

    def endswith(self, suffix: str) -> FilterExpression:
        class NameEndsWithFilter(FilterExpression):
            def __call__(self, node: TreeNode) -> bool:
                return node.name.endswith(suffix)

        return NameEndsWithFilter()


class AttributeQuery:
    """Query builder for node attributes"""

    def __init__(self, attr_key: str):
        self.attr_key = attr_key

    def __eq__(self, value: str) -> FilterExpression:
        class AttributeEqualsFilter(FilterExpression):
            def __init__(self, key: str, val: str):
                self.key = key
                self.val = val

            def __call__(self, node: TreeNode) -> bool:
                return node.attributes.get(self.key) == self.val

        return AttributeEqualsFilter(self.attr_key, value)

    def contains(self, substring: str) -> FilterExpression:
        class AttributeContainsFilter(FilterExpression):
            def __init__(self, key: str, substr: str):
                self.key = key
                self.substr = substr

            def __call__(self, node: TreeNode) -> bool:
                attr_value = node.attributes.get(self.key, "")
                return self.substr in attr_value

        return AttributeContainsFilter(self.attr_key, substring)


# Singleton query objects
Type = TypeQuery()
Name = NameQuery()


class GeometryTree:
    """Manages Geometry hierarchy tree and face grouping"""

    def __init__(self, tree_json_path: str):
        """
        Initialize geometry tree from JSON file

        Parameters
        ----------
        tree_json_path : str
            Path to the tree JSON file
        """
        with open(tree_json_path, "r", encoding="utf-8") as f:
            tree_data = json.load(f)

        self.root = TreeNode.from_dict(tree_data)
        self.all_faces = self.root.get_all_faces()
        self.face_to_group: Dict[str, str] = {}  # face_uuid -> group_name
        self.face_groups: Dict[str, List[TreeNode]] = {}  # group_name -> list of faces

    def create_face_group(
        self, name: str, filter_expr: FilterExpression
    ) -> List[TreeNode]:
        """
        Create a face group by filtering nodes and collecting their faces.
        If a face already belongs to another group, it will be reassigned to the new group.

        Parameters
        ----------
        name : str
            Name of the face group
        filter_expr : FilterExpression
            Filter expression to match nodes

        Returns
        -------
        List[TreeNode]
            List of faces in the created group
        """
        # Find matching nodes
        matching_nodes = self.root.find_nodes(lambda node: filter_expr(node))

        # Collect faces from matching nodes
        group_faces = []
        new_face_uuids = set()
        
        for node in matching_nodes:
            faces = node.get_all_faces()
            for face in faces:
                if face.uuid:
                    group_faces.append(face)
                    new_face_uuids.add(face.uuid)

        # Remove these faces from their previous groups
        for group_name, faces in list(self.face_groups.items()):
            if group_name != name:
                self.face_groups[group_name] = [
                    f for f in faces if f.uuid not in new_face_uuids
                ]

        # Update face-to-group mapping
        for uuid in new_face_uuids:
            self.face_to_group[uuid] = name

        # Store the group
        self.face_groups[name] = group_faces

        return group_faces

    def print_grouping_stats(self) -> None:
        """Print statistics about face grouping"""
        total_faces = len(self.all_faces)
        
        # Count faces currently in groups
        faces_in_groups = sum(len(faces) for faces in self.face_groups.values())

        print(f"\n=== Face Grouping Statistics ===")
        print(f"Total faces: {total_faces}")
        print(f"Faces in groups: {faces_in_groups}")
        print(f"\nFace groups ({len(self.face_groups)}):")
        for group_name, faces in self.face_groups.items():
            print(f"  - {group_name}: {len(faces)} faces")
        print("=" * 33)

    def get_face_group_names(self) -> List[str]:
        """Get list of all face group names"""
        return list(self.face_groups.keys())

    def get_faces_in_group(self, group_name: str) -> List[TreeNode]:
        """
        Get faces in a specific group

        Parameters
        ----------
        group_name : str
            Name of the face group

        Returns
        -------
        List[TreeNode]
            List of faces in the group
        """
        return self.face_groups.get(group_name, [])



