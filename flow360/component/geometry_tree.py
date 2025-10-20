"""
Tree-based geometry grouping functionality for Geometry models
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

FLOW360_UUID_ATTRIBUTE_KEY = "Flow360UUID"

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
    TopoFacePointer = "TopoFacePointer"


class TreeNode:
    """Represents a node in the Geometry hierarchy tree"""

    def __init__(
        self,
        node_type: NodeType,
        name: str = "",
        color: str = "",
        attributes: Dict[str, str] = {},
        children: List[TreeNode] = [],
    ):
        self.type = node_type
        self.name = name
        self.attributes = attributes 
        self.color = color
        self.children = children
        self.parent: Optional[TreeNode] = None
        self.uuid = None
        if FLOW360_UUID_ATTRIBUTE_KEY in attributes:
            self.uuid = attributes[FLOW360_UUID_ATTRIBUTE_KEY]
        for child in self.children:
            child.parent = self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TreeNode:
        """Create TreeNode from dictionary"""
        children = [cls.from_dict(child) for child in data.get("children", [])]
        node = cls(
            node_type=NodeType[data.get("type")],
            name=data.get("name"),
            color=data.get("color"),
            attributes=data.get("attributes"),
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

    def find_nodes(self, filter_func: Callable[[TreeNode], bool]) -> List[TreeNode]:
        """Find all nodes matching the filter function"""
        matches = []
        if filter_func(self):
            matches.append(self)
        for child in self.children:
            matches.extend(child.find_nodes(filter_func))
        return matches

    def get_uuid_to_face(self) -> Dict[str, TreeNode]:
        uuid_to_face = {}
        if self.type == NodeType.TopoFace:
            uuid_to_face[self.uuid] = self
        for child in self.children:
            uuid_to_face.update(child.get_uuid_to_face())
        return uuid_to_face

    def __repr__(self):
        return f"TreeNode(type={self.type.value}, name={self.name})"


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
    """Pure tree structure representing Geometry hierarchy"""

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

        self.root: TreeNode = TreeNode.from_dict(tree_data)
        self.uuid_to_face = self.root.get_uuid_to_face()


    def find_nodes(self, filter_expr: FilterExpression) -> List[TreeNode]:
        """
        Find all nodes matching the filter expression

        Parameters
        ----------
        filter_expr : FilterExpression
            Filter expression to match nodes

        Returns
        -------
        List[TreeNode]
            List of matching nodes
        """
        return self.root.find_nodes(lambda node: filter_expr(node))

    def get_all_faces(self) -> List[TreeNode]:
        return list(self.uuid_to_face.values())

