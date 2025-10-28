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
    """
    Represents a node in the geometry hierarchy tree.
    Manages fundamental tree structure and traversal operations.
    """

    def __init__(
        self,
        node_type: NodeType,
        name: str = "",
        color: str = "",
        attributes: Dict[str, str] = None,
        children: List[TreeNode] = None,
    ):
        """
        Initialize a tree node
        
        Parameters
        ----------
        node_type : NodeType
            Type of this node
        name : str
            Name of this node
        color : str
            Color attribute of this node
        attributes : Dict[str, str]
            Additional attributes for this node
        children : List[TreeNode]
            Child nodes
        """
        self.type = node_type
        self.name = name
        self.color = color
        self.attributes = attributes if attributes is not None else {}
        self.children = children if children is not None else []
        self.parent: Optional[TreeNode] = None
        self.uuid = None
        
        if FLOW360_UUID_ATTRIBUTE_KEY in self.attributes:
            self.uuid = self.attributes[FLOW360_UUID_ATTRIBUTE_KEY]
        
        for child in self.children:
            child.parent = self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TreeNode:
        """
        Create TreeNode from dictionary
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation of the tree
            
        Returns
        -------
        TreeNode
            Root node of the created tree
        """
        children = [cls.from_dict(child) for child in data.get("children", [])]
        node = cls(
            node_type=NodeType[data.get("type")],
            name=data.get("name", ""),
            color=data.get("color", ""),
            attributes=data.get("attributes", {}),
            children=children,
        )
        return node

    def get_path(self) -> List[TreeNode]:
        """
        Get the path from root to this node
        
        Returns
        -------
        List[TreeNode]
            List of nodes from root to this node
        """
        path = []
        current = self
        while current is not None:
            path.insert(0, current)
            current = current.parent
        return path

    def find_nodes(self, filter_func: Callable[[TreeNode], bool]) -> List[TreeNode]:
        """
        Find all nodes in the subtree matching the filter function
        
        Parameters
        ----------
        filter_func : Callable
            Filter function to match nodes
            
        Returns
        -------
        List[TreeNode]
            List of matching nodes
        """
        matches = []
        if filter_func(self):
            matches.append(self)
        for child in self.children:
            matches.extend(child.find_nodes(filter_func))
        return matches

    def collect_faces_in_subtree(self) -> List[TreeNode]:
        """
        Collect all face nodes (TopoFace and TopoFacePointer) in the subtree rooted at this node
        
        Returns
        -------
        List[TreeNode]
            List of all TopoFace and TopoFacePointer nodes in this subtree
        """
        faces = []
        if self.type == NodeType.TopoFace or self.type == NodeType.TopoFacePointer:
            faces.append(self)
        for child in self.children:
            faces.extend(child.collect_faces_in_subtree())
        return faces

    def _build_uuid_to_face(self) -> Dict[str, TreeNode]:
        """
        Build UUID to face mapping for all TopoFace nodes in the subtree
        
        Returns
        -------
        Dict[str, TreeNode]
            Mapping from UUID to TopoFace nodes
        """
        uuid_to_face = {}
        if self.type == NodeType.TopoFace and self.uuid:
            uuid_to_face[self.uuid] = self
        for child in self.children:
            uuid_to_face.update(child._build_uuid_to_face())
        return uuid_to_face

    def __repr__(self):
        return f"TreeNode(type={self.type.value}, name={self.name})"


class GeometryTree:
    """
    Wrapper class that manages the root node and UUID to face mapping.
    This class provides the main interface for working with the geometry tree.
    """

    def __init__(self, root: TreeNode):
        """
        Initialize geometry tree with a root node
        
        Parameters
        ----------
        root : TreeNode
            Root node of the tree
        """
        self.root = root
        self.uuid_to_face: Dict[str, TreeNode] = self.root._build_uuid_to_face()

    @classmethod
    def from_json(cls, tree_json_path: str) -> GeometryTree:
        """
        Load geometry tree from JSON file
        
        Parameters
        ----------
        tree_json_path : str
            Path to the tree JSON file
            
        Returns
        -------
        GeometryTree
            GeometryTree instance with loaded root and UUID mapping
        """
        with open(tree_json_path, "r", encoding="utf-8") as f:
            tree_data = json.load(f)
        
        root = TreeNode.from_dict(tree_data)
        return cls(root)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GeometryTree:
        """
        Create GeometryTree from dictionary
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation of the tree
            
        Returns
        -------
        GeometryTree
            GeometryTree instance
        """
        root = TreeNode.from_dict(data)
        return cls(root)

    def resolve_node(self, node: TreeNode) -> TreeNode:
        """
        Resolve a node to its actual representation.
        If the node is a TopoFacePointer, resolve it to the actual TopoFace it points to.
        Otherwise, return the node itself.
        
        Parameters
        ----------
        node : TreeNode
            The node to resolve
            
        Returns
        -------
        TreeNode
            The resolved node (actual TopoFace if input was TopoFacePointer, otherwise the original node)
        """
        if node.type == NodeType.TopoFacePointer and node.uuid:
            # Resolve TopoFacePointer to actual TopoFace using UUID
            actual_face = self.uuid_to_face.get(node.uuid)
            if actual_face:
                return actual_face
            # If no matching TopoFace found, return the pointer itself
            return node
        return node

    def find_nodes(self, filter_expr: FilterExpression) -> List[TreeNode]:
        """
        Find all nodes matching the filter expression.
        
        TopoFacePointer nodes are resolved to their actual TopoFace counterparts
        before filter evaluation, so filters operate on the actual face properties.
        
        Parameters
        ----------
        filter_expr : FilterExpression
            Filter expression to match nodes
            
        Returns
        -------
        List[TreeNode]
            List of matching nodes
        """
        import pdb
        pdb.set_trace()
        def filter_with_resolution(node: TreeNode) -> bool:
            # Resolve the node (TopoFacePointer -> TopoFace) before applying filter
            resolved_node = self.resolve_node(node)
            return filter_expr(resolved_node)
        
        return self.root.find_nodes(filter_with_resolution)

    def get_all_faces(self) -> List[TreeNode]:
        """
        Get all TopoFace nodes in the tree (not including TopoFacePointer)
        
        Returns
        -------
        List[TreeNode]
            List of all TopoFace nodes in the tree
        """
        return list(self.uuid_to_face.values())

    def __repr__(self):
        return f"GeometryTree(root={self.root})"


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
            target_type = node_type
        else:
            target_type = NodeType[node_type]

        class TypeFilter(FilterExpression):
            def __call__(self, node: TreeNode) -> bool:
                return node.type == target_type

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

