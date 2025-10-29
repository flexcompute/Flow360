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
    TopoFacePointer = "TopoFacePointer"  # References to TopoFace nodes


class TreeNode:
    """Represents a node in the Geometry hierarchy tree"""

    def __init__(
        self,
        node_type: NodeType,
        name: str = "",
        colorRGB: str = "",
        material: str = "",
        attributes: Dict[str, str] = {},
        children: List[TreeNode] = [],
    ):
        self.type = node_type
        self.name = name
        self.attributes = attributes
        self.colorRGB = colorRGB
        self.material = material
        self._children = children  # Renamed to avoid conflict with children() method
        self.parent: Optional[TreeNode] = None
        self.uuid = None
        if FLOW360_UUID_ATTRIBUTE_KEY in attributes:
            self.uuid = attributes[FLOW360_UUID_ATTRIBUTE_KEY]
        for child in self._children:
            child.parent = self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TreeNode:
        """
        Create TreeNode from dictionary
        
        Supports both old format (color) and new format (colorRGB) for backward compatibility
        """
        children = [cls.from_dict(child) for child in data.get("children", [])]
        
        node = cls(
            node_type=NodeType[data.get("type")],
            name=data.get("name", ""),
            colorRGB=data.get("colorRGB"),
            material=data.get("material", ""),
            attributes=data.get("attributes", {}),
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

    def get_uuid_to_face(self) -> Dict[str, TreeNode]:
        uuid_to_face = {}
        if self.type == NodeType.TopoFace:
            uuid_to_face[self.uuid] = self
        for child in self._children:
            uuid_to_face.update(child.get_uuid_to_face())
        return uuid_to_face

    def get_all_faces(self) -> List[TreeNode]:
        """
        Recursively collect all TopoFace and TopoFacePointer nodes in the subtree

        TopoFacePointer nodes are references to actual TopoFace nodes and are collected
        alongside TopoFace nodes. Both have Flow360UUID attributes that can be used
        for face grouping.

        Returns
        -------
        List[TreeNode]
            List of all TopoFace and TopoFacePointer nodes under this node
        """
        faces = []
        if self.type == NodeType.TopoFace or self.type == NodeType.TopoFacePointer:
            faces.append(self)
        for child in self._children:
            faces.extend(child.get_all_faces())
        return faces

    def search(
        self,
        type: Optional[NodeType] = None,
        name: Optional[str] = None,
        colorRGB: Optional[str] = None,
        material: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
    ) -> "TreeSearch":
        """
        Create a deferred search operation for nodes in the subtree matching the given criteria.

        This method returns a TreeSearch instance that captures the search criteria
        but does not execute the search immediately. The search is executed when
        the TreeSearch instance is used (e.g., in create_face_group()).

        Supports wildcard matching for name using '*' character.
        All criteria are ANDed together.

        Parameters
        ----------
        type : Optional[NodeType]
            Node type to match (e.g., NodeType.FRMFeature)
        name : Optional[str]
            Name pattern to match. Supports wildcards:
            - "*wing*" matches any name containing "wing"
            - "wing*" matches any name starting with "wing"
            - "*wing" matches any name ending with "wing"
            - "wing" matches exact name "wing"
        colorRGB : Optional[str]
            RGB color string to match (e.g., "255,0,0" for red)
        material : Optional[str]
            Material name to match. Supports wildcard matching like name parameter.
        attributes : Optional[Dict[str, str]]
            Dictionary of attribute key-value pairs to match

        Returns
        -------
        TreeSearch
            A TreeSearch instance that can be executed later

        Examples
        --------
        >>> # Create a search for FRMFeature nodes with "wing" in the name
        >>> wing_search = root.search(type=NodeType.FRMFeature, name="*wing*")
        >>>
        >>> # Pass to create_face_group (will execute internally)
        >>> geometry.create_face_group(name="wing", selection=wing_search)
        >>>
        >>> # Or execute manually to get nodes
        >>> wing_nodes = wing_search.execute()
        """
        return TreeSearch(
            node=self,
            type=type,
            name=name,
            colorRGB=colorRGB,
            material=material,
            attributes=attributes,
        )

    def children(
        self,
        type: Optional[NodeType] = None,
        name: Optional[str] = None,
        colorRGB: Optional[str] = None,
        material: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
    ) -> "NodeCollection":
        """
        Get children of this node, optionally filtered by exact criteria.
        
        This method filters only the direct children (not the entire subtree) using
        exact matching - no wildcards or patterns. It's designed for certain navigation
        like clicking through folders in a file system.
        
        For pattern matching and recursive search, use .search() instead.
        
        Returns a NodeCollection that supports further chaining via .children() calls.
        
        Parameters
        ----------
        type : Optional[NodeType]
            Node type to filter by (exact match, e.g., NodeType.FRMFeature)
        name : Optional[str]
            Exact name to match (no wildcards)
        colorRGB : Optional[str]
            Exact RGB color string to match
        material : Optional[str]
            Exact material name to match (no wildcards)
        attributes : Optional[Dict[str, str]]
            Dictionary of attribute key-value pairs to match (exact matches)
            
        Returns
        -------
        NodeCollection
            Collection of direct child nodes matching the criteria
            
        Examples
        --------
        >>> # Get all direct children
        >>> all_children = node.children()
        >>> 
        >>> # Get children of specific type (exact match)
        >>> features = node.children(type=NodeType.FRMFeature)
        >>> 
        >>> # Chain to navigate tree structure with certainty
        >>> result = root.children().children().children(
        ...     type=NodeType.FRMFeature, 
        ...     name="body_main"  # Exact name, no wildcards
        ... )
        """
        filtered_children = []
        
        for child in self._children:
            match = True
            
            # Exact type matching
            if type is not None:
                if child.type != type:
                    match = False
            
            # Exact name matching (no wildcards)
            if match and name is not None:
                if child.name != name:
                    match = False
            
            # Exact colorRGB matching
            if match and colorRGB is not None:
                if child.colorRGB != colorRGB:
                    match = False
            
            # Exact material matching (no wildcards)
            if match and material is not None:
                if child.material != material:
                    match = False
            
            # Exact attribute matching
            if match and attributes is not None:
                for key, value in attributes.items():
                    if child.attributes.get(key) != value:
                        match = False
                        break
            
            if match:
                filtered_children.append(child)
        
        return NodeCollection(filtered_children)

    def __repr__(self):
        return f"TreeNode(type={self.type.value}, name={self.name})"


class NodeCollection:
    """
    A collection of TreeNode objects that supports method chaining.
    
    This class wraps one or more TreeNode objects and provides a .children()
    method to enable fluent tree navigation patterns like:
    root.children().children().children(type=NodeType.FRMFeature)
    """
    
    def __init__(self, nodes: List[TreeNode]):
        """
        Initialize a NodeCollection with a list of nodes.
        
        Parameters
        ----------
        nodes : List[TreeNode]
            List of TreeNode objects to wrap
        """
        self._nodes = nodes if isinstance(nodes, list) else [nodes]
    
    @property
    def nodes(self) -> List[TreeNode]:
        """Get the list of nodes in this collection"""
        return self._nodes
    
    def children(
        self,
        type: Optional[NodeType] = None,
        name: Optional[str] = None,
        colorRGB: Optional[str] = None,
        material: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
    ) -> "NodeCollection":
        """
        Get all children from all nodes in this collection, optionally filtered by exact criteria.
        
        Uses exact matching only (no wildcards or patterns) for certain navigation.
        This enables chaining like: collection.children().children(type=...)
        
        For pattern matching, use .search() on individual nodes instead.
        
        Parameters
        ----------
        type : Optional[NodeType]
            Node type to filter by (exact match)
        name : Optional[str]
            Exact name to match (no wildcards)
        colorRGB : Optional[str]
            Exact RGB color string to match
        material : Optional[str]
            Exact material name to match (no wildcards)
        attributes : Optional[Dict[str, str]]
            Dictionary of attribute key-value pairs to match (exact matches)
            
        Returns
        -------
        NodeCollection
            New collection containing children from all nodes
        """
        all_children = []
        for node in self._nodes:
            # Use the TreeNode.children() method which handles exact matching
            child_collection = node.children(
                type=type, name=name, colorRGB=colorRGB, material=material, attributes=attributes
            )
            all_children.extend(child_collection.nodes)
        
        return NodeCollection(all_children)
    
    def __len__(self) -> int:
        """Return the number of nodes in this collection"""
        return len(self._nodes)
    
    def __iter__(self):
        """Make the collection iterable"""
        return iter(self._nodes)
    
    def __getitem__(self, index: int) -> TreeNode:
        """Allow indexing into the collection"""
        return self._nodes[index]
    
    def __repr__(self):
        return f"NodeCollection({len(self._nodes)} nodes)"


class TreeSearch:
    """
    Represents a deferred tree search operation.
    
    This class captures search criteria and the node from which to search,
    but does not execute the search until explicitly requested via execute().
    This allows for lazy evaluation and cleaner API usage.
    """

    def __init__(
        self,
        node: TreeNode,
        type: Optional[NodeType] = None,
        name: Optional[str] = None,
        colorRGB: Optional[str] = None,
        material: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize a TreeSearch with search criteria.

        Parameters
        ----------
        node : TreeNode
            The node from which to start the search (searches its subtree)
        type : Optional[NodeType]
            Node type to match (e.g., NodeType.FRMFeature)
        name : Optional[str]
            Name pattern to match. Supports wildcards (e.g., "*wing*")
        colorRGB : Optional[str]
            RGB color string to match (e.g., "255,0,0")
        material : Optional[str]
            Material name to match. Supports wildcards.
        attributes : Optional[Dict[str, str]]
            Dictionary of attribute key-value pairs to match
        """
        self.node = node
        self.type = type
        self.name = name
        self.colorRGB = colorRGB
        self.material = material
        self.attributes = attributes

    def execute(self) -> List[TreeNode]:
        """
        Execute the search and return matching nodes.

        Returns
        -------
        List[TreeNode]
            List of nodes matching the search criteria
        """
        import fnmatch

        matches = []

        def search_recursive(current_node: TreeNode):
            # Check if this node matches all criteria
            match = True

            if self.type is not None:
                if current_node.type != self.type:
                    match = False

            if match and self.name is not None:
                # Use fnmatch for wildcard matching (case-insensitive)
                if not fnmatch.fnmatch(current_node.name.lower(), self.name.lower()):
                    match = False

            if match and self.colorRGB is not None:
                if current_node.colorRGB != self.colorRGB:
                    match = False

            if match and self.material is not None:
                # Support wildcard matching for material
                if not fnmatch.fnmatch(current_node.material.lower(), self.material.lower()):
                    match = False

            if match and self.attributes is not None:
                for key, value in self.attributes.items():
                    if current_node.attributes.get(key) != value:
                        match = False
                        break

            if match:
                matches.append(current_node)

            # Recursively search children
            for child in current_node._children:
                search_recursive(child)

        search_recursive(self.node)
        return matches

    def __repr__(self):
        criteria = []
        if self.type is not None:
            criteria.append(f"type={self.type.value}")
        if self.name is not None:
            criteria.append(f"name='{self.name}'")
        if self.colorRGB is not None:
            criteria.append(f"colorRGB='{self.colorRGB}'")
        if self.material is not None:
            criteria.append(f"material='{self.material}'")
        if self.attributes is not None:
            criteria.append(f"attributes={self.attributes}")
        criteria_str = ", ".join(criteria)
        return f"TreeSearch({criteria_str})"


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

    @property
    def all_faces(self) -> List[TreeNode]:
        """
        Get all face nodes in the tree

        Returns
        -------
        List[TreeNode]
            List of all TopoFace nodes in the tree
        """
        return list(self.uuid_to_face.values())
