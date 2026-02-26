"""
tree_backend.py - NetworkX backend for geometry tree

Stores the geometry tree structure in a NetworkX DiGraph.
Provides low-level operations for tree traversal and querying.
"""

import json
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from .filters import get_face_uuid, is_face_node, matches_criteria


class TreeBackend:
    """
    NetworkX-based backend for storing and querying geometry tree.

    The tree is stored as a directed graph (DiGraph) where:
    - Nodes represent tree elements (ModelFile, PartDefinition, TopoFace, etc.)
    - Edges represent parent-child relationships (parent -> child)
    - Node attributes store metadata (name, type, colorRGB, material, etc.)
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.root_id: Optional[str] = None
        self._node_counter = 0

    def load_from_json(self, json_data: dict) -> str:
        """
        Load geometry tree from JSON dictionary into NetworkX graph.

        Args:
            json_data: Tree structure as dictionary

        Returns:
            Root node ID
        """
        self.graph.clear()
        self._node_counter = 0
        self.root_id = self._add_node_recursive(json_data, parent_id=None)
        return self.root_id

    def load_from_file(self, filepath: str) -> str:
        """
        Load geometry tree from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Root node ID
        """
        with open(filepath, "r") as f:
            json_data = json.load(f)
        return self.load_from_json(json_data)

    def _add_node_recursive(self, node_data: dict, parent_id: Optional[str]) -> str:
        """
        Recursively add nodes to the graph.

        Args:
            node_data: Node data dictionary
            parent_id: Parent node ID (None for root)

        Returns:
            Node ID of the added node
        """
        attributes = node_data.get("attributes", {})
        node_id = attributes.get("Flow360UUID")

        if node_id is None or node_id in self.graph:
            self._node_counter += 1
            node_id = f"node_{self._node_counter}"

        node_attrs = {
            "name": node_data.get("name", ""),
            "type": node_data.get("type", ""),
            "colorRGB": node_data.get("colorRGB", ""),
            "material": node_data.get("material", ""),
            "faceCount": node_data.get("faceCount"),
            "attributes": attributes,
        }

        self.graph.add_node(node_id, **node_attrs)

        if parent_id is not None:
            self.graph.add_edge(parent_id, node_id)

        for child_data in node_data.get("children", []):
            self._add_node_recursive(child_data, parent_id=node_id)

        return node_id

    def get_root(self) -> Optional[str]:
        """Get root node ID."""
        return self.root_id

    def get_node_attrs(self, node_id: str) -> Dict[str, Any]:
        """Get attributes of a node."""
        if node_id not in self.graph:
            return {}
        return dict(self.graph.nodes[node_id])

    def get_children(self, node_id: str) -> List[str]:
        """Get direct children of a node."""
        if node_id not in self.graph:
            return []
        return list(self.graph.successors(node_id))

    def get_parent(self, node_id: str) -> Optional[str]:
        """Get parent of a node."""
        if node_id not in self.graph:
            return None
        predecessors = list(self.graph.predecessors(node_id))
        return predecessors[0] if predecessors else None

    def get_descendants(self, node_id: str) -> Set[str]:
        """Get all descendants of a node (recursive children)."""
        if node_id not in self.graph:
            return set()
        return nx.descendants(self.graph, node_id)

    def get_ancestors(self, node_id: str) -> Set[str]:
        """Get all ancestors of a node (recursive parents)."""
        if node_id not in self.graph:
            return set()
        return nx.ancestors(self.graph, node_id)

    def get_siblings(self, node_id: str) -> Set[str]:
        """Get siblings of a node (same parent, excluding self)."""
        parent = self.get_parent(node_id)
        if parent is None:
            return set()
        children = set(self.get_children(parent))
        children.discard(node_id)
        return children

    def filter_nodes(self, node_ids: Set[str], **criteria) -> Set[str]:
        """
        Filter nodes by criteria.

        Args:
            node_ids: Set of node IDs to filter
            **criteria: Filter criteria (name, type, colorRGB, etc.)

        Returns:
            Set of matching node IDs
        """
        if not criteria:
            return node_ids

        result = set()
        for node_id in node_ids:
            attrs = self.get_node_attrs(node_id)
            if matches_criteria(attrs, criteria):
                result.add(node_id)
        return result

    def get_faces_in_nodes(self, node_ids: Set[str], **filters) -> Set[str]:
        """
        Get all face nodes within given nodes (including descendants).

        Args:
            node_ids: Set of node IDs to search within
            **filters: Additional filters for faces

        Returns:
            Set of face node IDs (with Flow360UUID)
        """
        face_ids = set()

        all_nodes = set()
        for node_id in node_ids:
            all_nodes.add(node_id)
            all_nodes.update(self.get_descendants(node_id))

        for node_id in all_nodes:
            attrs = self.get_node_attrs(node_id)
            if is_face_node(attrs):
                if filters and not matches_criteria(attrs, filters):
                    continue

                uuid = get_face_uuid(attrs)
                if uuid:
                    face_ids.add(uuid)

        return face_ids

    def get_all_faces(self) -> Set[str]:
        """Get all face UUIDs in the entire tree."""
        if self.root_id is None:
            return set()
        return self.get_faces_in_nodes({self.root_id})

    def get_all_nodes(self) -> Set[str]:
        """Get all node IDs in the tree."""
        return set(self.graph.nodes())

    def node_count(self) -> int:
        """Get total number of nodes."""
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        """Get total number of edges."""
        return self.graph.number_of_edges()
