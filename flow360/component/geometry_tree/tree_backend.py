"""
tree_backend.py - NetworkX backend for geometry tree

Stores the geometry tree structure in a NetworkX DiGraph.
Provides low-level operations for tree traversal and querying.
"""

import json
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from .filters import is_face_node, matches_criteria
from .node_type import NodeType


class TreeBackend:
    """
    NetworkX-based backend for storing and querying geometry tree.

    The tree is stored as a directed graph (DiGraph) where:
    - Nodes represent tree elements (ModelFile, Assembly, Part, Face, etc.)
    - Edges represent parent-child relationships (parent -> child)
    - Node attributes store metadata (name, type, colorRGB, material, etc.)
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.root_id: Optional[str] = None

    def load_from_json(self, json_data: dict) -> str:
        """
        Load geometry tree from JSON dictionary into NetworkX graph.

        Args:
            json_data: Versioned tree structure {"version": "...", "tree": {...}}

        Returns:
            Root node ID
        """
        self.graph.clear()
        self.root_id = self._add_node_recursive(json_data["tree"], parent_id=None)
        return self.root_id

    def load_from_file(self, filepath: str) -> str:
        """
        Load geometry tree from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            Root node ID
        """
        with open(filepath, "r", encoding="utf-8") as f:
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
        node_id = attributes.get("_Flow360UUID")
        node_name = node_data.get("name", "<unnamed>")
        node_type = node_data.get("type", "<unknown>")

        if node_id is None:
            raise ValueError(
                f"Node '{node_name}' (type={node_type}) is missing " f"'_Flow360UUID' attribute."
            )
        if node_id in self.graph:
            raise ValueError(
                f"Duplicate _Flow360UUID '{node_id}' found on node "
                f"'{node_name}' (type={node_type})."
            )

        try:
            resolved_type = NodeType(node_type)
        except ValueError as exc:
            raise ValueError(
                f"Node '{node_name}' has unknown type '{node_type}'. "
                f"Valid types: {[t.value for t in NodeType]}"
            ) from exc

        node_attrs = {
            "name": node_data.get("name", ""),
            "type": resolved_type,
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

    def get_all_faces(self) -> Set[str]:
        """Get all face node IDs in the entire tree."""
        if self.root_id is None:
            return set()
        result = set()
        for node_id in self.graph.nodes():
            attrs = self.get_node_attrs(node_id)
            if is_face_node(attrs):
                result.add(node_id)
        return result

    def get_all_nodes(self) -> Set[str]:
        """Get all node IDs in the tree."""
        return set(self.graph.nodes())

    def node_count(self) -> int:
        """Get total number of nodes."""
        return self.graph.number_of_nodes()
