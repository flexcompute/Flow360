"""
tree_backend.py - Plain-dict backend for geometry tree

Stores the geometry tree structure using three dictionaries instead of NetworkX.
Provides low-level operations for tree traversal and querying.
"""

from typing import Any, Dict, List, Optional, Set

from .filters import matches_criteria


class TreeBackend:
    """
    Dict-based backend for storing and querying geometry tree.

    The tree is stored using three dictionaries:
    - _nodes: node_id -> attribute dict
    - _children: node_id -> [child_ids]
    - _parent: node_id -> parent_id
    """

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._children: Dict[str, List[str]] = {}
        self._parent: Dict[str, str] = {}
        self.root_id: Optional[str] = None
        self._node_counter = 0

    def load_from_json(self, json_data: dict) -> str:
        """
        Load geometry tree from JSON dictionary.

        Args:
            json_data: Tree structure as dictionary

        Returns:
            Root node ID
        """
        self._nodes.clear()
        self._children.clear()
        self._parent.clear()
        self._node_counter = 0
        self.root_id = self._add_node_recursive(json_data, parent_id=None)
        return self.root_id

    def _add_node_recursive(self, node_data: dict, parent_id: Optional[str]) -> str:
        """Recursively add nodes to the tree."""
        attributes = node_data.get("attributes", {})
        node_id = attributes.get("Flow360UUID")

        if node_id is None or node_id in self._nodes:
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

        self._nodes[node_id] = node_attrs
        self._children[node_id] = []

        if parent_id is not None:
            self._parent[node_id] = parent_id
            self._children[parent_id].append(node_id)

        for child_data in node_data.get("children", []):
            self._add_node_recursive(child_data, parent_id=node_id)

        return node_id

    def get_root(self) -> Optional[str]:
        """Get root node ID."""
        return self.root_id

    def get_node_attrs(self, node_id: str) -> Dict[str, Any]:
        """Get attributes of a node."""
        if node_id not in self._nodes:
            return {}
        return dict(self._nodes[node_id])

    def get_children(self, node_id: str) -> List[str]:
        """Get direct children of a node."""
        return list(self._children.get(node_id, []))

    def get_parent(self, node_id: str) -> Optional[str]:
        """Get parent of a node."""
        return self._parent.get(node_id)

    def get_descendants(self, node_id: str) -> Set[str]:
        """Get all descendants of a node (BFS traversal)."""
        if node_id not in self._nodes:
            return set()
        result = set()
        stack = list(self._children.get(node_id, []))
        while stack:
            current = stack.pop()
            if current not in result:
                result.add(current)
                stack.extend(self._children.get(current, []))
        return result

    def filter_nodes(self, node_ids: Set[str], **criteria) -> Set[str]:
        """Filter nodes by criteria."""
        if not criteria:
            return node_ids
        result = set()
        for node_id in node_ids:
            attrs = self.get_node_attrs(node_id)
            if matches_criteria(attrs, criteria):
                result.add(node_id)
        return result
