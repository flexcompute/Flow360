"""
node_set.py - NodeSet class for tree navigation

A NodeSet represents a set of nodes at the current navigation scope.
Supports method chaining for fluent tree traversal and set operations.
"""

from typing import Iterator, Set

from .filters import is_face_node, matches_criteria
from .node import Node


class NodeSet:
    """
    A set of nodes at the current navigation scope.

    Supports fluent navigation through method chaining and set operations.
    """

    def __init__(self, geometry, backend, node_ids: Set[str]):
        self._geometry = geometry
        self._backend = backend
        self._node_ids = node_ids.copy()

    # ================================================================
    # Navigation Methods
    # ================================================================

    def children(self, **filters) -> "NodeSet":
        """Get direct children of all nodes in this set."""
        child_ids = set()
        for node_id in self._node_ids:
            child_ids.update(self._backend.get_children(node_id))
        if filters:
            child_ids = self._backend.filter_nodes(child_ids, **filters)
        return NodeSet(self._geometry, self._backend, child_ids)

    def descendants(self, **filters) -> "NodeSet":
        """Get all descendants of all nodes in this set."""
        descendant_ids = set()
        for node_id in self._node_ids:
            descendant_ids.update(self._backend.get_descendants(node_id))
        if filters:
            descendant_ids = self._backend.filter_nodes(descendant_ids, **filters)
        return NodeSet(self._geometry, self._backend, descendant_ids)

    def faces(self, **filters) -> "NodeSet":
        """Get all face nodes within this node scope."""
        all_nodes = set()
        for node_id in self._node_ids:
            all_nodes.add(node_id)
            all_nodes.update(self._backend.get_descendants(node_id))

        face_node_ids = set()
        for node_id in all_nodes:
            attrs = self._backend.get_node_attrs(node_id)
            if is_face_node(attrs):
                if filters:
                    if not matches_criteria(attrs, filters):
                        continue
                face_node_ids.add(node_id)

        return NodeSet(self._geometry, self._backend, face_node_ids)

    # ================================================================
    # Set Operations
    # ================================================================

    def __or__(self, other: "NodeSet") -> "NodeSet":
        """Union of two NodeSets."""
        if not isinstance(other, NodeSet):
            return NotImplemented
        return NodeSet(self._geometry, self._backend, self._node_ids | other._node_ids)

    def __and__(self, other: "NodeSet") -> "NodeSet":
        """Intersection of two NodeSets."""
        if not isinstance(other, NodeSet):
            return NotImplemented
        return NodeSet(self._geometry, self._backend, self._node_ids & other._node_ids)

    def __sub__(self, other) -> "NodeSet":
        """Difference: supports NodeSet and FaceGroup."""
        from .face_group import FaceGroup

        if isinstance(other, NodeSet):
            return NodeSet(self._geometry, self._backend, self._node_ids - other._node_ids)
        elif isinstance(other, FaceGroup):
            return NodeSet(self._geometry, self._backend, self._node_ids - other._node_ids)
        else:
            return NotImplemented

    # ================================================================
    # Collection Methods
    # ================================================================

    def is_empty(self) -> bool:
        """Check if NodeSet is empty."""
        return len(self._node_ids) == 0

    def __len__(self) -> int:
        return len(self._node_ids)

    def __iter__(self) -> Iterator[Node]:
        for node_id in self._node_ids:
            yield Node(self._geometry, self._backend, node_id)

    def __contains__(self, item) -> bool:
        if isinstance(item, Node):
            return item.node_id in self._node_ids
        return item in self._node_ids

    def __bool__(self) -> bool:
        return len(self._node_ids) > 0

    def __eq__(self, other) -> bool:
        if not isinstance(other, NodeSet):
            return False
        return self._node_ids == other._node_ids

    def __hash__(self):
        return None

    def __repr__(self) -> str:
        if not self._node_ids:
            return "NodeSet(0 nodes)"

        lines = [f"NodeSet({len(self._node_ids)} nodes):"]
        for node_id in sorted(self._node_ids):
            attrs = self._backend.get_node_attrs(node_id)
            name = attrs.get("name", "<unnamed>")
            node_type = attrs.get("type", "<no type>")
            color = attrs.get("colorRGB", "")

            info = f"  - {name} ({node_type})"
            if color:
                info += f" [color: {color}]"
            lines.append(info)

        return "\n".join(lines)
