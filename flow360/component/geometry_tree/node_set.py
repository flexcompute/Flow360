"""
node_set.py - GeometryTreeNodeSet class for tree navigation

A GeometryTreeNodeSet represents a set of nodes at the current navigation scope.
Supports method chaining for fluent tree traversal and set operations.
"""

from typing import Iterator, Set

from .filters import is_face_node, matches_criteria
from .node import GeometryTreeNode


class GeometryTreeNodeSet:
    """
    A set of nodes at the current navigation scope.

    Supports fluent navigation through method chaining and set operations.
    """

    def __init__(self, geometry, tree, node_ids: Set[str]):
        self._geometry = geometry
        self._tree = tree
        self._node_ids = node_ids.copy()

    # ================================================================
    # Navigation Methods
    # ================================================================

    def children(self, **filters) -> "GeometryTreeNodeSet":
        """Get direct children of all nodes in this set."""
        child_ids = set()
        for node_id in self._node_ids:
            child_ids.update(self._tree.get_children(node_id))
        if filters:
            child_ids = self._tree.filter_nodes(child_ids, **filters)
        return GeometryTreeNodeSet(self._geometry, self._tree, child_ids)

    def descendants(self, **filters) -> "GeometryTreeNodeSet":
        """Get all descendants of all nodes in this set."""
        descendant_ids = set()
        for node_id in self._node_ids:
            descendant_ids.update(self._tree.get_descendants(node_id))
        if filters:
            descendant_ids = self._tree.filter_nodes(descendant_ids, **filters)
        return GeometryTreeNodeSet(self._geometry, self._tree, descendant_ids)

    def faces(self, **filters) -> "GeometryTreeNodeSet":
        """Get all face nodes within this node scope."""
        all_nodes = set()
        for node_id in self._node_ids:
            all_nodes.add(node_id)
            all_nodes.update(self._tree.get_descendants(node_id))

        face_node_ids = set()
        for node_id in all_nodes:
            attrs = self._tree.get_node_attrs(node_id)
            if is_face_node(attrs):
                if filters:
                    if not matches_criteria(attrs, filters):
                        continue
                face_node_ids.add(node_id)

        return GeometryTreeNodeSet(self._geometry, self._tree, face_node_ids)

    # ================================================================
    # Set Operations
    # ================================================================

    def __or__(self, other: "GeometryTreeNodeSet") -> "GeometryTreeNodeSet":
        """Union of two GeometryTreeNodeSets."""
        if not isinstance(other, GeometryTreeNodeSet):
            return NotImplemented
        return GeometryTreeNodeSet(self._geometry, self._tree, self._node_ids | other._node_ids)

    def __and__(self, other: "GeometryTreeNodeSet") -> "GeometryTreeNodeSet":
        """Intersection of two GeometryTreeNodeSets."""
        if not isinstance(other, GeometryTreeNodeSet):
            return NotImplemented
        return GeometryTreeNodeSet(self._geometry, self._tree, self._node_ids & other._node_ids)

    def __sub__(self, other) -> "GeometryTreeNodeSet":
        """Difference: supports GeometryTreeNodeSet and FaceGroup."""
        from .face_group import FaceGroup  # pylint: disable=import-outside-toplevel

        if isinstance(other, (GeometryTreeNodeSet, FaceGroup)):
            return GeometryTreeNodeSet(self._geometry, self._tree, self._node_ids - other._node_ids)
        return NotImplemented

    # ================================================================
    # Collection Methods
    # ================================================================

    def is_empty(self) -> bool:
        """Check if GeometryTreeNodeSet is empty."""
        return len(self._node_ids) == 0

    def __len__(self) -> int:
        return len(self._node_ids)

    def __iter__(self) -> Iterator[GeometryTreeNode]:
        for node_id in self._node_ids:
            yield GeometryTreeNode(self._geometry, self._tree, node_id)

    def __contains__(self, item) -> bool:
        if isinstance(item, GeometryTreeNode):
            return item._node_id in self._node_ids  # pylint: disable=protected-access
        return item in self._node_ids

    def __bool__(self) -> bool:
        return len(self._node_ids) > 0

    def __eq__(self, other) -> bool:
        if not isinstance(other, GeometryTreeNodeSet):
            return False
        return self._node_ids == other._node_ids

    __hash__ = None

    def __repr__(self) -> str:
        if not self._node_ids:
            return "GeometryTreeNodeSet(0 nodes)"

        lines = [f"GeometryTreeNodeSet({len(self._node_ids)} nodes):"]
        for node_id in sorted(self._node_ids):
            attrs = self._tree.get_node_attrs(node_id)
            name = attrs.get("name", "<unnamed>")
            node_type = attrs.get("type", "<no type>")
            color = attrs.get("colorRGB", "")

            info = f"  - {name} ({node_type})"
            if color:
                info += f" [color: {color}]"
            lines.append(info)

        return "\n".join(lines)
