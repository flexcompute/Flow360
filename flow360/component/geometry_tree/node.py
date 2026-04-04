"""
node.py - GeometryTreeNode class for individual tree nodes

A GeometryTreeNode represents a single node in the geometry tree with direct attribute access.
"""

from typing import TYPE_CHECKING, Optional

from .node_type import NodeType

if TYPE_CHECKING:
    from .node_set import GeometryTreeNodeSet


class GeometryTreeNode:
    """
    Represents a single node in the geometry tree.

    Provides direct attribute access and navigation from a single node.
    """

    def __init__(self, geometry, tree, node_id: str):
        self._geometry = geometry
        self._tree = tree
        self._node_id = node_id
        self._attrs = tree.get_node_attrs(node_id)

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._attrs.get("name", "")

    @property
    def type(self) -> Optional[NodeType]:
        """Get the node type (e.g., NodeType.PART, NodeType.FACE)."""
        return self._attrs.get("type")

    @property
    def color(self) -> str:
        """Get the node color (colorRGB value)."""
        return self._attrs.get("colorRGB", "")

    def children(self, **filters) -> "GeometryTreeNodeSet":
        """Get direct children of this node."""
        from .node_set import (  # pylint: disable=import-outside-toplevel
            GeometryTreeNodeSet,
        )

        child_ids = set(self._tree.get_children(self._node_id))
        if filters:
            child_ids = self._tree.filter_nodes(child_ids, **filters)
        return GeometryTreeNodeSet(self._geometry, self._tree, child_ids)

    def descendants(self, **filters) -> "GeometryTreeNodeSet":
        """Get all descendants of this node."""
        from .node_set import (  # pylint: disable=import-outside-toplevel
            GeometryTreeNodeSet,
        )

        descendant_ids = self._tree.get_descendants(self._node_id)
        if filters:
            descendant_ids = self._tree.filter_nodes(descendant_ids, **filters)
        return GeometryTreeNodeSet(self._geometry, self._tree, descendant_ids)

    def faces(self, **filters) -> "GeometryTreeNodeSet":
        """Get all face nodes under this node."""
        from .node_set import (  # pylint: disable=import-outside-toplevel
            GeometryTreeNodeSet,
        )

        node_set = GeometryTreeNodeSet(self._geometry, self._tree, {self._node_id})
        return node_set.faces(**filters)

    def is_face(self) -> bool:
        """Check if this node is a face."""
        return self.type == NodeType.FACE

    def __repr__(self) -> str:
        info = f"GeometryTreeNode('{self.name}', type='{self.type}'"
        if self.color:
            info += f", color='{self.color}'"
        info += ")"
        return info

    def __eq__(self, other) -> bool:
        if not isinstance(other, GeometryTreeNode):
            return False
        return self._node_id == other._node_id

    def __hash__(self) -> int:
        return hash(self._node_id)
