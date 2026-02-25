"""
node.py - Node class for individual tree nodes

A Node represents a single node in the geometry tree with direct attribute access.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .node_set import NodeSet


class Node:
    """
    Represents a single node in the geometry tree.

    Provides direct attribute access and navigation from a single node.
    """

    def __init__(self, geometry, backend, node_id: str):
        self._geometry = geometry
        self._backend = backend
        self._node_id = node_id
        self._attrs = backend.get_node_attrs(node_id)

    @property
    def node_id(self) -> str:
        """Get the internal node ID."""
        return self._node_id

    @property
    def name(self) -> str:
        """Get the node name."""
        return self._attrs.get("name", "")

    @property
    def type(self) -> str:
        """Get the node type (e.g., 'PartDefinition', 'Face')."""
        return self._attrs.get("type", "")

    @property
    def color(self) -> str:
        """Get the node color (colorRGB value)."""
        return self._attrs.get("colorRGB", "")

    @property
    def material(self) -> str:
        """Get the node material."""
        return self._attrs.get("material", "")

    @property
    def face_count(self) -> Optional[int]:
        """Get the face count (if available)."""
        return self._attrs.get("faceCount")

    @property
    def uuid(self) -> Optional[str]:
        """Get the _Flow360UUID (if available)."""
        attrs = self._attrs.get("attributes", {})
        return attrs.get("_Flow360UUID")

    @property
    def attributes(self) -> dict:
        """Get all attributes as a dictionary."""
        return self._attrs.copy()

    def children(self, **filters) -> "NodeSet":
        """Get direct children of this node."""
        from .node_set import NodeSet

        child_ids = set(self._backend.get_children(self._node_id))
        if filters:
            child_ids = self._backend.filter_nodes(child_ids, **filters)
        return NodeSet(self._geometry, self._backend, child_ids)

    def descendants(self, **filters) -> "NodeSet":
        """Get all descendants of this node."""
        from .node_set import NodeSet

        descendant_ids = self._backend.get_descendants(self._node_id)
        if filters:
            descendant_ids = self._backend.filter_nodes(descendant_ids, **filters)
        return NodeSet(self._geometry, self._backend, descendant_ids)

    def faces(self, **filters) -> "NodeSet":
        """Get all face nodes under this node."""
        from .node_set import NodeSet

        node_set = NodeSet(self._geometry, self._backend, {self._node_id})
        return node_set.faces(**filters)

    def is_face(self) -> bool:
        """Check if this node is a face (Face or FacePointer)."""
        return self.type in ("Face", "FacePointer")

    def __repr__(self) -> str:
        info = f"Node('{self.name}', type='{self.type}'"
        if self.color:
            info += f", color='{self.color}'"
        info += ")"
        return info

    def __eq__(self, other) -> bool:
        if not isinstance(other, Node):
            return False
        return self._node_id == other._node_id

    def __hash__(self) -> int:
        return hash(self._node_id)
