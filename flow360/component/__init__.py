"""Flow360 Component Module"""

from flow360.component.geometry_tree import (
    GeometryTree,
    NodeCollection,
    NodeType,
    TreeNode,
    TreeSearch,
)

__all__ = [
    "GeometryTree",
    "NodeCollection",
    "NodeType",
    "TreeNode",
    "TreeSearch",
]

# Note: FaceGroup is available but not exported here to avoid circular imports.
# Import it directly when needed: from flow360.component.geometry import FaceGroup



