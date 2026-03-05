"""
geometry_tree - Hierarchical geometry tree for face grouping

Provides tree navigation and face grouping using a fluent, scope-based API.
"""

from .face_group import FaceGroup
from .node import Node
from .node_set import NodeSet
from .node_type import NodeType
from .tree_backend import TreeBackend

__all__ = ["TreeBackend", "NodeSet", "Node", "NodeType", "FaceGroup"]
