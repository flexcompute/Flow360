"""
geometry_tree - Hierarchical geometry tree for face grouping

Provides tree navigation and face grouping using a fluent, scope-based API.
"""

from .tree_backend import TreeBackend
from .node_set import NodeSet
from .node import Node
from .face_group import FaceGroup

__all__ = ["TreeBackend", "NodeSet", "Node", "FaceGroup"]
