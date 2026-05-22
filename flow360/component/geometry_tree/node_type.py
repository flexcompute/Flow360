"""
node_type.py - Enum for geometry tree node types.
"""

from enum import Enum


class NodeType(str, Enum):
    """Valid node types in the geometry tree."""

    MODEL_FILE = "ModelFile"
    ASSEMBLY = "Assembly"
    PART = "Part"
    BODY = "Body"
    BODY_COLLECTION = "BodyCollection"
    SHELL_COLLECTION = "ShellCollection"
    SHELL = "Shell"
    FACE = "Face"
