"""
face_group.py - FaceGroup class for named face groups
"""

from typing import Set


class FaceGroup:
    """
    Represents a named group of faces.

    Can be used in set operations with Geometry to get remaining faces.
    """

    def __init__(self, name: str, node_ids: Set[str]):
        self.name = name
        self._node_ids = node_ids.copy()

    @property
    def face_ids(self) -> Set[str]:
        """Get the set of face node IDs in this group."""
        return self._node_ids.copy()

    def face_count(self) -> int:
        """Get number of faces in group."""
        return len(self._node_ids)

    def __repr__(self) -> str:
        return f"FaceGroup('{self.name}', {self.face_count()} faces)"
