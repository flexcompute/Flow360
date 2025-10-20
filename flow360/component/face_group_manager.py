"""
Face grouping manager for geometry models
"""

from __future__ import annotations

from typing import Dict, List

from flow360.component.geometry_tree import FilterExpression, GeometryTree, TreeNode


class FaceGroupManager:
    """Manages face grouping logic"""

    def __init__(self, tree: GeometryTree):
        """
        Initialize face group manager

        Parameters
        ----------
        tree : GeometryTree
            The geometry tree to manage groupings for
        """
        self.tree = tree
        self.face_to_group: Dict[str, str] = {}  # face_uuid -> group_name
        self.face_groups: Dict[str, List[TreeNode]] = {}  # group_name -> list of faces

    def create_face_group(
        self, name: str, filter_expr: FilterExpression
    ) -> List[TreeNode]:
        """
        Create a face group by filtering nodes and collecting their faces.
        If a face already belongs to another group, it will be reassigned to the new group.

        Parameters
        ----------
        name : str
            Name of the face group
        filter_expr : FilterExpression
            Filter expression to match nodes

        Returns
        -------
        List[TreeNode]
            List of faces in the created group
        """
        # Find matching nodes using the tree
        matching_nodes = self.tree.find_nodes(filter_expr)

        # Collect faces from matching nodes
        group_faces = []
        new_face_uuids = set()
        
        for node in matching_nodes:
            faces = node.get_all_faces()
            for face in faces:
                if face.uuid:
                    group_faces.append(face)
                    new_face_uuids.add(face.uuid)

        # Remove these faces from their previous groups
        for group_name, faces in list(self.face_groups.items()):
            if group_name != name:
                self.face_groups[group_name] = [
                    f for f in faces if f.uuid not in new_face_uuids
                ]

        # Update face-to-group mapping
        for uuid in new_face_uuids:
            self.face_to_group[uuid] = name

        # Store the group
        self.face_groups[name] = group_faces

        return group_faces

    def get_face_groups(self) -> Dict[str, int]:
        """
        Get dictionary of face groups and their face counts

        Returns
        -------
        Dict[str, int]
            Dictionary mapping group names to number of faces
        """
        return {name: len(faces) for name, faces in self.face_groups.items()}

    def print_grouping_stats(self) -> None:
        """Print statistics about face grouping"""
        total_faces = len(self.tree.all_faces)
        
        # Count faces currently in groups
        faces_in_groups = sum(len(faces) for faces in self.face_groups.values())

        print(f"\n=== Face Grouping Statistics ===")
        print(f"Total faces: {total_faces}")
        print(f"Faces in groups: {faces_in_groups}")
        print(f"\nFace groups ({len(self.face_groups)}):")
        for group_name, faces in self.face_groups.items():
            print(f"  - {group_name}: {len(faces)} faces")
        print("=" * 33)

    def get_face_group_names(self) -> List[str]:
        """Get list of all face group names"""
        return list(self.face_groups.keys())

    def get_faces_in_group(self, group_name: str) -> List[TreeNode]:
        """
        Get faces in a specific group

        Parameters
        ----------
        group_name : str
            Name of the face group

        Returns
        -------
        List[TreeNode]
            List of faces in the group
        """
        return self.face_groups.get(group_name, [])

