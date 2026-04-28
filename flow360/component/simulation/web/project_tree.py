"""
Project tree models and construction helpers.
"""

# Pylint misidentifies pydantic model fields as FieldInfo in this copied
# lightweight tree model.
# pylint: disable=no-member,missing-function-docstring,duplicate-code

from __future__ import annotations

from typing import List, Literal, Optional

import pydantic as pd
from pydantic import PositiveInt

from flow360.component.utils import AssetShortID, get_short_asset_id, wrapstring


class ProjectTreeNode(pd.BaseModel):
    """
    ProjectTreeNode class containing the info of an asset item in a project tree.
    """

    asset_id: str = pd.Field()
    asset_name: str = pd.Field()
    asset_type: str = pd.Field()
    parent_id: Optional[str] = pd.Field(None)
    case_mesh_id: Optional[str] = pd.Field(None)
    case_mesh_label: Optional[str] = pd.Field(None)
    children: List = pd.Field([])
    min_length_short_id: PositiveInt = pd.Field(7)

    def construct_string(self, line_width):
        title_line = "<<" + self.asset_type + ">>"
        name_line = f"name: {self.asset_name}"
        id_line = f"id:   {self.short_id}"

        max_line_width = min(line_width, max(len(name_line), len(id_line)))
        block_line_width = max(len(title_line), max_line_width)

        name_line = wrapstring(long_str=f"name: {self.asset_name}", str_length=block_line_width)
        id_line = wrapstring(long_str=f"id:   {self.short_id}", str_length=block_line_width)
        return f"{title_line.center(block_line_width)}\n{name_line}\n{id_line}"

    def add_child(self, child: "ProjectTreeNode"):
        self.children.append(child)

    def remove_child(self, child_to_remove: "ProjectTreeNode"):
        self.children = [child for child in self.children if child is not child_to_remove]

    @property
    def short_id(self) -> str:
        return get_short_asset_id(
            full_asset_id=self.asset_id, num_character=self.min_length_short_id
        )

    @property
    def edge_label(self) -> str:
        if self.case_mesh_label:
            prefix = "Using VolumeMesh:\n"
            mesh_short_id = get_short_asset_id(
                full_asset_id=self.case_mesh_label,
                num_character=self.min_length_short_id,
            )
            return prefix + mesh_short_id.center(len(prefix))
        return None


class ProjectTree(pd.BaseModel):
    """
    ProjectTree class containing the project tree.
    """

    root: ProjectTreeNode = pd.Field(None)
    nodes: dict[str, ProjectTreeNode] = pd.Field({})
    short_id_map: dict[str, List[str]] = pd.Field({})

    def _update_case_mesh_label(self):
        for node_id in self._get_asset_ids_by_type(asset_type="Case"):
            node = self.nodes.get(node_id)
            parent_node = self._get_parent_node(node=node)
            if not parent_node:
                continue
            if parent_node.asset_type != "Case" or node.case_mesh_id == parent_node.case_mesh_id:
                node.case_mesh_label = None

    def _update_node_short_id(self):
        if len(self.nodes) == len(self.short_id_map):
            pass
        full_id_to_update = []
        short_id_duplicate = []
        for short_id, full_ids in self.short_id_map.items():
            if len(full_ids) > 1:
                short_id_duplicate.append(short_id)
                common_prefix = full_ids[0]
                for full_id in full_ids[1:]:
                    while not full_id.startswith(common_prefix):
                        common_prefix = common_prefix[:-1]
                common_prefix_processed = "".join(common_prefix.split("-")[1:])
                for full_id in full_ids:
                    self.nodes[full_id].min_length_short_id = len(common_prefix_processed) + 1
                    full_id_to_update.append(full_id)
        for full_id in full_id_to_update:
            self.short_id_map.update({self.nodes[full_id].short_id: [full_id]})
        for short_id in short_id_duplicate:
            self.short_id_map.pop(short_id, None)

    def _get_parent_node(self, node: ProjectTreeNode):
        if not node.parent_id:
            return None
        return self.nodes.get(node.parent_id, None)

    def _has_node(self, asset_id: str) -> bool:
        return asset_id in self.nodes.keys()

    def _get_asset_ids_by_type(
        self, asset_type: str = Literal["Geometry", "SurfaceMesh", "VolumeMesh", "Case"]
    ):
        return [node.asset_id for node in self.nodes.values() if node.asset_type == asset_type]

    @classmethod
    def _create_new_node(cls, asset_record: dict):
        parent_id = (
            asset_record["parentCaseId"]
            if asset_record["parentCaseId"]
            else asset_record["parentId"]
        )
        case_mesh_id = asset_record["parentId"] if asset_record["type"] == "Case" else None

        return ProjectTreeNode(
            asset_id=asset_record["id"],
            asset_name=asset_record["name"],
            asset_type=asset_record["type"],
            parent_id=parent_id,
            case_mesh_id=case_mesh_id,
            case_mesh_label=case_mesh_id,
        )

    def _update_short_id_map(self, new_node: ProjectTreeNode):
        if new_node.short_id not in self.short_id_map.keys():
            self.short_id_map[new_node.short_id] = []
        self.short_id_map[new_node.short_id].append(new_node.asset_id)

    def add(self, asset_record: dict):
        if self._has_node(asset_id=asset_record["id"]):
            return False

        new_node = ProjectTree._create_new_node(asset_record)
        self._update_short_id_map(new_node)
        if new_node.parent_id is None:
            self.root = new_node
        for node in self.nodes.values():
            if node.parent_id == new_node.asset_id:
                new_node.add_child(child=node)
            if node.asset_id == new_node.parent_id:
                node.add_child(child=new_node)
        self.nodes.update({new_node.asset_id: new_node})
        self._update_node_short_id()
        self._update_case_mesh_label()
        return True

    def remove_node(self, node_id: str):
        node = self.nodes.get(node_id)
        if not node:
            return
        if node.parent_id and self._has_node(node.parent_id):
            self.nodes[node.parent_id].remove_child(node)
        self.nodes.pop(node.asset_id)

    def construct_tree(self, asset_records: List[dict]):
        for asset_record in asset_records:
            new_node = ProjectTree._create_new_node(asset_record)
            self._update_short_id_map(new_node)
            if new_node.parent_id is None:
                self.root = new_node
            self.nodes.update({new_node.asset_id: new_node})

        for node in self.nodes.values():
            if node.parent_id and self._has_node(node.parent_id):
                self.nodes[node.parent_id].add_child(node)
        self._update_node_short_id()
        self._update_case_mesh_label()

    @pd.validate_call
    def get_full_asset_id(self, query_asset: AssetShortID) -> str:
        if query_asset.asset_id is None:
            asset_ids = self._get_asset_ids_by_type(asset_type=query_asset.asset_type)
            if not asset_ids:
                raise ValueError(f"No {query_asset.asset_type} is available in this project.")
            return asset_ids[-1]

        if query_asset.asset_id in self.nodes:
            return query_asset.asset_id

        full_ids = self.short_id_map.get(query_asset.asset_id, None)
        if full_ids is None:
            raise ValueError(
                f"This asset does not exist in this project. Please check the input asset ID ({query_asset.asset_id})"
            )
        if len(full_ids) > 1:
            raise ValueError(
                f"The input asset ID ({query_asset.asset_id}) is too short to retrieve the correct asset."
            )
        return full_ids[0]
