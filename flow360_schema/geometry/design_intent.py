"""Data models for CAD design intent (feature tree) metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from flow360_schema.framework.base_model import Flow360BaseModel


class CADEntityRef(Flow360BaseModel):
    """A reference to a topological entity (face or edge) produced or used by a node."""

    occurrence_uuid: str = Field(
        description="UUID of the part occurrence that owns this entity.",
    )
    path: list[str] = Field(
        description="Persistent IDs of each assembly level from root to the owning occurrence.",
    )
    entity_id: str = Field(
        default="",
        description="HOOPS persistent ID (CLSID or numeric string) of the topology entity.",
    )
    entity_uuid: str = Field(
        description="Deterministic UUIDv5 of the face or edge.",
    )


class CADTreeNodeDataEntry(Flow360BaseModel):
    """A single key-value data entry attached to a tree node."""

    key: str = Field(description="Entry name (e.g. 'value', 'depth').")
    value: str = Field(description="Entry value as a string.")


class CADTreeNode(Flow360BaseModel):
    """A node in the CAD FRM tree.

    isFeature=True for semantic CAD operations, False for parameters.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "$displayOrder": [
                "nodeNumber",
                "nodeId",
                "name",
                "nodeType",
                "isFeature",
                "isSubTree",
                "parentNode",
                "producedEntities",
                "referencedEntities",
                "data",
                "subFeatures",
                "parameters",
            ],
        }
    )

    node_number: int = Field(
        description="Integer identifier, scoped to its component.",
    )
    node_id: str = Field(
        default="",
        description="HOOPS persistent ID (CLSID or numeric string); empty if unavailable.",
    )
    name: str = Field(
        description="Node name as reported by the CAD kernel.",
    )
    node_type: str = Field(
        description="FRM node type string (e.g., 'Node', 'Sketch', 'Depth').",
    )
    is_feature: bool = Field(
        default=False,
        description="True if this node is a visible feature in the CAD tree.",
    )
    is_sub_tree: bool = Field(
        default=False,
        description="True if this node represents a sub-tree.",
    )
    parent_node: int | None = Field(
        default=None,
        description="nodeNumber of the parent, or null for root.",
    )
    produced_entities: list[CADEntityRef] = Field(
        default=[],
        description="Entities produced by this node.",
    )
    referenced_entities: list[CADEntityRef] = Field(
        default=[],
        description="Entities referenced by this node.",
    )
    data: list[CADTreeNodeDataEntry] = Field(
        default=[],
        description="Embedded data values (doubles, ints, strings, enums).",
    )
    sub_features: list[int] = Field(
        default=[],
        description="nodeNumbers of child nodes that are features.",
    )
    parameters: list[int] = Field(
        default=[],
        description="nodeNumbers of child nodes that are parameters.",
    )


class CADComponent(Flow360BaseModel):
    """A single component (part definition or assembly) with its FRM tree nodes."""

    model_config = ConfigDict(
        json_schema_extra={
            "$displayOrder": [
                "componentType",
                "name",
                "entityId",
                "path",
                "features",
            ],
        }
    )

    component_type: Literal["partDefinition", "assembly"] = Field(
        description="Whether this component is a part definition or an assembly.",
    )
    name: str = Field(
        description="Component name as reported by the CAD kernel.",
    )
    entity_id: str = Field(
        description="Deterministic UUIDv5 for this component.",
    )
    path: list[str] = Field(
        description="Persistent IDs from root to this component.",
    )
    features: list[CADTreeNode] = Field(
        default=[],
        description="FRM tree nodes for this component.",
    )


class CADDesignIntent(Flow360BaseModel):
    """Top-level container for CAD design intent extracted from the feature tree."""

    model_config = ConfigDict(
        json_schema_extra={
            "$version": "1.0.0",
            "$displayOrder": ["version", "components"],
        }
    )

    version: str = Field(
        default="1.0.0",
        frozen=True,
        description="Schema version for this design intent payload.",
    )
    components: list[CADComponent] = Field(
        default=[],
        description="List of components with their FRM tree nodes.",
    )
