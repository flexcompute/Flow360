"""Data models for CAD design intent (feature tree) metadata."""

from typing import Dict, List, Literal, Optional

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel


class EntityRef(Flow360BaseModel):
    """A reference to a topological entity (face or edge) produced or used by a node."""

    occurrence_uuid: str = pd.Field(
        description="UUID of the part occurrence that owns this entity."
    )
    path: List[str] = pd.Field(
        description="Persistent IDs of each assembly level from root to the owning occurrence."
    )
    entity_id: str = pd.Field("", description="HOOPS persistent ID (CLSID or numeric string) of the topology entity.")
    entity_uuid: str = pd.Field(description="Deterministic UUIDv5 of the face or edge.")


class CADTreeNode(Flow360BaseModel):
    """A node in the CAD FRM tree. isFeature=True for semantic CAD operations, False for parameters."""

    node_number: int = pd.Field(description="Integer identifier, scoped to its component.")
    node_id: str = pd.Field("", description="HOOPS persistent ID (CLSID or numeric string); empty if unavailable.")
    name: str = pd.Field(description="Node name as reported by the CAD kernel.")
    node_type: str = pd.Field(description="FRM node type string (e.g., 'Node', 'Sketch', 'Depth').")
    is_feature: bool = pd.Field(False, description="True if this node is a visible feature in the CAD tree.")
    is_sub_tree: bool = pd.Field(False, description="True if this node represents a sub-tree.")
    parent_node: Optional[int] = pd.Field(None, description="nodeNumber of the parent, or null for root.")
    produced_entities: List[EntityRef] = pd.Field([], description="Entities produced by this node.")
    referenced_entities: List[EntityRef] = pd.Field([], description="Entities referenced by this node.")
    data: Dict[str, str] = pd.Field({}, description="Embedded data values (doubles, ints, strings, enums).")
    sub_features: List[int] = pd.Field([], description="nodeNumbers of child nodes that are features.")
    parameters: List[int] = pd.Field([], description="nodeNumbers of child nodes that are parameters.")

    def get_all_entity_ids(self) -> List[str]:
        """Return the union of all entity UUIDs referenced by this node."""
        return list(
            dict.fromkeys(
                [r.entity_uuid for r in self.produced_entities + self.referenced_entities]
            )
        )


class CADComponent(Flow360BaseModel):
    """A single component (part definition or assembly) with its FRM tree nodes."""

    component_type: Literal["partDefinition", "assembly"] = pd.Field(
        description="Whether this component is a part definition or an assembly."
    )
    name: str = pd.Field(description="Component name as reported by the CAD kernel.")
    entity_id: str = pd.Field(description="Deterministic UUIDv5 for this component.")
    path: List[str] = pd.Field(description="Persistent IDs from root to this component.")
    features: List[CADTreeNode] = pd.Field([], description="FRM tree nodes for this component.")

    def get_node_by_number(self, node_number: int) -> Optional[CADTreeNode]:
        """Return the node with the given number, or None if not found."""
        for node in self.features:
            if node.node_number == node_number:
                return node
        return None

    def get_nodes_by_type(self, node_type: str) -> List[CADTreeNode]:
        """Return all nodes whose node_type matches (case-insensitive)."""
        lower = node_type.lower()
        return [n for n in self.features if n.node_type.lower() == lower]

    def get_feature_nodes(self) -> List[CADTreeNode]:
        """Return only the feature nodes (isFeature=True)."""
        return [n for n in self.features if n.is_feature]

    def get_producing_features(self, entity_id: str) -> List[CADTreeNode]:
        """Return all nodes that produced the given entity UUID."""
        return [n for n in self.features if any(r.entity_uuid == entity_id for r in n.produced_entities)]


class CADDesignIntent(Flow360BaseModel):
    """Top-level container for CAD design intent extracted from the feature tree."""

    version: str = pd.Field("1.0.0", frozen=True)
    components: List[CADComponent] = pd.Field(
        [], description="List of components with their FRM tree nodes."
    )

    def get_all_nodes(self) -> List[CADTreeNode]:
        """Return a flat list of all nodes across all components."""
        return [n for component in self.components for n in component.features]

    def get_component_by_name(self, name: str) -> Optional[CADComponent]:
        """Return the first component with the given name, or None."""
        for component in self.components:
            if component.name == name:
                return component
        return None
