"""Data models for CAD design intent (feature tree) metadata."""

from typing import Dict, List, Optional

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel


class CADFeature(Flow360BaseModel):
    """A single CAD feature (e.g., fillet, chamfer, extrusion) from the CAD feature tree."""

    id: str = pd.Field(description="Deterministic UUID identifying this feature.")
    native_id: str = pd.Field(
        "",
        description="Persistent GUID assigned by the authoring tool (A3DBIMRootData.m_pcGUID). "
        "Empty for formats that do not expose a stable native identifier.",
    )
    name: str = pd.Field(description="Feature name as reported by the CAD kernel.")
    cad_type: str = pd.Field(description="CAD feature type string (e.g., 'Fillet', 'Extrude').")
    parent_id: str = pd.Field(
        "", description="ID of the parent feature, or empty string for root features."
    )
    produced_entity_ids: List[str] = pd.Field(
        [],
        description="IDs of geometry entities (faces/edges) produced as the primary output of this feature.",
    )
    reference_entity_ids: List[str] = pd.Field(
        [],
        description="IDs of geometry entities used as construction or positioning references.",
    )
    support_entity_ids: List[str] = pd.Field(
        [],
        description="IDs of geometry entities used as support (e.g., faces selected for a fillet).",
    )
    properties: Dict[str, str] = pd.Field(
        {},
        description="Key-value pairs of feature parameters (e.g., radius, depth).",
    )
    child_feature_ids: List[str] = pd.Field(
        [],
        description="IDs of child features in the feature tree.",
    )

    def get_all_entity_ids(self) -> List[str]:
        """Return the union of all entity IDs referenced by this feature."""
        return list(
            dict.fromkeys(
                self.produced_entity_ids + self.reference_entity_ids + self.support_entity_ids
            )
        )


class CADDesignIntent(Flow360BaseModel):
    """Top-level container for CAD design intent extracted from the feature tree."""

    version: str = pd.Field("0.1.0", frozen=True)
    features: List[CADFeature] = pd.Field(
        [], description="Flat list of all CAD features extracted from the model."
    )

    def get_feature_by_id(self, feature_id: str) -> Optional[CADFeature]:
        """Return the feature with the given ID, or None if not found."""
        for feature in self.features:
            if feature.id == feature_id:
                return feature
        return None

    def get_features_by_type(self, cad_type: str) -> List[CADFeature]:
        """Return all features whose cad_type matches (case-insensitive)."""
        lower = cad_type.lower()
        return [f for f in self.features if f.cad_type.lower() == lower]

    def get_producing_features(self, entity_id: str) -> List[CADFeature]:
        """Return all features that produced the given entity ID."""
        return [f for f in self.features if entity_id in f.produced_entity_ids]

    def get_sibling_features(self, feature_id: str) -> List[CADFeature]:
        """Return features that share the same parent as the given feature (excluding itself)."""
        feature = self.get_feature_by_id(feature_id)
        if feature is None:
            return []
        return [
            f
            for f in self.features
            if f.parent_id == feature.parent_id and f.id != feature_id
        ]
