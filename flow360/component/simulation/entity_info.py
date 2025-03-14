"""Deserializer for entity info retrieved from asset metadata pipeline."""

from abc import ABCMeta, abstractmethod
from typing import Annotated, List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.outputs.output_entities import (
    Point,
    PointArray,
    Slice,
)
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    Edge,
    GenericVolume,
    GhostCircularPlane,
    GhostSphere,
    Surface,
)

DraftEntityTypes = Annotated[
    Union[Box, Cylinder, Point, PointArray, Slice],
    pd.Field(discriminator="private_attribute_entity_type_name"),
]

GhostSurfaceTypes = Annotated[
    Union[GhostSphere, GhostCircularPlane],
    pd.Field(discriminator="private_attribute_entity_type_name"),
]


class EntityInfoModel(pd.BaseModel, metaclass=ABCMeta):
    """Base model for asset entity info JSON"""

    model_config = pd.ConfigDict(
        ##:: Pydantic kwargs
        extra="ignore",
        frozen=False,
        populate_by_name=True,
        validate_assignment=True,
        validate_default=True,
    )
    # Storing entities that appeared in the simulation JSON. (Otherwise when front end loads the JSON it will delete
    # entities that appear in simulation JSON but did not appear in EntityInfo)
    draft_entities: List[DraftEntityTypes] = pd.Field([])
    ghost_entities: List[GhostSurfaceTypes] = pd.Field([])

    @abstractmethod
    def get_boundaries(self, attribute_name: str = None) -> list[Surface]:
        """
        Helper function.
        Get the full list of boundary.
        If it is geometry then use supplied attribute name to get the list.
        """

    @abstractmethod
    def update_persistent_entities(self, *, param_entity_registry: EntityRegistry) -> None:
        """
        Update self persistent entities with param ones by simple id/name matching.
        """


class GeometryEntityInfo(EntityInfoModel):
    """Data model for geometry entityInfo.json"""

    type_name: Literal["GeometryEntityInfo"] = pd.Field("GeometryEntityInfo", frozen=True)
    face_ids: list[str] = pd.Field(
        [],
        description="A full list of faceIDs/model IDs that appear in the geometry.",
        alias="faceIDs",
    )
    face_attribute_names: List[str] = pd.Field(
        [],
        description="A full list of attribute names that the user can"
        "select to achieve grouping of faces. It has same length as `grouped_faces`",
        alias="faceAttributeNames",
    )
    grouped_faces: List[List[Surface]] = pd.Field(
        [[]],
        description="The resulting list "
        "of `Surface` entities after grouping using the attribute name.",
        alias="groupedFaces",
    )

    edge_ids: list[str] = pd.Field(
        [],
        description="A full list of edgeIDs/model IDs that appear in the geometry.",
        alias="edgeIDs",
    )
    edge_attribute_names: List[str] = pd.Field(
        [],
        description="A full list of attribute names that the user can"
        "select to achieve grouping of edges. It has same length as `grouped_edges`",
        alias="edgeAttributeNames",
    )
    grouped_edges: List[List[Edge]] = pd.Field(
        [[]],
        description="The resulting list "
        "of `Edge` entities after grouping using the attribute name.",
        alias="groupedEdges",
    )
    face_group_tag: Optional[str] = pd.Field(None, frozen=True)
    edge_group_tag: Optional[str] = pd.Field(None, frozen=True)

    def group_in_registry(
        self,
        entity_type_name: Literal["face", "edge"],
        attribute_name: str,
        registry: EntityRegistry,
    ) -> List[Union[Surface, Edge]]:
        """
        Group items with given attribute_name.
        """
        entity_list = self._get_list_of_entities(attribute_name, entity_type_name)
        for item in entity_list:
            registry.register(item)
        return registry

    def _get_list_of_entities(
        self,
        attribute_name: Union[str, None] = None,
        entity_type_name: Union[Literal["face", "edge"], None] = None,
    ) -> list:
        # Validations
        if entity_type_name is None:
            raise ValueError("Entity type name is required.")
        if entity_type_name not in ["face", "edge"]:
            raise ValueError(
                f"Invalid entity type name, expected 'face' or 'edge' but got {entity_type_name}."
            )
        if entity_type_name == "face":
            entity_attribute_names = self.face_attribute_names
            entity_full_list = self.grouped_faces
        else:
            entity_attribute_names = self.edge_attribute_names
            entity_full_list = self.grouped_edges

        # Use the supplied one if not None
        if attribute_name is not None:
            specified_attribute_name = attribute_name
        else:
            specified_attribute_name = (
                self.face_group_tag if entity_type_name == "face" else self.edge_group_tag
            )

            # pylint: disable=unsupported-membership-test,unsubscriptable-object
        if specified_attribute_name in entity_attribute_names:
            # pylint: disable=no-member
            return entity_full_list[entity_attribute_names.index(specified_attribute_name)]
        raise ValueError(
            f"The given attribute_name {attribute_name} is not found"
            f" in geometry metadata. Available: {entity_attribute_names}"
        )

    def get_boundaries(self, attribute_name: str = None) -> list[Surface]:
        """
        Get the full list of boundaries.
        If attribute_name is supplied then ignore stored face_group_tag and use supplied one.
        """
        return self._get_list_of_entities(attribute_name, "face")

    def update_persistent_entities(self, *, param_entity_registry: EntityRegistry) -> None:
        """
        1. Changed `Surface`/`Edge` names? (TODO: Add support for bodyGroup too)
        """

        def _search_and_replace(grouped_entities, entity_registry: EntityRegistry):
            for i_group, _ in enumerate(grouped_entities):
                for i_entity, _ in enumerate(grouped_entities[i_group]):
                    assigned_entity = entity_registry.find_by_asset_id(
                        entity_id=grouped_entities[i_group][i_entity].id,
                        entity_class=grouped_entities[i_group][i_entity].__class__,
                    )
                    if assigned_entity is not None:
                        grouped_entities[i_group][i_entity] = assigned_entity

        _search_and_replace(self.grouped_faces, param_entity_registry)
        _search_and_replace(self.grouped_edges, param_entity_registry)


class VolumeMeshEntityInfo(EntityInfoModel):
    """Data model for volume mesh entityInfo.json"""

    type_name: Literal["VolumeMeshEntityInfo"] = pd.Field("VolumeMeshEntityInfo", frozen=True)
    zones: list[GenericVolume] = pd.Field([])
    boundaries: list[Surface] = pd.Field([])

    @pd.field_validator("boundaries", mode="after")
    @classmethod
    def check_all_surface_has_interface_indicator(cls, value):
        """private_attribute_is_interface should have been set coming from volume mesh."""
        for item in value:
            if item.private_attribute_is_interface is None:
                raise ValueError(
                    "[INTERNAL] {item.name} is missing private_attribute_is_interface attribute!."
                )
        return value

    # pylint: disable=arguments-differ
    def get_boundaries(self) -> list:
        """
        Get the full list of boundary.
        """
        # pylint: disable=not-an-iterable
        return [item for item in self.boundaries if item.private_attribute_is_interface is False]

    def update_persistent_entities(self, *, param_entity_registry: EntityRegistry) -> None:
        """
        1. Changed GenericVolume axis and center etc
        """

        for i_zone, _ in enumerate(self.zones):
            # pylint:disable = unsubscriptable-object
            assigned_zone = param_entity_registry.find_by_asset_id(
                entity_id=self.zones[i_zone].id, entity_class=self.zones[i_zone].__class__
            )
            if assigned_zone is not None:
                # pylint:disable = unsupported-assignment-operation
                self.zones[i_zone] = assigned_zone


class SurfaceMeshEntityInfo(EntityInfoModel):
    """Data model for surface mesh entityInfo.json"""

    type_name: Literal["SurfaceMeshEntityInfo"] = pd.Field("SurfaceMeshEntityInfo", frozen=True)
    boundaries: list[Surface] = pd.Field([])
    ghost_entities: List[GhostSurfaceTypes] = pd.Field([])

    # pylint: disable=arguments-differ
    def get_boundaries(self) -> list:
        """
        Get the full list of boundary.
        """
        # pylint: disable=not-an-iterable
        return self.boundaries

    def update_persistent_entities(self, *, param_entity_registry: EntityRegistry) -> None:
        """
        Nothing related to SurfaceMeshEntityInfo for now.
        """
        return


EntityInfoUnion = Annotated[
    Union[GeometryEntityInfo, VolumeMeshEntityInfo, SurfaceMeshEntityInfo],
    pd.Field(discriminator="type_name"),
]


def parse_entity_info_model(data) -> EntityInfoUnion:
    """
    parse entity info data and return one of [GeometryEntityInfo, VolumeMeshEntityInfo, SurfaceMeshEntityInfo]
    """
    return pd.TypeAdapter(EntityInfoUnion).validate_python(data)


def get_entity_info_type_from_str(entity_type: str) -> type[EntityInfoModel]:
    """Get EntityInfo type from the asset type from the project tree"""
    entity_info_type = None
    if entity_type == "Geometry":
        entity_info_type = GeometryEntityInfo
    if entity_type == "VolumeMesh":
        entity_info_type = VolumeMeshEntityInfo

    return entity_info_type
