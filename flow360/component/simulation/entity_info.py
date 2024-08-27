"""Desearlizer for entity info retrieved from asset metadata pipeline."""

from abc import ABCMeta, abstractmethod
from typing import List, Literal, Union

import pydantic as pd

from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.primitives import Edge, GenericVolume, Surface


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

    @abstractmethod
    def get_full_list_of_boundaries(self, attribute_name: str = None) -> list:
        """
        Helper function.
        Get the full list of boundary names. If it is geometry then use supplied attribute name to get the list.
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

    def group_items_with_given_tag(
        self,
        entity_type_name: Literal["face", "edge"],
        attribute_name: str,
        registry: EntityRegistry,
    ) -> List[Union[Surface, Edge]]:
        """
        Group items with given attribute_name.
        """
        entity_list = self._get_full_list_of_given_entity_type(attribute_name, entity_type_name)
        for item in entity_list:
            registry.register(item)
        return registry

    def _get_full_list_of_given_entity_type(
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
        entity_attribute_names = getattr(self, f"{entity_type_name}_attribute_names")
        entity_full_list = getattr(self, f"grouped_{entity_type_name}s")

        if attribute_name is not None:
            if attribute_name in entity_attribute_names:
                return entity_full_list[entity_attribute_names.index(attribute_name)]
            raise ValueError(
                f"The given attribute_name {attribute_name} is not found"
                f" in geometry metadata. Available: {entity_attribute_names}"
            )
        raise ValueError("Attribute name is required to get the full list of boundaries.")

    def get_full_list_of_boundaries(self, attribute_name: str = None) -> list:
        """
        Get the full list of boundary names. If it is geometry then use supplied attribute name to get the list.
        """
        return self._get_full_list_of_given_entity_type(attribute_name, "face")


class VolumeMeshEntityInfo(EntityInfoModel):
    """Data model for volume mesh entityInfo.json"""

    type_name: Literal["VolumeMeshEntityInfo"] = pd.Field("VolumeMeshEntityInfo", frozen=True)
    zones: list[GenericVolume] = pd.Field([])
    boundaries: list[Surface] = pd.Field([])

    # pylint: disable=arguments-differ
    def get_full_list_of_boundaries(self) -> list:
        """
        Get the full list of boundary names. If it is geometry then use supplied attribute name to get the list.
        """
        return self.boundaries


class SurfaceMeshEntityInfo(EntityInfoModel):
    """Data model for surface mesh entityInfo.json"""

    type_name: Literal["SurfaceMeshEntityInfo"] = pd.Field("SurfaceMeshEntityInfo", frozen=True)

    # pylint: disable=arguments-differ
    def get_full_list_of_boundaries(self) -> list:
        """
        Get the full list of boundary names. If it is geometry then use supplied attribute name to get the list.
        """
        raise NotImplementedError("Not implemented yet.")
