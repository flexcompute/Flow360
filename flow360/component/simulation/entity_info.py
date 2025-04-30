"""Deserializer for entity info retrieved from asset metadata pipeline."""

from abc import ABCMeta, abstractmethod
from typing import Annotated, List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.outputs.output_entities import (
    Point,
    PointArray,
    PointArray2D,
    Slice,
)
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    Edge,
    GenericVolume,
    GeometryBodyGroup,
    GhostCircularPlane,
    GhostSphere,
    Surface,
)
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.component.utils import GeometryFiles
from flow360.log import log

DraftEntityTypes = Annotated[
    Union[Box, Cylinder, Point, PointArray, PointArray2D, Slice],
    pd.Field(discriminator="private_attribute_entity_type_name"),
]

GhostSurfaceTypes = Annotated[
    Union[GhostSphere, GhostCircularPlane],
    pd.Field(discriminator="private_attribute_entity_type_name"),
]


class EntityInfoModel(Flow360BaseModel, metaclass=ABCMeta):
    """Base model for asset entity info JSON"""

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
    def update_persistent_entities(self, *, asset_entity_registry: EntityRegistry) -> None:
        """
        Update self persistent entities with param ones by simple id/name matching.
        """

    @abstractmethod
    def get_registry(self, internal_registry, **kwargs):
        """
        Ensure that `internal_registry` exists and if not, initialize `internal_registry`.
        """


class GeometryEntityInfo(EntityInfoModel):
    """Data model for geometry entityInfo.json"""

    type_name: Literal["GeometryEntityInfo"] = pd.Field("GeometryEntityInfo", frozen=True)

    body_ids: list[str] = pd.Field(
        [],
        description="A full list of body IDs that appear in the geometry.",
        alias="bodyIDs",
    )
    body_attribute_names: List[str] = pd.Field(
        [],
        description="A full list of attribute names that the user can"
        "select to achieve grouping of bodies. It has same length as `grouped_bodies`",
        alias="bodyAttributeNames",
    )
    grouped_bodies: List[List[GeometryBodyGroup]] = pd.Field(
        [[]],
        description="The resulting list "
        "of `GeometryBodyGroup` entities after grouping using the attribute name.",
        alias="groupedBodies",
    )

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

    body_group_tag: Optional[str] = pd.Field(None, frozen=True)
    face_group_tag: Optional[str] = pd.Field(None, frozen=True)
    edge_group_tag: Optional[str] = pd.Field(None, frozen=True)

    def group_in_registry(
        self,
        entity_type_name: Literal["face", "edge", "body"],
        attribute_name: str,
        registry: EntityRegistry,
    ) -> EntityRegistry:
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
        entity_type_name: Literal["face", "edge", "body"] = None,
    ) -> Union[List[Surface], List[Edge], List[GeometryBodyGroup]]:
        # Validations
        if entity_type_name is None:
            raise ValueError("Entity type name is required.")
        if entity_type_name not in ["face", "edge", "body"]:
            raise ValueError(
                f"Invalid entity type name, expected 'body, 'face' or 'edge' but got {entity_type_name}."
            )
        if entity_type_name == "face":
            entity_attribute_names = self.face_attribute_names
            entity_full_list = self.grouped_faces
            specified_attribute_name = self.face_group_tag
        elif entity_type_name == "edge":
            entity_attribute_names = self.edge_attribute_names
            entity_full_list = self.grouped_edges
            specified_attribute_name = self.edge_group_tag
        else:
            entity_attribute_names = self.body_attribute_names
            entity_full_list = self.grouped_bodies
            specified_attribute_name = self.body_group_tag

        # Use the supplied one if not None
        if attribute_name is not None:
            specified_attribute_name = attribute_name

        # pylint: disable=unsupported-membership-test
        if specified_attribute_name in entity_attribute_names:
            # pylint: disable=no-member, unsubscriptable-object
            return entity_full_list[entity_attribute_names.index(specified_attribute_name)]

        raise ValueError(
            f"The given attribute_name `{attribute_name}` is not found"
            f" in geometry metadata. Available: {entity_attribute_names}"
        )

    def get_boundaries(self, attribute_name: str = None) -> list[Surface]:
        """
        Get the full list of boundaries.
        If attribute_name is supplied then ignore stored face_group_tag and use supplied one.
        """
        return self._get_list_of_entities(attribute_name, "face")

    def update_persistent_entities(self, *, asset_entity_registry: EntityRegistry) -> None:
        """
        Update the persistent entities stored inside `self` according to `asset_entity_registry`
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

        _search_and_replace(self.grouped_faces, asset_entity_registry)  # May changed entity name
        _search_and_replace(self.grouped_edges, asset_entity_registry)
        _search_and_replace(self.grouped_bodies, asset_entity_registry)  # May changed entity name

    def _get_processed_file_list(self):
        """
        Return the list of files that are uploaded by geometryConversionPipeline.

        This function examines the files mentioned under `grouped_bodies->groupByFile`
        and append folder prefix if necessary.
        """
        body_groups_grouped_by_file = self._get_list_of_entities("groupByFile", "body")
        unprocessed_file_names = [item.private_attribute_id for item in body_groups_grouped_by_file]
        processed_geometry_file_names = []
        surface_mesh_file_names = []
        for unprocessed_file_name in unprocessed_file_names:
            # All geometry source file gets lumped into a single file
            if GeometryFiles.check_is_valid_geometry_file_format(file_name=unprocessed_file_name):
                # This is a geometry file
                processed_geometry_file_names.append(f"{unprocessed_file_name}.egads")
            else:
                # Not a geometry file. Maybe a surface mesh file. No special treatment needed.
                surface_mesh_file_names.append(unprocessed_file_name)
        return processed_geometry_file_names, surface_mesh_file_names

    def _get_id_to_file_map(
        self, *, entity_type_name: Literal["face", "edge", "body"]
    ) -> dict[str, str]:
        """Returns faceId/edgeId/bodyId to file name mapping."""

        if entity_type_name not in ("face", "edge", "body"):
            raise ValueError(
                f"Invalid entity_type_name given:{entity_type_name}. Valid options are 'face', 'edge', 'body'"
            )

        if entity_type_name in ("face", "edge"):
            # No direct/consistent way of getting this info compared to bodies
            # Also need to figure out what mesher team needs exactly.
            raise NotImplementedError()

        id_to_file_name = {}

        body_groups_grouped_by_file = self._get_list_of_entities("groupByFile", "body")
        for item in body_groups_grouped_by_file:
            if GeometryFiles.check_is_valid_geometry_file_format(
                file_name=item.private_attribute_id
            ):
                file_name = f"{item.private_attribute_id}.egads"
            else:
                file_name = item.private_attribute_id
            for sub_component_id in item.private_attribute_sub_components:
                id_to_file_name[sub_component_id] = file_name

        return id_to_file_name

    def _get_default_grouping_tag(self, entity_type_name: Literal["face", "edge", "body"]) -> str:
        """
        Returns the default grouping tag for the given entity type.
        The selection logic is intended to mimic the webUI behavior.
        """

        def _get_the_first_non_id_tag(
            attribute_names: list[str], entity_type_name: Literal["face", "edge", "body"]
        ):
            if not attribute_names:
                raise ValueError(
                    f"[Internal] No valid tag available for grouping {entity_type_name}."
                )
            id_tag = f"{entity_type_name}Id"
            for item in attribute_names:
                if item != id_tag:
                    return item
            return id_tag

        if entity_type_name == "body":
            return _get_the_first_non_id_tag(self.body_attribute_names, entity_type_name)

        if entity_type_name == "face":
            return _get_the_first_non_id_tag(self.face_attribute_names, entity_type_name)

        if entity_type_name == "edge":
            return _get_the_first_non_id_tag(self.edge_attribute_names, entity_type_name)

        raise ValueError(f"[Internal] Invalid entity type name: {entity_type_name}.")

    def _group_entity_by_tag(
        self,
        entity_type_name: Literal["face", "edge", "body"],
        tag_name: str,
        registry: EntityRegistry = None,
    ) -> EntityRegistry:

        if entity_type_name not in ["face", "edge", "body"]:
            raise ValueError(
                f"[Internal] Unknown entity type: `{entity_type_name}`, allowed entity: 'face', 'edge', 'body'."
            )

        if registry is None:
            registry = EntityRegistry()

        existing_tag = None
        if entity_type_name == "face" and self.face_group_tag is not None:
            existing_tag = self.face_group_tag

        elif entity_type_name == "edge" and self.edge_group_tag is not None:
            existing_tag = self.edge_group_tag

        elif entity_type_name == "body" and self.body_group_tag is not None:
            existing_tag = self.body_group_tag

        if existing_tag:
            if existing_tag != tag_name:
                log.info(
                    f"Regrouping {entity_type_name} entities under `{tag_name}` tag (previous `{existing_tag}`)."
                )
            registry = self._reset_grouping(entity_type_name=entity_type_name, registry=registry)

        registry = self.group_in_registry(
            entity_type_name, attribute_name=tag_name, registry=registry
        )
        if entity_type_name == "face":
            with model_attribute_unlock(self, "face_group_tag"):
                self.face_group_tag = tag_name
        elif entity_type_name == "edge":
            with model_attribute_unlock(self, "edge_group_tag"):
                self.edge_group_tag = tag_name
        else:
            with model_attribute_unlock(self, "body_group_tag"):
                self.body_group_tag = tag_name

        return registry

    def _reset_grouping(
        self, entity_type_name: Literal["face", "edge", "body"], registry: EntityRegistry
    ) -> EntityRegistry:
        if entity_type_name == "face":
            registry.clear(Surface)
            with model_attribute_unlock(self, "face_group_tag"):
                self.face_group_tag = None
        elif entity_type_name == "edge":
            registry.clear(Edge)
            with model_attribute_unlock(self, "edge_group_tag"):
                self.edge_group_tag = None
        else:
            registry.clear(GeometryBodyGroup)
            with model_attribute_unlock(self, "body_group_tag"):
                self.body_group_tag = None
        return registry

    def get_registry(self, internal_registry, **_) -> EntityRegistry:
        if internal_registry is None:
            internal_registry = EntityRegistry()
            if self.face_group_tag is None:
                face_group_tag = self._get_default_grouping_tag("face")
                log.info(f"Using `{face_group_tag}` as default grouping for faces.")
            else:
                face_group_tag = self.face_group_tag

            internal_registry = self._group_entity_by_tag(
                "face", face_group_tag, registry=internal_registry
            )

            if self.edge_group_tag is None:
                edge_group_tag = self._get_default_grouping_tag("edge")
                log.info(f"Using `{edge_group_tag}` as default grouping for edges.")
            else:
                edge_group_tag = self.edge_group_tag

            internal_registry = self._group_entity_by_tag(
                "edge", edge_group_tag, registry=internal_registry
            )

            if self.body_attribute_names:
                # Post-25.5 geometry asset. For Pre 25.5 we just skip body grouping.
                if self.body_group_tag is None:
                    body_group_tag = self._get_default_grouping_tag("body")
                    log.info(f"Using `{body_group_tag}` as default grouping for bodies.")
                else:
                    body_group_tag = self.body_group_tag

                internal_registry = self._group_entity_by_tag(
                    "body", self.body_group_tag, registry=internal_registry
                )
        return internal_registry

    def compute_transformation_matrices(self):
        """
        Computes the transformation matrices for the **selected** body group and store
        matrices under `private_attribute_matrix`.
        Won't compute for any `GeometryBodyGroup` that is not asked by the user to save expense.
        """
        assert self.body_group_tag is not None, "[Internal] no body grouping specified."
        assert (
            self.body_group_tag
            in self.body_attribute_names  # pylint:disable=unsupported-membership-test
        ), f"[Internal] invalid body grouping. {self.body_attribute_names} allowed but got {self.body_group_tag}."

        i_body_group = self.body_attribute_names.index(  # pylint:disable=no-member
            self.body_group_tag
        )
        for body_group in self.grouped_bodies[  # pylint:disable=unsubscriptable-object
            i_body_group
        ]:
            body_group.transformation.private_attribute_matrix = (
                body_group.transformation.get_transformation_matrix().flatten().tolist()
            )


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

    def update_persistent_entities(self, *, asset_entity_registry: EntityRegistry) -> None:
        """
        1. Changed GenericVolume axis and center etc
        """

        for i_zone, _ in enumerate(self.zones):
            # pylint:disable = unsubscriptable-object
            assigned_zone = asset_entity_registry.find_by_asset_id(
                entity_id=self.zones[i_zone].id, entity_class=self.zones[i_zone].__class__
            )
            if assigned_zone is not None:
                # pylint:disable = unsupported-assignment-operation
                self.zones[i_zone] = assigned_zone

    def get_registry(self, internal_registry, **_) -> EntityRegistry:
        if internal_registry is None:
            # Initialize the local registry
            internal_registry = EntityRegistry()

            # Populate boundaries
            # pylint: disable=not-an-iterable
            for boundary in self.boundaries:
                internal_registry.register(boundary)

            # Populate zones
            # pylint: disable=not-an-iterable
            for zone in self.zones:
                internal_registry.register(zone)

        return internal_registry


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
        return self.boundaries

    def update_persistent_entities(self, *, asset_entity_registry: EntityRegistry) -> None:
        """
        Nothing related to SurfaceMeshEntityInfo for now.
        """
        return

    def get_registry(self, internal_registry, **_) -> EntityRegistry:
        if internal_registry is None:
            # Initialize the local registry
            internal_registry = EntityRegistry()
            # Populate boundaries
            # pylint: disable=not-an-iterable
            for boundary in self.boundaries:
                internal_registry.register(boundary)
            return internal_registry
        return internal_registry


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
