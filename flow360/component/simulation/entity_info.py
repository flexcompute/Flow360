"""Deserializer for entity info retrieved from asset metadata pipeline."""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Annotated, Dict, List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_registry import (
    EntityRegistry,
    SnappyBodyRegistry,
)
from flow360.component.simulation.outputs.output_entities import (
    Point,
    PointArray,
    PointArray2D,
    Slice,
)
from flow360.component.simulation.primitives import (
    AxisymmetricBody,
    Box,
    CustomVolume,
    Cylinder,
    Edge,
    GenericVolume,
    GeometryBodyGroup,
    GhostCircularPlane,
    GhostSphere,
    SnappyBody,
    Surface,
    WindTunnelGhostSurface,
)
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.utils import BoundingBoxType, model_attribute_unlock
from flow360.component.utils import GeometryFiles
from flow360.log import log

DraftEntityTypes = Annotated[
    Union[
        AxisymmetricBody,
        Box,
        Cylinder,
        Point,
        PointArray,
        PointArray2D,
        Slice,
        CustomVolume,
    ],
    pd.Field(discriminator="private_attribute_entity_type_name"),
]


class EntityInfoModel(Flow360BaseModel, metaclass=ABCMeta):
    """Base model for asset entity info JSON"""

    # entities that appear in simulation JSON but did not appear in EntityInfo
    draft_entities: List[DraftEntityTypes] = pd.Field([])
    ghost_entities: List[
        Annotated[
            Union[GhostSphere, GhostCircularPlane, WindTunnelGhostSurface],
            pd.Field(discriminator="private_attribute_entity_type_name"),
        ]
    ] = pd.Field([])

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
    def get_persistent_entity_registry(self, internal_registry, **kwargs):
        """
        Ensure that `internal_registry` exists and if not, initialize `internal_registry`.
        """


class BodyComponentInfo(Flow360BaseModel):
    """Data model for body component info."""

    face_ids: list[str] = pd.Field(
        description="A full list of face IDs that appear in the body.",
    )
    edge_ids: Optional[list[str]] = pd.Field(
        None,
        description="A full list of edge IDs that appear in the body. Optional for surface mesh geometry.",
    )


class GeometryEntityInfo(EntityInfoModel):
    """Data model for geometry entityInfo.json"""

    type_name: Literal["GeometryEntityInfo"] = pd.Field("GeometryEntityInfo", frozen=True)

    bodies_face_edge_ids: Optional[Dict[str, BodyComponentInfo]] = pd.Field(
        None,
        description="Mapping from body ID to the face and edge IDs of the body.",
    )
    # bodies_face_edge_ids: Mostly just used by front end. On python side this
    # is less useful as users do not operate on face/body/edge IDs directly.
    # But at least this can replace `face_ids`, `body_ids`, and `edge_ids` since these contains less info.

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

    global_bounding_box: Optional[BoundingBoxType] = pd.Field(None)

    # pylint: disable=no-member
    default_geometry_accuracy: Optional[LengthType.Positive] = pd.Field(
        None,
        description="The default value based on uploaded geometry for geometry_accuracy.",
    )

    @property
    def all_face_ids(self) -> list[str]:
        """
        Returns a full list of face IDs that appear in the geometry.
        Use `bodies_face_edge_ids` if available, otherwise fall back to use `face_ids`.
        """
        if self.bodies_face_edge_ids is not None:
            return [
                face_id
                for body_component_info in self.bodies_face_edge_ids.values()
                for face_id in body_component_info.face_ids
            ]
        return self.face_ids

    @property
    def all_edge_ids(self) -> list[str]:
        """
        Returns a full list of edge IDs that appear in the geometry.
        Use `bodies_face_edge_ids` if available, otherwise fall back to use `edge_ids`.
        """
        if self.bodies_face_edge_ids is not None:
            return [
                edge_id
                for body_component_info in self.bodies_face_edge_ids.values()
                # edge_ids can be None for surface-only geometry; treat it as an empty list.
                for edge_id in (body_component_info.edge_ids or [])
            ]
        return self.edge_ids

    @property
    def all_body_ids(self) -> list[str]:
        """
        Returns a full list of body IDs that appear in the geometry.
        Use `bodies_face_edge_ids` if available, otherwise fall back to use `body_ids`.
        """
        if self.bodies_face_edge_ids is not None:
            return list(self.bodies_face_edge_ids.keys())
        return self.body_ids

    def group_in_registry(
        self,
        entity_type_name: Literal["face", "edge", "body", "snappy_body"],
        attribute_name: str,
        registry: EntityRegistry,
    ) -> EntityRegistry:
        """
        Group items with given attribute_name.
        """
        entity_list = self._get_list_of_entities(attribute_name, entity_type_name)
        known_frozen_hashes = set()
        for item in entity_list:
            known_frozen_hashes = registry.fast_register(item, known_frozen_hashes)
        return registry

    def _get_snappy_bodies(self) -> List[SnappyBody]:

        snappy_body_mapping = {}
        for patch in self.grouped_faces[self.face_attribute_names.index("faceId")]:
            name_components = patch.name.split("::")
            body_name = name_components[0]
            if body_name not in snappy_body_mapping:
                snappy_body_mapping[body_name] = []
            if patch not in snappy_body_mapping[body_name]:
                snappy_body_mapping[body_name].append(patch)

        return [
            SnappyBody(name=snappy_body, surfaces=body_entities)
            for snappy_body, body_entities in snappy_body_mapping.items()
        ]

    def _get_list_of_entities(
        self,
        attribute_name: Union[str, None] = None,
        entity_type_name: Literal["face", "edge", "body", "snappy_body"] = None,
    ) -> Union[List[Surface], List[Edge], List[GeometryBodyGroup], List[SnappyBody]]:
        # Validations
        if entity_type_name is None:
            raise ValueError("Entity type name is required.")
        if entity_type_name not in ["face", "edge", "body", "snappy_body"]:
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
        elif entity_type_name == "body":
            entity_attribute_names = self.body_attribute_names
            entity_full_list = self.grouped_bodies
            specified_attribute_name = self.body_group_tag
        else:
            return self._get_snappy_bodies()

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

    def _group_faces_by_snappy_format(self):
        registry = SnappyBodyRegistry()

        registry = self.group_in_registry("snappy_body", attribute_name="faceId", registry=registry)

        return registry

    @pd.validate_call
    def _reset_grouping(
        self, entity_type_name: Literal["face", "edge", "body"], registry: EntityRegistry
    ) -> EntityRegistry:
        if entity_type_name == "face":
            registry.clear(Surface)
            registry.clear(SnappyBody)
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

    def get_persistent_entity_registry(self, internal_registry, **_) -> EntityRegistry:
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

            if len(self.all_edge_ids) > 0:
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
                    "body", body_group_tag, registry=internal_registry
                )
        return internal_registry

    def get_body_group_to_face_group_name_map(self) -> dict[str, list[str]]:
        """
        Returns bodyId to file name mapping.
        """

        # pylint: disable=too-many-locals
        def create_group_to_sub_component_mapping(group):
            mapping = defaultdict(list)
            for item in group:
                mapping[item.private_attribute_id].extend(item.private_attribute_sub_components)
            return mapping

        body_group_to_body = create_group_to_sub_component_mapping(
            self._get_list_of_entities(entity_type_name="body", attribute_name=self.body_group_tag)
        )
        boundary_to_face = create_group_to_sub_component_mapping(
            self._get_list_of_entities(entity_type_name="face", attribute_name=self.face_group_tag)
        )

        if "groupByBodyId" not in self.face_attribute_names:
            # This likely means the geometry asset is pre-25.5.
            raise ValueError(
                "Geometry cloud resource is too old."
                " Please consider re-uploading the geometry with newer solver version (>25.5)."
            )

        face_group_by_body_id_to_face = create_group_to_sub_component_mapping(
            self._get_list_of_entities(entity_type_name="face", attribute_name="groupByBodyId")
        )

        body_group_to_face = defaultdict(list)
        for body_group, body_ids in body_group_to_body.items():
            for body_id in body_ids:
                body_group_to_face[body_group].extend(face_group_by_body_id_to_face[body_id])

        face_to_body_group = {}
        for body_group_name, face_ids in body_group_to_face.items():
            for face_id in face_ids:
                face_to_body_group[face_id] = body_group_name

        body_group_to_boundary = defaultdict(list)
        for boundary_name, face_ids in boundary_to_face.items():
            body_group_in_this_face_group = set()
            for face_id in face_ids:
                owning_body = face_to_body_group.get(face_id)
                if owning_body is None:
                    raise ValueError(
                        f"Face ID '{face_id}' found in face group '{boundary_name}' "
                        "but not found in any body group."
                    )
                body_group_in_this_face_group.add(owning_body)
            if len(body_group_in_this_face_group) > 1:
                raise ValueError(
                    f"Face group '{boundary_name}' contains faces belonging to multiple body groups: "
                    f"{list(sorted(body_group_in_this_face_group))}. "
                    "The mapping between body and face groups cannot be created."
                )

            owning_body = list(body_group_in_this_face_group)[0]
            body_group_to_boundary[owning_body].append(boundary_name)

        return body_group_to_boundary

    def get_face_group_to_body_group_id_map(self) -> dict[str, str]:
        """
        Returns a mapping from face group (Surface) name to the owning body group ID.

        This is the inverse of :meth:`get_body_group_to_face_group_name_map` and uses the
        same underlying assumptions and validations about the grouping tags.
        """

        body_group_to_boundary = self.get_body_group_to_face_group_name_map()

        face_group_to_body_group: dict[str, str] = {}
        for body_group_id, boundary_names in body_group_to_boundary.items():
            for boundary_name in boundary_names:
                existing_owner = face_group_to_body_group.get(boundary_name)
                if existing_owner is not None and existing_owner != body_group_id:
                    raise ValueError(
                        f"[Internal] Face group '{boundary_name}' is mapped to multiple body groups: "
                        f"{existing_owner}, {body_group_id}. Data is likely corrupted."
                    )
                face_group_to_body_group[boundary_name] = body_group_id

        return face_group_to_body_group


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

    def get_persistent_entity_registry(self, internal_registry, **_) -> EntityRegistry:
        if internal_registry is None:
            # Initialize the local registry
            internal_registry = EntityRegistry()

            # Populate boundaries
            known_frozen_hashes = set()
            # pylint: disable=not-an-iterable
            for boundary in self.boundaries:
                known_frozen_hashes = internal_registry.fast_register(boundary, known_frozen_hashes)

            # Populate zones
            # pylint: disable=not-an-iterable
            known_frozen_hashes = set()
            for zone in self.zones:
                known_frozen_hashes = internal_registry.fast_register(zone, known_frozen_hashes)

        return internal_registry


class SurfaceMeshEntityInfo(EntityInfoModel):
    """Data model for surface mesh entityInfo.json"""

    type_name: Literal["SurfaceMeshEntityInfo"] = pd.Field("SurfaceMeshEntityInfo", frozen=True)
    boundaries: list[Surface] = pd.Field([])
    global_bounding_box: Optional[BoundingBoxType] = pd.Field(None)

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

    def get_persistent_entity_registry(self, internal_registry, **_) -> EntityRegistry:
        if internal_registry is None:
            # Initialize the local registry
            internal_registry = EntityRegistry()
            known_frozen_hashes = set()
            # Populate boundaries
            # pylint: disable=not-an-iterable
            for boundary in self.boundaries:
                known_frozen_hashes = internal_registry.fast_register(boundary, known_frozen_hashes)
            return internal_registry
        return internal_registry


EntityInfoUnion = Annotated[
    Union[GeometryEntityInfo, VolumeMeshEntityInfo, SurfaceMeshEntityInfo],
    pd.Field(discriminator="type_name"),
]


def parse_entity_info_model(data) -> EntityInfoUnion:
    """
    parse entity info data and return one of [GeometryEntityInfo, VolumeMeshEntityInfo, SurfaceMeshEntityInfo]

    # TODO: Add a fast mode by popping entities that are not needed due to wrong grouping tags before deserialization.
    """
    return pd.TypeAdapter(EntityInfoUnion).validate_python(data)


def update_geometry_entity_info(
    current_entity_info: GeometryEntityInfo,
    entity_info_components: List[GeometryEntityInfo],
) -> GeometryEntityInfo:
    """
    Update a GeometryEntityInfo by including/merging data from a list of other GeometryEntityInfo objects.

    Args:
        current_entity_info: Used as reference to preserve user settings such as group tags,
            mesh_exterior, attribute name order.
        entity_info_components: List of GeometryEntityInfo objects that contain all data for the new entity info

    Returns:
        A new GeometryEntityInfo with merged data from entity_info_components, preserving user settings from current

    The merge logic:
    1. IDs: Union of body_ids, face_ids, edge_ids from entity_info_components
    2. Attribute names: Intersection of attribute_names from entity_info_components
    3. Group tags: Use tags from current_entity_info
    4. Bounding box: Merge global bounding boxes from entity_info_components
    5. Grouped entities: Merge from entity_info_components,
        preserving mesh_exterior from current_entity_info for grouped_bodies
    6. Draft and Ghost entities: Preserve from current_entity_info.
    """
    # pylint: disable=too-many-locals, too-many-statements
    if not entity_info_components:
        raise ValueError("entity_info_components cannot be empty")

    # 1. Compute union of IDs from entity_info_components
    all_body_ids = set()
    all_face_ids = set()
    all_edge_ids = set()
    all_bodies_face_edge_ids = {}

    for entity_info in entity_info_components:
        all_body_ids.update(entity_info.body_ids)
        all_face_ids.update(entity_info.face_ids)
        all_edge_ids.update(entity_info.edge_ids)
        all_bodies_face_edge_ids.update(entity_info.bodies_face_edge_ids or {})

    # 2. Compute intersection of attribute names from entity_info_components
    body_attr_sets = [set(ei.body_attribute_names) for ei in entity_info_components]
    face_attr_sets = [set(ei.face_attribute_names) for ei in entity_info_components]
    edge_attr_sets = [set(ei.edge_attribute_names) for ei in entity_info_components]

    body_attr_intersection = set.intersection(*body_attr_sets) if body_attr_sets else set()
    face_attr_intersection = set.intersection(*face_attr_sets) if face_attr_sets else set()
    edge_attr_intersection = set.intersection(*edge_attr_sets) if edge_attr_sets else set()

    # Preserve order from current_entity_info, but include all attributes from intersection
    def ordered_intersection(reference_list: List[str], intersection_set: set) -> List[str]:
        """Return all attributes from intersection_set, preserving order from reference_list where possible."""
        # First, add attributes that exist in reference_list (in order)
        result = [attr for attr in reference_list if attr in intersection_set]
        # Then, add remaining attributes from intersection_set that weren't in reference_list (sorted)
        remaining = sorted(intersection_set - set(result))
        return result + remaining

    result_body_attribute_names = ordered_intersection(
        current_entity_info.body_attribute_names, body_attr_intersection
    )
    result_face_attribute_names = ordered_intersection(
        current_entity_info.face_attribute_names, face_attr_intersection
    )
    result_edge_attribute_names = ordered_intersection(
        current_entity_info.edge_attribute_names, edge_attr_intersection
    )

    # 3. Update group tags: preserve from current if exists in intersection, otherwise use first
    def select_tag(
        current_tag: Optional[str], result_attrs: List[str], entity_type: str
    ) -> Optional[str]:
        if entity_type != "edge" and not result_attrs:
            raise ValueError(f"No attribute names available to select {entity_type} group tag.")
        log.info(f"Preserving {entity_type} group tag: {current_tag}")
        return current_tag

    result_body_group_tag = select_tag(
        current_entity_info.body_group_tag, result_body_attribute_names, "body"
    )
    result_face_group_tag = select_tag(
        current_entity_info.face_group_tag, result_face_attribute_names, "face"
    )
    result_edge_group_tag = select_tag(
        current_entity_info.edge_group_tag, result_edge_attribute_names, "edge"
    )

    # 4. Merge global bounding boxes from entity_info_components
    result_bounding_box = None
    for entity_info in entity_info_components:
        if entity_info.global_bounding_box is not None:
            if result_bounding_box is None:
                result_bounding_box = entity_info.global_bounding_box
            else:
                result_bounding_box = result_bounding_box.expand(entity_info.global_bounding_box)

    # Build mapping of body group ID to mesh_exterior from current_entity_info
    current_body_user_settings_map = {}
    for body_group_idx, body_group_name in enumerate(current_entity_info.body_attribute_names):
        current_body_user_settings_map[body_group_name] = {}
        for body in current_entity_info.grouped_bodies[body_group_idx]:
            body_id = body.private_attribute_id
            current_body_user_settings_map[body_group_name][body_id] = {
                "mesh_exterior": body.mesh_exterior,
                "name": body.name,
            }

    # 5. Merge grouped entities from entity_info_components
    def merge_grouped_entities(
        entity_type: Literal["body", "face", "edge"],
        result_attr_names: List[str],
    ):
        """Helper to merge grouped entities (bodies, faces, or edges) from entity_info_components"""

        # Determine which attributes to access based on entity type
        def get_attrs(entity_info):
            if entity_type == "body":
                return entity_info.body_attribute_names
            if entity_type == "face":
                return entity_info.face_attribute_names
            return entity_info.edge_attribute_names

        def get_groups(entity_info):
            if entity_type == "body":
                return entity_info.grouped_bodies
            if entity_type == "face":
                return entity_info.grouped_faces
            return entity_info.grouped_edges

        result_grouped = []

        # For each attribute name in the result intersection
        for attr_name in result_attr_names:
            # Dictionary to accumulate entities by their unique ID
            entity_map = {}

            # Process all include entity infos
            for entity_info in entity_info_components:
                entity_attrs = get_attrs(entity_info)
                if attr_name not in entity_attrs:
                    continue
                idx = entity_attrs.index(attr_name)
                entity_groups = get_groups(entity_info)
                for entity in entity_groups[idx]:
                    # Use private_attribute_id as the unique identifier
                    entity_id = entity.private_attribute_id
                    if entity_id not in entity_map:
                        # For bodies, check if we need to preserve mesh_exterior
                        if (
                            entity_type == "body"
                            and attr_name in current_body_user_settings_map
                            and entity_id in current_body_user_settings_map[attr_name]
                        ):
                            # Create a copy with preserved mesh_exterior
                            entity_data = entity.model_dump()
                            entity_data["mesh_exterior"] = current_body_user_settings_map[
                                attr_name
                            ][entity_id]["mesh_exterior"]
                            entity_data["name"] = current_body_user_settings_map[attr_name][
                                entity_id
                            ]["name"]
                            entity_map[entity_id] = GeometryBodyGroup.model_validate(entity_data)
                        else:
                            entity_map[entity_id] = entity

            # Convert map to list, maintaining a stable order (sorted by entity ID)
            result_grouped.append(sorted(entity_map.values(), key=lambda e: e.private_attribute_id))

        return result_grouped

    result_grouped_bodies = merge_grouped_entities("body", result_body_attribute_names)
    result_grouped_faces = merge_grouped_entities("face", result_face_attribute_names)
    result_grouped_edges = merge_grouped_entities("edge", result_edge_attribute_names)

    # Use default_geometry_accuracy from first include_entity_info
    result_default_geometry_accuracy = entity_info_components[0].default_geometry_accuracy

    # Create the result GeometryEntityInfo
    result = GeometryEntityInfo(
        bodies_face_edge_ids=all_bodies_face_edge_ids if all_bodies_face_edge_ids else None,
        body_ids=sorted(all_body_ids),
        body_attribute_names=result_body_attribute_names,
        grouped_bodies=result_grouped_bodies,
        face_ids=sorted(all_face_ids),
        face_attribute_names=result_face_attribute_names,
        grouped_faces=result_grouped_faces,
        edge_ids=sorted(all_edge_ids),
        edge_attribute_names=result_edge_attribute_names,
        grouped_edges=result_grouped_edges,
        body_group_tag=result_body_group_tag,
        face_group_tag=result_face_group_tag,
        edge_group_tag=result_edge_group_tag,
        global_bounding_box=result_bounding_box,
        default_geometry_accuracy=result_default_geometry_accuracy,
        draft_entities=current_entity_info.draft_entities,
        ghost_entities=current_entity_info.ghost_entities,
    )

    return result
