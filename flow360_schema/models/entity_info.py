"""Deserializer for entity info retrieved from asset metadata pipeline."""

import logging
import re
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from typing import Annotated, Any, Literal, TypeVar

import pydantic as pd

from flow360_schema.exceptions import Flow360ValueError
from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.bounding_box import BoundingBox, BoundingBoxType
from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_registry import EntityRegistry
from flow360_schema.framework.physical_dimensions import Length
from flow360_schema.framework.validation.context import DeserializationContext
from flow360_schema.models.entities.geometry_entities import (
    Edge,
    GeometryBodyGroup,
)
from flow360_schema.models.entities.output_entities import (
    Point,
    PointArray,
    PointArray2D,
    Slice,
)
from flow360_schema.models.entities.surface_entities import (
    GhostCircularPlane,
    GhostSphere,
    Surface,
    WindTunnelGhostSurface,
)
from flow360_schema.models.entities.volume_entities import (
    AxisymmetricBody,
    Box,
    CustomVolume,
    Cylinder,
    GenericVolume,
    SeedpointVolume,
    Sphere,
    VoxelGrid,
)

logger = logging.getLogger(__name__)

DraftEntityTypes = Annotated[
    AxisymmetricBody
    | Box
    | VoxelGrid
    | Cylinder
    | Sphere
    | Point
    | PointArray
    | PointArray2D
    | Slice
    | CustomVolume
    | SeedpointVolume,
    pd.Field(discriminator="private_attribute_entity_type_name"),
]


def iter_entity_dicts_from_entity_info_dict(
    entity_info_dict: dict[str, Any],
    *,
    grouped: Literal["all", "active"],
) -> Iterator[dict[str, Any]]:
    """Yield every entity dict stored in a RAW entity_info dict.

    Covers ``boundaries``/``zones``/``draft_entities``/``ghost_entities`` plus the
    geometry grouped lists. ``grouped="active"`` yields only the grouping selected
    by each ``*_group_tag`` (mirroring what ``EntityRegistry.from_entity_info``
    registers at load time); ``grouped="all"`` yields every grouping.
    """
    for key in ("boundaries", "zones", "draft_entities", "ghost_entities"):
        for entity in entity_info_dict.get(key) or []:
            if isinstance(entity, dict):
                yield entity

    for tag_key, names_key, grouped_key in (
        ("face_group_tag", "face_attribute_names", "grouped_faces"),
        ("edge_group_tag", "edge_attribute_names", "grouped_edges"),
        ("body_group_tag", "body_attribute_names", "grouped_bodies"),
    ):
        grouped_items = entity_info_dict.get(grouped_key) or []
        if grouped == "active":
            group_tag = entity_info_dict.get(tag_key)
            attribute_names = entity_info_dict.get(names_key) or []
            if not group_tag or group_tag not in attribute_names:
                continue
            index = attribute_names.index(group_tag)
            grouped_items = [grouped_items[index]] if index < len(grouped_items) else []
        for group in grouped_items:
            for entity in group or []:
                if isinstance(entity, dict):
                    yield entity


# The only types `CustomVolume.bounding_entities` accepts.
_BOUNDING_ENTITY_TYPE_NAMES = frozenset({"Surface", "Cylinder", "AxisymmetricBody", "Sphere"})


def _expand_bounding_entity_reference_groups(entity_info_dict: dict[str, Any]) -> dict[str, Any]:
    """Expand compact reference groups inside ``CustomVolume.bounding_entities`` in-place.

    ``bounding_entities`` is the one EntityList living INSIDE entity_info, so its
    compact references cannot wait for the registry-backed materializer (the
    registry is built FROM this very dict). They are expanded dict-level against
    the entity_info's own lists before the single pydantic parse. The expanded
    entities are parsed copies of their definitions — wire-level binding via the
    reference plus entity_info-as-source-of-truth is the consistency guarantee;
    in-memory aliasing is not required here.
    """
    holders = [
        entity
        for key in ("zones", "draft_entities")
        for entity in entity_info_dict.get(key) or []
        if isinstance(entity, dict) and isinstance(entity.get("bounding_entities"), dict)
    ]
    if not any(
        isinstance(item, dict) and set(item.keys()) == {"type", "ids"}
        for holder in holders
        for item in holder["bounding_entities"].get("stored_entities") or []
    ):
        return entity_info_dict

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for entity in iter_entity_dicts_from_entity_info_dict(entity_info_dict, grouped="all"):
        type_name = entity.get("private_attribute_entity_type_name")
        entity_id = entity.get("private_attribute_id")
        if type_name in _BOUNDING_ENTITY_TYPE_NAMES and isinstance(entity_id, str):
            index.setdefault((type_name, entity_id), entity)

    for holder in holders:
        stored = holder["bounding_entities"].get("stored_entities") or []
        expanded: list[Any] = []
        for item in stored:
            if not (isinstance(item, dict) and set(item.keys()) == {"type", "ids"}):
                expanded.append(item)
                continue
            for entity_id in item["ids"]:
                referenced = index.get((item["type"], entity_id))
                if referenced is None:
                    raise ValueError(
                        f"[EntityInfo] bounding_entities reference (type: {item['type']}, "
                        f"id: {entity_id}) of CustomVolume '{holder.get('name')}' has no "
                        "definition in the entity info."
                    )
                expanded.append(referenced)
        holder["bounding_entities"]["stored_entities"] = expanded
    return entity_info_dict


class EntityInfoModel(Flow360BaseModel, metaclass=ABCMeta):
    """Base model for asset entity info JSON"""

    # entities that appear in simulation JSON but did not appear in EntityInfo
    draft_entities: list[DraftEntityTypes] = pd.Field([])
    ghost_entities: list[
        Annotated[
            GhostSphere | GhostCircularPlane | WindTunnelGhostSurface,
            pd.Field(discriminator="private_attribute_entity_type_name"),
        ]
    ] = pd.Field([])

    @pd.model_validator(mode="before")
    @classmethod
    def _expand_bounding_reference_groups(cls, values: Any) -> Any:
        if isinstance(values, dict):
            return _expand_bounding_entity_reference_groups(values)
        return values

    @abstractmethod
    def get_boundaries(self, attribute_name: str | None = None) -> list[Surface]:
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
        # TODO: consider removing mutation-based approach in favor of immutable pattern (D4)

    @abstractmethod
    def get_persistent_entity_registry(self, internal_registry: EntityRegistry | None, **kwargs: Any) -> EntityRegistry:
        """
        Ensure that `internal_registry` exists and if not, initialize `internal_registry`.
        """


class BodyComponentInfo(Flow360BaseModel):
    """Data model for body component info."""

    face_ids: list[str] = pd.Field(
        description="A full list of face IDs that appear in the body.",
    )
    edge_ids: list[str] | None = pd.Field(
        None,
        description="A full list of edge IDs that appear in the body. Optional for surface mesh geometry.",
    )


class GeometryEntityInfo(EntityInfoModel):
    """Data model for geometry entityInfo.json"""

    type_name: Literal["GeometryEntityInfo"] = pd.Field("GeometryEntityInfo", frozen=True)

    bodies_face_edge_ids: dict[str, BodyComponentInfo] | None = pd.Field(
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
    body_attribute_names: list[str] = pd.Field(
        [],
        description="A full list of attribute names that the user can"
        "select to achieve grouping of bodies. It has same length as `grouped_bodies`",
        alias="bodyAttributeNames",
    )
    grouped_bodies: list[list[GeometryBodyGroup]] = pd.Field(
        [[]],
        description="The resulting list of `GeometryBodyGroup` entities after grouping using the attribute name.",
        alias="groupedBodies",
    )

    face_ids: list[str] = pd.Field(
        [],
        description="A full list of faceIDs/model IDs that appear in the geometry.",
        alias="faceIDs",
    )
    face_attribute_names: list[str] = pd.Field(
        [],
        description="A full list of attribute names that the user can"
        "select to achieve grouping of faces. It has same length as `grouped_faces`",
        alias="faceAttributeNames",
    )
    grouped_faces: list[list[Surface]] = pd.Field(
        [[]],
        description="The resulting list of `Surface` entities after grouping using the attribute name.",
        alias="groupedFaces",
    )

    edge_ids: list[str] = pd.Field(
        [],
        description="A full list of edgeIDs/model IDs that appear in the geometry.",
        alias="edgeIDs",
    )
    edge_attribute_names: list[str] = pd.Field(
        [],
        description="A full list of attribute names that the user can"
        "select to achieve grouping of edges. It has same length as `grouped_edges`",
        alias="edgeAttributeNames",
    )
    grouped_edges: list[list[Edge]] = pd.Field(
        [[]],
        description="The resulting list of `Edge` entities after grouping using the attribute name.",
        alias="groupedEdges",
    )

    body_group_tag: str | None = pd.Field(None, frozen=True)
    face_group_tag: str | None = pd.Field(None, frozen=True)
    edge_group_tag: str | None = pd.Field(None, frozen=True)

    global_bounding_box: BoundingBoxType | None = pd.Field(None)

    default_geometry_accuracy: Length.PositiveFloat64 | None = pd.Field(  # type: ignore[valid-type]
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
        entity_type_name: Literal["face", "edge", "body"],
        attribute_name: str,
        registry: EntityRegistry,
    ) -> EntityRegistry:
        """
        Group items with given attribute_name.
        """
        entity_list = self._get_list_of_entities(attribute_name, entity_type_name)
        known_entity_keys: set[tuple[Any, ...]] = set()
        for item in entity_list:
            known_entity_keys = registry.fast_register(item, known_entity_keys)
        return registry

    def _get_list_of_entities(
        self,
        attribute_name: str | None = None,
        entity_type_name: Literal["face", "edge", "body"] | None = None,
    ) -> list[Surface] | list[Edge] | list[GeometryBodyGroup]:
        # Validations
        if entity_type_name is None:
            raise ValueError("Entity type name is required.")
        if entity_type_name not in ["face", "edge", "body"]:
            raise ValueError(f"Invalid entity type name, expected 'body, 'face' or 'edge' but got {entity_type_name}.")
        if entity_type_name == "face":
            entity_attribute_names = self.face_attribute_names
            entity_full_list: list[list[Surface]] | list[list[Edge]] | list[list[GeometryBodyGroup]] = (
                self.grouped_faces
            )
            specified_attribute_name = self.face_group_tag
        elif entity_type_name == "edge":
            entity_attribute_names = self.edge_attribute_names
            entity_full_list = self.grouped_edges
            specified_attribute_name = self.edge_group_tag
        elif entity_type_name == "body":
            entity_attribute_names = self.body_attribute_names
            entity_full_list = self.grouped_bodies
            specified_attribute_name = self.body_group_tag

        # Use the supplied one if not None
        if attribute_name is not None:
            specified_attribute_name = attribute_name

        if specified_attribute_name in entity_attribute_names:
            return entity_full_list[entity_attribute_names.index(specified_attribute_name)]

        raise ValueError(
            f"The given attribute_name `{attribute_name}` is not found"
            f" in geometry metadata. Available: {entity_attribute_names}"
        )

    def get_boundaries(self, attribute_name: str | None = None) -> list[Surface]:
        """
        Get the full list of boundaries.
        If attribute_name is supplied then ignore stored face_group_tag and use supplied one.
        """
        return self._get_list_of_entities(attribute_name, "face")  # type: ignore[return-value]

    def update_persistent_entities(self, *, asset_entity_registry: EntityRegistry) -> None:
        """
        Update the persistent entities stored inside `self` according to `asset_entity_registry`
        """
        # TODO: consider removing mutation-based approach in favor of immutable pattern (D4)

        def _search_and_replace(grouped_entities: list[list[Any]], entity_registry: EntityRegistry) -> None:
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

    # TOAI: THIS REALLY IS MORE OF A client side function. Should just take in an GeomtryEntityInfo instance instead.
    def _get_processed_file_list(self) -> tuple[list[str], list[str]]:
        """
        Return the list of files that are uploaded by geometryConversionPipeline.

        This function examines the files mentioned under `grouped_bodies->groupByFile`
        and append folder prefix if necessary.
        """
        # Lazy import: these are client-side utilities for file format validation
        from flow360.component.utils import (  # type: ignore[import-not-found]
            GeometryFiles,
            MeshNameParser,
        )

        # Reverse-engineers the root file name, as a compromise: both fields carrying it
        # drift away from it (the id becomes the minted "model{N}" identity, the name is
        # renameable), so take whichever still looks like a model file. The direction is
        # to improve the geometry resource's model metadata -- ModelFileMetadata's
        # index-keyed modelType/outputFiles -- and have these readers consume that.
        def _root_file_name(body_group: EntityBase) -> str:
            for candidate in (body_group.name, body_group.private_attribute_id):
                if candidate and (
                    GeometryFiles.check_is_valid_geometry_file_format(file_name=candidate)
                    or MeshNameParser(candidate).is_valid_surface_mesh()
                ):
                    return candidate
            return body_group.name

        body_groups_grouped_by_file = self._get_list_of_entities("groupByFile", "body")
        unprocessed_file_names = [_root_file_name(item) for item in body_groups_grouped_by_file]
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

    def _get_default_grouping_tag(self, entity_type_name: Literal["face", "edge", "body"]) -> str:
        """
        Returns the default grouping tag for the given entity type.
        The selection logic is intended to mimic the webUI behavior.
        """

        def _get_the_first_non_id_tag(
            attribute_names: list[str], entity_type_name: Literal["face", "edge", "body"]
        ) -> str:
            if not attribute_names:
                raise ValueError(f"[Internal] No valid tag available for grouping {entity_type_name}.")
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
        registry: EntityRegistry | None = None,
    ) -> EntityRegistry:
        if entity_type_name not in ["face", "edge", "body"]:
            raise ValueError(
                f"[Internal] Unknown entity type: `{entity_type_name}`, allowed entity: 'face', 'edge', 'body'."
            )

        if registry is None:
            registry = EntityRegistry()  # type: ignore[call-arg]

        existing_tag = None
        if entity_type_name == "face" and self.face_group_tag is not None:
            existing_tag = self.face_group_tag

        elif entity_type_name == "edge" and self.edge_group_tag is not None:
            existing_tag = self.edge_group_tag

        elif entity_type_name == "body" and self.body_group_tag is not None:
            existing_tag = self.body_group_tag

        if existing_tag:
            if existing_tag != tag_name:
                logger.info(
                    "Regrouping %s entities under `%s` tag (previous `%s`).",
                    entity_type_name,
                    tag_name,
                    existing_tag,
                )
            registry = self._reset_grouping(entity_type_name=entity_type_name, registry=registry)

        registry = self.group_in_registry(entity_type_name, attribute_name=tag_name, registry=registry)
        if entity_type_name == "face":
            self._force_set_attr("face_group_tag", tag_name)
        elif entity_type_name == "edge":
            self._force_set_attr("edge_group_tag", tag_name)
        else:
            self._force_set_attr("body_group_tag", tag_name)

        return registry

    @pd.validate_call
    def _reset_grouping(
        self, entity_type_name: Literal["face", "edge", "body"], registry: EntityRegistry
    ) -> EntityRegistry:
        if entity_type_name == "face":
            registry.clear(Surface)
            self._force_set_attr("face_group_tag", None)
        elif entity_type_name == "edge":
            registry.clear(Edge)
            self._force_set_attr("edge_group_tag", None)
        else:
            registry.clear(GeometryBodyGroup)
            self._force_set_attr("body_group_tag", None)
        return registry

    def get_persistent_entity_registry(self, internal_registry: EntityRegistry | None, **_: Any) -> EntityRegistry:
        if internal_registry is None:
            internal_registry = EntityRegistry()  # type: ignore[call-arg]
            if self.face_group_tag is None:
                face_group_tag = self._get_default_grouping_tag("face")
                logger.info("Using `%s` as default grouping for faces.", face_group_tag)
            else:
                face_group_tag = self.face_group_tag

            internal_registry = self._group_entity_by_tag("face", face_group_tag, registry=internal_registry)

            if len(self.all_edge_ids) > 0:
                if self.edge_group_tag is None:
                    edge_group_tag = self._get_default_grouping_tag("edge")
                    logger.info("Using `%s` as default grouping for edges.", edge_group_tag)
                else:
                    edge_group_tag = self.edge_group_tag

                internal_registry = self._group_entity_by_tag("edge", edge_group_tag, registry=internal_registry)

            if self.body_attribute_names:
                # Post-25.5 geometry asset. For Pre 25.5 we just skip body grouping.
                if self.body_group_tag is None:
                    body_group_tag = self._get_default_grouping_tag("body")
                    logger.info("Using `%s` as default grouping for bodies.", body_group_tag)
                else:
                    body_group_tag = self.body_group_tag

                internal_registry = self._group_entity_by_tag("body", body_group_tag, registry=internal_registry)
        return internal_registry

    def get_num_exterior_bodies(self) -> int:
        """Number of individual bodies whose active body group has ``mesh_exterior=True``,
        or 0 when body grouping is unresolved (no tag, or a tag absent from the metadata)."""
        if not self.body_attribute_names:
            return 0
        body_group_tag = self.body_group_tag or self._get_default_grouping_tag("body")
        try:
            body_groups: list[Any] = self._get_list_of_entities(entity_type_name="body", attribute_name=body_group_tag)
        except ValueError:
            return 0
        return sum(len(group.private_attribute_sub_components) for group in body_groups if group.mesh_exterior)

    def get_body_group_to_surface_mapping(self) -> dict[str, list[str]]:
        """
        Return body group's (id, name) to Surfaces' (face groups') (id, name) mapping
        """

        def create_group_to_sub_component_mapping(group: list[Any]) -> dict[str, tuple[str, list[str]]]:
            return {item.private_attribute_id: (item.name, item.private_attribute_sub_components) for item in group}

        # body_group_id to (body_group_name, body_ids) of the current body group
        body_group_to_body = create_group_to_sub_component_mapping(
            self._get_list_of_entities(entity_type_name="body", attribute_name=self.body_group_tag)
        )
        # surface_id to (surface_name, face_ids) of the current face group
        surface_to_face = create_group_to_sub_component_mapping(
            self._get_list_of_entities(entity_type_name="face", attribute_name=self.face_group_tag)
        )

        # Create body id to face ids mapping:
        if self.bodies_face_edge_ids:
            # With bodies_face_edge_ids
            body_id_to_face_ids = {
                body_id: body_component_info.face_ids
                for body_id, body_component_info in self.bodies_face_edge_ids.items()
            }
        else:
            # Fallback: With the face group:"groupByBodyId" where face_group_name is body_id
            if "groupByBodyId" not in self.face_attribute_names:
                # This likely means the geometry asset is pre-25.5.
                raise Flow360ValueError(
                    "Geometry cloud resource is too old."
                    " Please consider re-uploading the geometry with newer solver version (>25.5)."
                )
            body_id_to_face_ids = {
                face_group.name: face_group.private_attribute_sub_components  # type: ignore[misc]
                for face_group in self._get_list_of_entities(entity_type_name="face", attribute_name="groupByBodyId")
            }

        # body_group_id to (body_group_name, face_ids) of the current body group
        body_group_to_face = {}
        for body_group_id, (body_group_name, body_ids) in body_group_to_body.items():
            face_ids = []
            for body_id in body_ids:
                face_ids.extend(body_id_to_face_ids[body_id])
            body_group_to_face[body_group_id] = (body_group_name, face_ids)

        # face_id to (body_group_id, body_group_name) of the current body group
        face_to_body_group = {}
        for body_group_id, (body_group_name, face_ids) in body_group_to_face.items():
            for face_id in face_ids:
                face_to_body_group[face_id] = (body_group_id, body_group_name)

        # body_group (id, name) to surface (id, name)
        body_group_to_surface: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for surface_id, (surface_name, face_ids) in surface_to_face.items():
            body_group_in_this_face_group = set()
            for face_id in face_ids:
                owning_body = face_to_body_group.get(face_id)
                if owning_body is None:
                    raise Flow360ValueError(
                        f"Face ID '{face_id}' found in face group '{surface_name}' but not found in any body group."
                    )
                body_group_in_this_face_group.add(owning_body)
            if len(body_group_in_this_face_group) > 1:
                raise Flow360ValueError(
                    f"Face group '{surface_name}' contains faces belonging to multiple body groups: "
                    f"{sorted(body_group_in_this_face_group)}. "
                    "The mapping between body and face groups cannot be created."
                )

            owning_body = list(body_group_in_this_face_group)[0]
            if owning_body not in body_group_to_surface:
                body_group_to_surface[owning_body] = []
            body_group_to_surface[owning_body].append((surface_id, surface_name))

        return body_group_to_surface  # type: ignore[return-value]

    def get_body_group_to_face_group_name_map(self) -> dict[str, list[str]]:
        """
        Returns body group name to face group (Surface) name mapping.
        """

        body_group_to_surface: dict[tuple[str, str], list[tuple[str, str]]] = self.get_body_group_to_surface_mapping()  # type: ignore[assignment]
        body_group_to_surface_name: defaultdict[str, list[str]] = defaultdict(list)

        for (_, body_group_name), boundaries in body_group_to_surface.items():
            body_group_to_surface_name[body_group_name].extend([surface_name for (_, surface_name) in boundaries])

        return body_group_to_surface_name

    def get_face_group_to_body_group_id_map(self) -> dict[str, str]:
        """
        Returns a mapping from face group (Surface) name to the owning body group ID.

        This is the inverse of :meth:`get_body_group_to_surface_mapping` and uses the
        same underlying assumptions and validations about the grouping tags.
        """

        body_group_to_surface: dict[tuple[str, str], list[tuple[str, str]]] = self.get_body_group_to_surface_mapping()  # type: ignore[assignment]
        face_group_to_body_group: dict[str, str] = {}
        for (body_group_id, _), surfaces in body_group_to_surface.items():
            for _, surface_name in surfaces:
                existing_owner = face_group_to_body_group.get(surface_name)
                if existing_owner is not None and existing_owner != body_group_id:
                    raise ValueError(
                        f"[Internal] Face group '{surface_name}' is mapped to multiple body groups: "
                        f"{existing_owner}, {body_group_id}. Data is likely corrupted."
                    )
                face_group_to_body_group[surface_name] = body_group_id

        return face_group_to_body_group


class VolumeMeshEntityInfo(EntityInfoModel):
    """Data model for volume mesh entityInfo.json"""

    type_name: Literal["VolumeMeshEntityInfo"] = pd.Field("VolumeMeshEntityInfo", frozen=True)
    zones: list[GenericVolume] = pd.Field([])
    boundaries: list[Surface] = pd.Field([])

    @pd.field_validator("boundaries", mode="after")
    @classmethod
    def check_all_surface_has_interface_indicator(cls, value: list[Surface]) -> list[Surface]:
        """private_attribute_is_interface should have been set coming from volume mesh."""
        for item in value:
            if item.private_attribute_is_interface is None:
                raise ValueError(f"[INTERNAL] {item.name} is missing private_attribute_is_interface attribute!.")
        return value

    def get_boundaries(self, attribute_name: str | None = None) -> list[Any]:
        """
        Get the full list of boundary.
        """
        return [item for item in self.boundaries if item.private_attribute_is_interface is False]

    def update_persistent_entities(self, *, asset_entity_registry: EntityRegistry) -> None:
        """
        1. Changed GenericVolume axis and center etc
        """
        # TODO: consider removing mutation-based approach in favor of immutable pattern (D4)

        for i_zone, _ in enumerate(self.zones):
            assigned_zone = asset_entity_registry.find_by_asset_id(
                entity_id=self.zones[i_zone].id, entity_class=self.zones[i_zone].__class__
            )
            if assigned_zone is not None:
                self.zones[i_zone] = assigned_zone  # type: ignore[call-overload]

    def get_persistent_entity_registry(self, internal_registry: EntityRegistry | None, **_: Any) -> EntityRegistry:
        if internal_registry is None:
            # Initialize the local registry
            internal_registry = EntityRegistry()  # type: ignore[call-arg]

            # Populate boundaries
            known_entity_keys: set[tuple[Any, ...]] = set()
            for boundary in self.boundaries:
                known_entity_keys = internal_registry.fast_register(boundary, known_entity_keys)

            # Populate zones
            known_entity_keys = set()
            for zone in self.zones:
                known_entity_keys = internal_registry.fast_register(zone, known_entity_keys)

        return internal_registry


class SurfaceMeshEntityInfo(EntityInfoModel):
    """Data model for surface mesh entityInfo.json"""

    type_name: Literal["SurfaceMeshEntityInfo"] = pd.Field("SurfaceMeshEntityInfo", frozen=True)
    boundaries: list[Surface] = pd.Field([])
    global_bounding_box: BoundingBoxType | None = pd.Field(None)

    def get_boundaries(self, attribute_name: str | None = None) -> list[Any]:
        """
        Get the full list of boundary.
        """
        return self.boundaries

    def update_persistent_entities(self, *, asset_entity_registry: EntityRegistry) -> None:
        """
        Nothing related to SurfaceMeshEntityInfo for now.
        """
        # TODO: consider removing mutation-based approach in favor of immutable pattern (D4)
        return

    def get_persistent_entity_registry(self, internal_registry: EntityRegistry | None, **_: Any) -> EntityRegistry:
        if internal_registry is None:
            # Initialize the local registry
            internal_registry = EntityRegistry()  # type: ignore[call-arg]
            known_entity_keys: set[tuple[Any, ...]] = set()
            # Populate boundaries
            for boundary in self.boundaries:
                known_entity_keys = internal_registry.fast_register(boundary, known_entity_keys)
            return internal_registry
        return internal_registry


EntityInfoUnion = Annotated[
    GeometryEntityInfo | VolumeMeshEntityInfo | SurfaceMeshEntityInfo,
    pd.Field(discriminator="type_name"),
]


def parse_entity_info_model(data: dict[str, Any]) -> EntityInfoUnion:
    """
    parse entity info data and return one of [GeometryEntityInfo, VolumeMeshEntityInfo, SurfaceMeshEntityInfo]

    # TODO: Add a fast mode by popping entities that are not needed due to wrong grouping tags before deserialization.
    """
    with DeserializationContext():
        return pd.TypeAdapter(EntityInfoUnion).validate_python(data)


# Entity types appearing in grouped_bodies/grouped_faces/grouped_edges; value-restricted so
# _prune returns the same grouping type it receives.
_GroupedEntityT = TypeVar("_GroupedEntityT", GeometryBodyGroup, Surface, Edge)


_RESOURCE_SCOPED_MODEL_ID_PREFIX = "model"


def _resource_scoped_model_id(model_index: int) -> str:
    """The bare, resource-scoped model identity minted by conversion: ``"model{N}"``.

    N is the model's index into ``ModelFileMetadata.models`` (rootFile order). Conversion
    (geometryMeta / _inject_cad_group_by_file) mints this onto each ``groupByFile`` body
    group's ``private_attribute_id``; the merge below keys exclusion and cross-geometry
    identity on it.
    """
    return f"{_RESOURCE_SCOPED_MODEL_ID_PREFIX}{model_index}"


_RESOURCE_SCOPED_MODEL_ID_PATTERN = re.compile(rf"{_RESOURCE_SCOPED_MODEL_ID_PREFIX}\d+")


def _is_resource_scoped_model_id(private_attribute_id: str) -> bool:
    """True for a bare minted ``"model{N}"`` id (not a legacy name, not an already-composed id).

    A digit class rather than ``str.isdigit``, which accepts superscripts and other non-ASCII
    digits, so
    ``"model²"`` would otherwise read as minted.

    One ambiguity is accepted: a legacy geometry whose group id is literally the file name
    ``"model3"`` is indistinguishable from a minted id and would be both prefixed and
    exclusion-matched. The alternative is a wider contract change (a marker field), and the
    collision needs a file named exactly ``model<digits>``.
    """
    return _RESOURCE_SCOPED_MODEL_ID_PATTERN.fullmatch(private_attribute_id) is not None


class GeometryComponentMergeContext(Flow360BaseModel):
    """Per-component identity context for the draft entity-info merge.

    Aligned by index with the merge's component list. ``excluded_model_indices`` are 0-based
    model indices — the position in ``ModelFileMetadata.models`` that conversion mints into each
    ``groupByFile`` group id as ``"model{N}"``. The merge keys entirely on that minted id, so no
    model file name is needed on the wire (this couples the merge to the conversion mint).
    """

    geometry_id: str = pd.Field(alias="geometryId")
    excluded_model_indices: list[int] = pd.Field(default_factory=list, alias="excludedModelIndices")


def _group_by_file_body_groups(entity_info: GeometryEntityInfo, purpose: str) -> list[GeometryBodyGroup]:
    """The ``groupByFile`` body groups -- one per model file -- or a logged failure.

    Conversion mints this grouping onto every geometry, so its absence means malformed input
    rather than a shape to degrade on. Checked explicitly so the message names the operation,
    instead of surfacing a bare ``ValueError`` from ``list.index``.
    """
    body_attribute_names = entity_info.body_attribute_names or []
    grouped_bodies = entity_info.grouped_bodies or []
    if "groupByFile" not in body_attribute_names:
        message = (
            f"Cannot {purpose}: its entity info has no 'groupByFile' body grouping "
            f"(has {body_attribute_names}); conversion mints one onto every geometry."
        )
        raise Flow360ValueError(message)
    index = body_attribute_names.index("groupByFile")
    return grouped_bodies[index]


def prefix_model_ids_with_geometry_id(
    entity_info: GeometryEntityInfo,
    geometry_id: str,
) -> GeometryEntityInfo:
    """Prefix each ``groupByFile`` body group's model id with its geometry id.

    Conversion mints each group's ``private_attribute_id`` as the resource-scoped ``"model{N}"``
    (N = the model's index into ``ModelFileMetadata.models``). Prefixing the geometry resource id
    makes that id globally unique within the merged draft: ``"model{N}"`` ->
    ``"{geometryId}/model{N}"``. The group ``name`` (the rootFile) stays untouched as the display
    label. Groups whose id isn't a bare ``"model{N}"`` (legacy geometry converted before the mint)
    are left untouched.

    Merged-draft consumers key exclusion toggles on this id, so two same-named model files in
    different component geometries stay distinguishable — the cross-resource relation is
    carried by the prefixed id, never by the user-given file name.
    """
    gbf_groups = _group_by_file_body_groups(entity_info, f"prefix model ids for geometry '{geometry_id}'")
    body_attribute_names = entity_info.body_attribute_names or []
    gbf_index = body_attribute_names.index("groupByFile")
    grouped_bodies = entity_info.grouped_bodies or []

    prefixed_groups: list[GeometryBodyGroup] = []
    changed = False
    for group in gbf_groups:
        model_id = group.private_attribute_id or ""
        # Prefix ONLY conversion's bare "model{N}" ids. Anything else is left as-is: a legacy id
        # (a file name, or a pre-mint "{resourceId}_{i}_{name}"), or an already-composed
        # "{geometryId}/model{N}" -- the latter also fails this check, so prefixing is idempotent.
        if _is_resource_scoped_model_id(model_id):
            # private_attribute_id is frozen; rebuild the group through deserialize (same pattern
            # as apply_user_settings_to_entity in the merge below).
            group_data = group.model_dump()
            group_data["private_attribute_id"] = f"{geometry_id}/{model_id}"
            prefixed_groups.append(group.__class__.deserialize(group_data))  # type: ignore[arg-type]
            changed = True
        else:
            prefixed_groups.append(group)
    if not changed:
        return entity_info

    result = entity_info.model_copy()
    result.grouped_bodies = [
        prefixed_groups if index == gbf_index else groups for index, groups in enumerate(grouped_bodies)
    ]
    return result


def _compute_face_id_to_bounding_box(per_face_group: list[Surface]) -> dict[str, BoundingBoxType]:
    """``faceId -> that face's own box``, from the per-face (``"faceId"``) grouping.

    Only that grouping is safe to read boxes from. A coarser grouping's box describes the
    aggregate its members had WHEN CONVERSION STAMPED IT, so it goes stale the moment pruning
    removes some of those members while keeping the group. A single-face group's box is intrinsic
    to the face and cannot.

    Raises rather than degrading: conversion stamps a box on every geometry face, so a group that
    is not exactly one boxed face means the entity info is malformed, and a box unioned from only
    part of the faces would silently understate the geometry's extent.
    """
    face_id_to_bbox: dict[str, BoundingBoxType] = {}
    for face in per_face_group:
        sub_components = face.private_attribute_sub_components or []
        if len(sub_components) != 1:
            raise Flow360ValueError(
                f"Malformed GeometryEntityInfo: 'faceId' group '{face.private_attribute_id}' holds "
                f"{len(sub_components)} faces; the per-face grouping must hold exactly one each."
            )
        box = face.private_attributes.bounding_box if face.private_attributes is not None else None
        if box is None:
            raise Flow360ValueError(
                f"Malformed GeometryEntityInfo: face '{sub_components[0]}' carries no "
                "private_attributes.bounding_box, so the geometry's extent cannot be recomputed."
            )
        face_id_to_bbox[sub_components[0]] = box
    return face_id_to_bbox


def _global_bounding_box_from_faces(
    face_id_to_bbox: dict[str, BoundingBoxType],
    face_ids: list[str] | None,
) -> BoundingBoxType | None:
    """Union of the boxes of exactly the faces ``face_ids`` names.

    Driven by ``face_ids`` -- the authoritative surviving-face list -- rather than by whichever
    groups survived pruning, so the result cannot disagree with the entity info it belongs to.

    ``None`` when no face survives: excluding every model is a legitimate projection, and it has
    no extent to report. Any other inconsistency raises.
    """
    if not face_ids:
        return None
    missing = [face_id for face_id in face_ids if face_id not in face_id_to_bbox]
    if missing:
        raise Flow360ValueError(
            f"Malformed GeometryEntityInfo: {len(missing)} surviving face(s) are absent from the "
            f"'faceId' grouping and have no box (first: '{missing[0]}')."
        )
    # expand() mutates in place -- start from the inf-sentinel, never from a face's own box
    result = BoundingBox.get_default_bounding_box()
    for face_id in face_ids:
        result.expand(face_id_to_bbox[face_id])
    return result


def project_entity_info_to_included_models(
    entity_info: GeometryEntityInfo,
    excluded_model_indices: set[int],
) -> GeometryEntityInfo:
    """Project a geometry component's entity info to its *included* models.

    Models are excluded by index, matched against the resource-scoped ``"model{N}"`` id that
    conversion mints onto each ``groupByFile`` body group. The excluded groups' bodies -- and
    those bodies' faces/edges (via ``bodies_face_edge_ids``) -- are removed from every parallel
    structure (``body_ids``/``grouped_bodies``, ``face_ids``/``grouped_faces``, ``edge_ids``/
    ``grouped_edges``, ``bodies_face_edge_ids``) so the result describes only included content.
    Used before merging components into a draft's ``simulation.json`` so that draft carries no
    entities for excluded models.

    Returns ``entity_info`` unchanged only when nothing is excluded. Anything else the caller
    asked for but cannot be delivered raises, after logging the reason: an excluded index that
    matches no minted ``"model{N}"`` group, a missing ``groupByFile`` grouping, or a face without
    its own box. A half-applied projection silently meshes a model the user excluded, which is
    worse than a rejected one.

    ``global_bounding_box`` is recomputed as the union of the REMAINING faces' per-surface boxes
    (conversion stamps ``private_attributes.bounding_box`` on every geometry face), so the
    projected component's box no longer covers excluded models. It is ``None`` when every model
    is excluded, since nothing is left to bound.
    """
    excluded_model_ids = {_resource_scoped_model_id(index) for index in (excluded_model_indices or set())}
    if not excluded_model_ids:
        return entity_info

    gbf_groups = _group_by_file_body_groups(entity_info, "project entity info to its included models")
    grouped_bodies = entity_info.grouped_bodies or []
    # Per-face boxes are intrinsic to a face, so this map is the same before and after pruning.
    # Built once and used both to correct shrunken groups and to recompute the global box.
    face_id_to_bbox = _compute_face_id_to_bounding_box(
        entity_info.grouped_faces[(entity_info.face_attribute_names or []).index("faceId")]
    )

    matched_ids = {group.private_attribute_id or "" for group in gbf_groups} & excluded_model_ids
    unmatched = sorted(excluded_model_ids - matched_ids)
    if unmatched:
        message = (
            f"Cannot exclude {unmatched} from this geometry: no 'groupByFile' body group carries "
            f"those minted ids (it has {sorted(g.private_attribute_id or '' for g in gbf_groups)}). "
            "The excluded model indices are out of range for this geometry."
        )
        raise Flow360ValueError(message)

    excluded_body_ids: set[str] = set()
    for group in gbf_groups:
        if (group.private_attribute_id or "") in excluded_model_ids:
            excluded_body_ids.update(group.private_attribute_sub_components or [])

    bodies_face_edge_ids = entity_info.bodies_face_edge_ids or {}
    excluded_face_ids: set[str] = set()
    excluded_edge_ids: set[str] = set()
    for body_id in excluded_body_ids:
        info = bodies_face_edge_ids[body_id]
        excluded_face_ids.update(info.face_ids or [])
        excluded_edge_ids.update(info.edge_ids or [])

    def _prune(
        groups_per_attribute: list[list[_GroupedEntityT]],
        excluded_sub_component_ids: set[str],
        face_id_to_bbox: dict[str, BoundingBoxType] | None = None,
    ) -> list[list[_GroupedEntityT]]:
        pruned: list[list[_GroupedEntityT]] = []
        for groups in groups_per_attribute or []:
            kept: list[_GroupedEntityT] = []
            for entity in groups or []:
                subs = entity.private_attribute_sub_components or []
                new_subs = [s for s in subs if s not in excluded_sub_component_ids]
                if subs and not new_subs:
                    continue  # every sub-component excluded -> drop the group
                if len(new_subs) == len(subs):
                    kept.append(entity)
                else:
                    pruned_entity = entity.model_copy()
                    pruned_entity.private_attribute_sub_components = new_subs
                    # A group's stamped box describes the members it HAD. Once it loses some the
                    # box is wrong for what remains, so recompute it from the survivors' own boxes
                    # rather than leave a stale one behind. Only Surface carries private_attributes;
                    # body and edge groups have no box to correct.
                    if (
                        face_id_to_bbox is not None
                        and isinstance(pruned_entity, Surface)
                        and pruned_entity.private_attributes is not None
                    ):
                        shrunken_box = BoundingBox.get_default_bounding_box()
                        for face_id in new_subs:
                            shrunken_box.expand(face_id_to_bbox[face_id])
                        # copy, or this would reach back into the caller's input
                        private_attributes = pruned_entity.private_attributes.model_copy()
                        private_attributes.bounding_box = shrunken_box
                        pruned_entity.private_attributes = private_attributes
                    kept.append(pruned_entity)
            pruned.append(kept)
        return pruned

    remaining_bodies_face_edge_ids = {
        body_id: info for body_id, info in bodies_face_edge_ids.items() if body_id not in excluded_body_ids
    } or None

    # Attribute assignment (not model_copy(update={...})) so a schema field rename fails loudly
    # here (pydantic rejects unknown attributes) instead of silently writing a ghost attribute
    # while the real field keeps stale data.
    filtered = entity_info.model_copy()
    filtered.body_ids = [b for b in (entity_info.body_ids or []) if b not in excluded_body_ids]
    filtered.grouped_bodies = _prune(grouped_bodies, excluded_body_ids)
    filtered.face_ids = [f for f in (entity_info.face_ids or []) if f not in excluded_face_ids]
    filtered.grouped_faces = _prune(entity_info.grouped_faces, excluded_face_ids, face_id_to_bbox)
    filtered.edge_ids = [e for e in (entity_info.edge_ids or []) if e not in excluded_edge_ids]
    filtered.grouped_edges = _prune(entity_info.grouped_edges, excluded_edge_ids)
    filtered.bodies_face_edge_ids = remaining_bodies_face_edge_ids
    # The resource-level box goes stale once models are dropped: recompute it from the
    # REMAINING faces' own per-surface boxes.
    filtered.global_bounding_box = _global_bounding_box_from_faces(face_id_to_bbox, filtered.face_ids)
    return filtered


def merge_geometry_entity_info(
    current_entity_info: GeometryEntityInfo,
    entity_info_components: list[GeometryEntityInfo],
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
    # Ignore the Geometry resource created from surface mesh that does not have any edge group
    edge_attr_sets = [set(ei.edge_attribute_names) for ei in entity_info_components if ei.edge_attribute_names]

    body_attr_intersection = set.intersection(*body_attr_sets) if body_attr_sets else set()
    face_attr_intersection = set.intersection(*face_attr_sets) if face_attr_sets else set()
    edge_attr_intersection = set.intersection(*edge_attr_sets) if edge_attr_sets else set()

    # Preserve order from current_entity_info, but include all attributes from intersection
    def ordered_intersection(reference_list: list[str], intersection_set: set[str]) -> list[str]:
        """Return all attributes from intersection_set, preserving order from reference_list where possible."""
        # First, add attributes that exist in reference_list (in order)
        result = [attr for attr in reference_list if attr in intersection_set]
        # Then, add remaining attributes from intersection_set that weren't in reference_list (sorted)
        remaining = sorted(intersection_set - set(result))
        return result + remaining

    result_body_attribute_names = ordered_intersection(current_entity_info.body_attribute_names, body_attr_intersection)
    result_face_attribute_names = ordered_intersection(current_entity_info.face_attribute_names, face_attr_intersection)
    result_edge_attribute_names = ordered_intersection(current_entity_info.edge_attribute_names, edge_attr_intersection)

    # 3. Update group tags: preserve from current if exists in intersection, otherwise use first
    def select_tag(current_tag: str | None, result_attrs: list[str], entity_type: str) -> str | None:
        if entity_type != "edge" and not result_attrs:
            raise ValueError(f"No attribute names available to select {entity_type} group tag.")
        logger.info("Preserving %s group tag: %s", entity_type, current_tag)
        return current_tag

    result_body_group_tag = select_tag(current_entity_info.body_group_tag, result_body_attribute_names, "body")
    result_face_group_tag = select_tag(current_entity_info.face_group_tag, result_face_attribute_names, "face")
    result_edge_group_tag = select_tag(current_entity_info.edge_group_tag, result_edge_attribute_names, "edge")

    # 4. Merge global bounding boxes from entity_info_components
    result_bounding_box = None
    for entity_info in entity_info_components:
        if entity_info.global_bounding_box is not None:
            if result_bounding_box is None:
                result_bounding_box = entity_info.global_bounding_box
            else:
                result_bounding_box = result_bounding_box.expand(entity_info.global_bounding_box)

    # 5. Get current user settings from body group and face
    def get_current_user_settings_map(
        entity_info: GeometryEntityInfo,
        entity_type: Literal["body", "face"],
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Extract user settings (like mesh_exterior, name) from entity_info.

        Args:
            entity_info: The GeometryEntityInfo to extract settings from
            entity_type: Either "body" or "face"

        Returns:
            A nested dictionary: {attribute_name: {entity_id: {setting_key: setting_value}}}
        """
        user_settings_map: dict[str, dict[str, dict[str, Any]]] = {}

        if entity_type == "body":
            attribute_names = entity_info.body_attribute_names
            grouped_entities: list[list[GeometryBodyGroup]] | list[list[Surface]] = entity_info.grouped_bodies
            settings_keys = ["mesh_exterior", "name"]
        elif entity_type == "face":
            attribute_names = entity_info.face_attribute_names
            grouped_entities = entity_info.grouped_faces
            settings_keys = ["name"]
        else:
            raise ValueError(f"Invalid entity_type: {entity_type}. Must be 'body' or 'face'.")

        for group_idx, group_name in enumerate(attribute_names):
            user_settings_map[group_name] = {}
            for entity in grouped_entities[group_idx]:
                entity_id = entity.private_attribute_id
                user_settings_map[group_name][entity_id] = {key: getattr(entity, key) for key in settings_keys}

        return user_settings_map

    current_body_user_settings_map = get_current_user_settings_map(current_entity_info, entity_type="body")
    current_face_user_settings_map = get_current_user_settings_map(current_entity_info, entity_type="face")

    # 6. Merge grouped entities from entity_info_components
    def apply_user_settings_to_entity(
        entity: GeometryBodyGroup | Surface | Edge,
        attr_name: str,
        user_settings_map: dict[str, dict[str, dict[str, Any]]] | None,
    ) -> GeometryBodyGroup | Surface | Edge:
        """
        Apply user settings to an entity if available in the user_settings_map.

        Args:
            entity: The entity to apply settings to
            attr_name: The attribute name (group name) for this entity
            user_settings_map: The user settings map from get_current_user_settings_map()

        Returns:
            The entity with user settings applied, or the original entity if no settings found
        """
        if user_settings_map is None:
            return entity

        entity_id = entity.private_attribute_id

        # Check if we have user settings for this entity
        if attr_name in user_settings_map and entity_id in user_settings_map[attr_name]:
            # Create a copy with updated user settings
            entity_data = entity.model_dump()
            entity_data.update(user_settings_map[attr_name][entity_id])
            return entity.__class__.deserialize(entity_data)  # type: ignore[return-value]

        return entity

    def merge_grouped_entities(
        entity_type: Literal["body", "face", "edge"],
        result_attr_names: list[str],
    ) -> list[list[Any]]:
        """Helper to merge grouped entities (bodies, faces, or edges) from entity_info_components"""

        # Determine which attributes to access based on entity type
        def get_attrs(entity_info: GeometryEntityInfo) -> list[str]:
            if entity_type == "body":
                return entity_info.body_attribute_names
            if entity_type == "face":
                return entity_info.face_attribute_names
            return entity_info.edge_attribute_names

        def get_groups(entity_info: GeometryEntityInfo) -> list[list[Any]]:
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
                    if entity_id in entity_map:
                        continue
                    # Apply user settings if available
                    user_settings_map = (
                        current_body_user_settings_map
                        if entity_type == "body"
                        else current_face_user_settings_map
                        if entity_type == "face"
                        else None
                    )
                    entity_map[entity_id] = apply_user_settings_to_entity(entity, attr_name, user_settings_map)

            # Convert map to list, maintaining a stable order (sorted by entity ID)
            result_grouped.append(sorted(entity_map.values(), key=lambda e: e.private_attribute_id or ""))

        return result_grouped

    result_grouped_bodies = merge_grouped_entities("body", result_body_attribute_names)
    result_grouped_faces = merge_grouped_entities("face", result_face_attribute_names)
    result_grouped_edges = merge_grouped_entities("edge", result_edge_attribute_names)

    # Use default_geometry_accuracy from first include_entity_info
    result_default_geometry_accuracy = entity_info_components[0].default_geometry_accuracy

    # Create the result GeometryEntityInfo
    return GeometryEntityInfo(  # type: ignore[call-arg]
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
