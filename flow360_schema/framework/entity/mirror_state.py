"""Mirror state and derived-entity orchestration shared by schema and client."""

from __future__ import annotations

import logging
from typing import cast

from flow360_schema.exceptions import Flow360ValueError
from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_registry import EntityRegistry, EntityRegistryView
from flow360_schema.framework.entity.entity_utils import is_exact_instance
from flow360_schema.models.asset_cache import MirrorStatus
from flow360_schema.models.entities.geometry_entities import GeometryBodyGroup, MirrorPlane
from flow360_schema.models.entities.surface_entities import (
    MirroredGeometryBodyGroup,
    MirroredSurface,
    Surface,
)

logger = logging.getLogger(__name__)

MIRROR_SUFFIX = "_<mirror>"


def _require_entity_id(entity: EntityBase) -> str:
    if entity.private_attribute_id is None:
        raise Flow360ValueError(
            f"Entity '{entity.name}' ({type(entity).__name__}) must have an id for mirror operations."
        )
    return entity.private_attribute_id


def _build_mirrored_geometry_groups(
    *,
    body_group_id_to_mirror_id: dict[str, str],
    body_groups_by_id: dict[str, GeometryBodyGroup],
    mirror_planes_by_id: dict[str, MirrorPlane],
) -> list[MirroredGeometryBodyGroup]:
    """Create mirrored geometry body groups for valid mirror actions."""
    mirrored_groups: list[MirroredGeometryBodyGroup] = []

    for body_group_id, mirror_plane_id in body_group_id_to_mirror_id.items():
        body_group = body_groups_by_id.get(body_group_id)
        if body_group is None:
            logger.warning(
                "Mirror action references unknown GeometryBodyGroup id '%s'; skipping.",
                body_group_id,
            )
            continue

        mirror_plane = mirror_planes_by_id.get(mirror_plane_id)
        if mirror_plane is None:
            logger.warning(
                "Mirror action references unknown MirrorPlane id '%s'; skipping.",
                mirror_plane_id,
            )
            continue

        mirrored_groups.append(
            MirroredGeometryBodyGroup(
                name=f"{body_group.name}{MIRROR_SUFFIX}",
                geometry_body_group_id=body_group_id,
                mirror_plane_id=mirror_plane_id,
                private_attribute_entity_type_name="MirroredGeometryBodyGroup",
            )
        )

    return mirrored_groups


def _build_mirrored_surfaces(
    *,
    body_group_id_to_mirror_id: dict[str, str],
    face_group_to_body_group: dict[str, str] | None,
    surfaces: list[Surface],
    mirror_planes_by_id: dict[str, MirrorPlane],
) -> list[MirroredSurface]:
    """Create mirrored surfaces for the requested body groups."""
    if not body_group_id_to_mirror_id or face_group_to_body_group is None:
        return []

    surfaces_by_name: dict[str, Surface] = {surface.name: surface for surface in surfaces}
    requested_body_group_ids = set(body_group_id_to_mirror_id)
    mirrored_surfaces: list[MirroredSurface] = []

    for surface_name, owning_body_group_id in face_group_to_body_group.items():
        if owning_body_group_id not in requested_body_group_ids:
            continue

        surface = surfaces_by_name.get(surface_name)
        if surface is None:
            logger.warning(
                "Surface '%s' referenced in GeometryEntityInfo was not found in draft registry; "
                "skipping mirroring for this surface.",
                surface_name,
            )
            continue

        mirror_plane_id = body_group_id_to_mirror_id.get(owning_body_group_id)
        if mirror_plane_id is None:
            logger.warning(
                "Body group '%s' was requested for mirroring but has no mirror-plane mapping; "
                "skipping mirroring for surface '%s'.",
                owning_body_group_id,
                surface_name,
            )
            continue

        mirror_plane = mirror_planes_by_id.get(mirror_plane_id)
        if mirror_plane is None:
            logger.warning(
                "Mirror action references unknown MirrorPlane id '%s' for body group '%s'; "
                "skipping mirroring for surface '%s'.",
                mirror_plane_id,
                owning_body_group_id,
                surface_name,
            )
            continue

        mirrored_surface = MirroredSurface(
            name=f"{surface.name}{MIRROR_SUFFIX}",
            surface_id=_require_entity_id(surface),
            mirror_plane_id=mirror_plane_id,
            private_attribute_entity_type_name="MirroredSurface",
            private_attribute_full_name=None,
        )
        mirrored_surface._geometry_body_group_id = owning_body_group_id
        mirrored_surfaces.append(mirrored_surface)

    return mirrored_surfaces


def _derive_mirrored_entities_from_actions(
    *,
    body_group_id_to_mirror_id: dict[str, str],
    face_group_to_body_group: dict[str, str] | None,
    entity_registry: EntityRegistry,
    mirror_planes: list[MirrorPlane],
) -> tuple[list[MirroredGeometryBodyGroup], list[MirroredSurface]]:
    """Derive mirrored entities from the requested body-group mirror actions."""
    if not body_group_id_to_mirror_id:
        return [], []

    body_groups = cast(list[GeometryBodyGroup], entity_registry.view(GeometryBodyGroup)._entities)
    surfaces = cast(list[Surface], entity_registry.view(Surface)._entities)

    body_groups_by_id = {_require_entity_id(body_group): body_group for body_group in body_groups}
    mirror_planes_by_id = {mirror_plane.private_attribute_id: mirror_plane for mirror_plane in mirror_planes}

    mirrored_geometry_groups = _build_mirrored_geometry_groups(
        body_group_id_to_mirror_id=body_group_id_to_mirror_id,
        body_groups_by_id=body_groups_by_id,
        mirror_planes_by_id=mirror_planes_by_id,
    )
    mirrored_surfaces = _build_mirrored_surfaces(
        body_group_id_to_mirror_id=body_group_id_to_mirror_id,
        face_group_to_body_group=face_group_to_body_group,
        surfaces=surfaces,
        mirror_planes_by_id=mirror_planes_by_id,
    )
    return mirrored_geometry_groups, mirrored_surfaces


def _extract_body_group_id_to_mirror_id_from_status(
    *,
    mirror_status: MirrorStatus | None,
    valid_body_group_ids: set[str] | None,
) -> dict[str, str]:
    """Extract body-group mirror actions from persisted mirror status."""
    if mirror_status is None:
        logger.debug("Mirror status not provided; no mirroring actions to restore.")
        return {}

    mirror_planes_by_id = {
        mirror_plane.private_attribute_id: mirror_plane for mirror_plane in mirror_status.mirror_planes
    }

    body_group_id_to_mirror_id: dict[str, str] = {}
    for mirrored_group in mirror_status.mirrored_geometry_body_groups:
        body_group_id = mirrored_group.geometry_body_group_id
        mirror_plane_id = mirrored_group.mirror_plane_id

        if valid_body_group_ids is not None and body_group_id not in valid_body_group_ids:
            logger.debug(
                "Ignoring mirroring of GeometryBodyGroup (ID:'%s') because it no longer exists.",
                body_group_id,
            )
            continue

        if mirror_plane_id not in mirror_planes_by_id:
            logger.debug(
                "Ignoring mirroring of GeometryBodyGroup (ID:'%s') because the referenced"
                " mirror plane (ID:'%s') no longer exists.",
                body_group_id,
                mirror_plane_id,
            )
            continue

        body_group_id_to_mirror_id[body_group_id] = mirror_plane_id

    return body_group_id_to_mirror_id


class MirrorState:
    """Core mirror state and derived-entity orchestration."""

    __slots__ = (
        "_mirror_status",
        "_body_group_id_to_mirror_id",
        "_face_group_to_body_group",
        "_entity_registry",
    )

    def __init__(
        self,
        *,
        face_group_to_body_group: dict[str, str] | None,
        entity_registry: EntityRegistry,
    ) -> None:
        self._body_group_id_to_mirror_id: dict[str, str] = {}
        self._face_group_to_body_group = face_group_to_body_group
        self._entity_registry = entity_registry
        self._mirror_status = MirrorStatus(
            mirror_planes=[],
            mirrored_geometry_body_groups=[],
            mirrored_surfaces=[],
        )

    @property
    def mirror_planes(self) -> EntityRegistryView:
        """Return all available mirror planes."""
        return self._entity_registry.view(MirrorPlane)

    def create_mirror_of(
        self,
        *,
        entities: list[GeometryBodyGroup] | GeometryBodyGroup,
        mirror_plane: MirrorPlane,
    ) -> tuple[list[MirroredGeometryBodyGroup], list[MirroredSurface]]:
        """Create mirrored entities for one or more geometry body groups."""
        normalized_entities = self._validate_and_normalize_create_inputs(
            entities=entities,
            mirror_plane=mirror_plane,
        )
        self._prepare_for_mirror_update(entities=normalized_entities)
        self._ensure_mirror_plane_registered(mirror_plane=mirror_plane)
        return self._apply_actions_and_generate_entities(
            entities=normalized_entities,
            mirror_plane=mirror_plane,
        )

    def remove_mirror_of(self, *, entities: list[GeometryBodyGroup] | GeometryBodyGroup) -> None:
        """Remove the mirror of the given entities."""
        if isinstance(entities, GeometryBodyGroup):
            normalized_entities = [entities]
        elif isinstance(entities, list):
            normalized_entities = entities
        else:
            raise Flow360ValueError(
                f"`entities` accepts a single entity or a list of entities. Received type: {type(entities).__name__}."
            )

        for entity in normalized_entities:
            if not is_exact_instance(entity, GeometryBodyGroup):
                raise Flow360ValueError(
                    "Only GeometryBodyGroup entities are supported by `remove_mirror_of()`. "
                    f"Received: {type(entity).__name__}."
                )

        for body_group in normalized_entities:
            body_group_id = _require_entity_id(body_group)
            self._body_group_id_to_mirror_id.pop(body_group_id, None)
            mirrored_groups_to_remove = [
                mirrored_group
                for mirrored_group in list(self._mirror_status.mirrored_geometry_body_groups)
                if mirrored_group.geometry_body_group_id == body_group_id
            ]
            for mirrored_group in mirrored_groups_to_remove:
                self._remove(mirrored_group)

    @property
    def _mirror_planes(self) -> list[MirrorPlane]:
        """Return the list of mirror planes."""
        return self._mirror_status.mirror_planes

    def _validate_and_normalize_create_inputs(
        self,
        *,
        entities: list[GeometryBodyGroup] | GeometryBodyGroup,
        mirror_plane: MirrorPlane,
    ) -> list[GeometryBodyGroup]:
        """Validate inputs for create_mirror_of and normalize entities to a list."""
        if isinstance(entities, GeometryBodyGroup):
            normalized_entities = [entities]
        elif isinstance(entities, list):
            normalized_entities = entities
        else:
            raise Flow360ValueError(
                f"`entities` accepts a single entity or a list of entities. Received type: {type(entities).__name__}."
            )

        for entity in normalized_entities:
            if not is_exact_instance(entity, GeometryBodyGroup):
                raise Flow360ValueError(
                    "Only GeometryBodyGroup entities are supported by `create()` currently. "
                    f"Received: {type(entity).__name__}."
                )
            _require_entity_id(entity)

        if not is_exact_instance(mirror_plane, MirrorPlane):
            raise Flow360ValueError(
                f"`mirror_plane` must be a MirrorPlane entity. Instead received: {type(mirror_plane).__name__}."
            )

        if self._face_group_to_body_group is None:
            raise Flow360ValueError(
                "Mirroring is not available because the surface-to-body-group mapping could not be derived. "
                "This typically happens when face groupings span across multiple body groups."
            )

        return normalized_entities

    def _prepare_for_mirror_update(self, *, entities: list[GeometryBodyGroup]) -> None:
        """Warn on overwrites and remove previously-derived mirrored entities for these body groups."""
        body_group_ids_to_update: set[str] = set()
        for body_group in entities:
            body_group_id = _require_entity_id(body_group)
            body_group_ids_to_update.add(body_group_id)
            if body_group_id in self._body_group_id_to_mirror_id:
                logger.warning(
                    "GeometryBodyGroup `%s` was already mirrored; resetting to the latest mirror plane request.",
                    body_group.name,
                )

        existing_mirrored_groups = [
            mirrored_group
            for mirrored_group in list(self._mirror_status.mirrored_geometry_body_groups)
            if mirrored_group.geometry_body_group_id in body_group_ids_to_update
        ]
        for mirrored_group in existing_mirrored_groups:
            self._remove(mirrored_group)

    def _ensure_mirror_plane_registered(self, *, mirror_plane: MirrorPlane) -> None:
        """Validate mirror plane name uniqueness and register the plane if needed."""
        for existing_plane in self._mirror_planes:
            if (
                existing_plane.name == mirror_plane.name
                and existing_plane.private_attribute_id != mirror_plane.private_attribute_id
            ):
                raise Flow360ValueError(f"Mirror plane name '{mirror_plane.name}' already exists in the draft.")

        if any(
            existing_plane.private_attribute_id == mirror_plane.private_attribute_id
            for existing_plane in self._mirror_planes
        ):
            return
        self._add(mirror_plane)

    def _apply_actions_and_generate_entities(
        self,
        *,
        entities: list[GeometryBodyGroup],
        mirror_plane: MirrorPlane,
    ) -> tuple[list[MirroredGeometryBodyGroup], list[MirroredSurface]]:
        """Update actions for the given entities and generate/register derived mirrored entities."""
        mirror_plane_id = mirror_plane.private_attribute_id
        body_group_id_to_mirror_id_update: dict[str, str] = {}
        for body_group in entities:
            body_group_id = _require_entity_id(body_group)
            body_group_id_to_mirror_id_update[body_group_id] = mirror_plane_id
            self._body_group_id_to_mirror_id[body_group_id] = mirror_plane_id

        mirrored_geometry_groups, mirrored_surfaces = _derive_mirrored_entities_from_actions(
            body_group_id_to_mirror_id=body_group_id_to_mirror_id_update,
            face_group_to_body_group=self._face_group_to_body_group,
            entity_registry=self._entity_registry,
            mirror_planes=self._mirror_status.mirror_planes,
        )

        for mirrored_geometry_group in mirrored_geometry_groups:
            self._add(mirrored_geometry_group)
        for mirrored_surface in mirrored_surfaces:
            self._add(mirrored_surface)

        return mirrored_geometry_groups, mirrored_surfaces

    def _add(self, entity: MirrorPlane | MirroredGeometryBodyGroup | MirroredSurface) -> None:
        """Add an entity to the mirror status and registry."""
        if self._entity_registry.contains(entity):
            return

        if type(entity) is MirrorPlane:
            self._mirror_status.mirror_planes.append(entity)
            self._entity_registry.register(entity)
            return
        if type(entity) is MirroredGeometryBodyGroup:
            self._mirror_status.mirrored_geometry_body_groups.append(entity)
            self._entity_registry.register(entity)
            return
        if type(entity) is MirroredSurface:
            self._mirror_status.mirrored_surfaces.append(entity)
            self._entity_registry.register(entity)
            return
        raise Flow360ValueError(f"[Internal] Unsupported entity type: {type(entity).__name__}.")

    def _remove(self, entity: MirroredGeometryBodyGroup) -> None:
        """Remove a mirrored geometry body group and its mirrored surfaces."""
        if entity in self._mirror_status.mirrored_geometry_body_groups:
            self._mirror_status.mirrored_geometry_body_groups.remove(entity)
        self._entity_registry.remove(entity)

        body_group_id = entity.geometry_body_group_id
        mirrored_surfaces_to_remove = [
            mirrored_surface
            for mirrored_surface in list(self._mirror_status.mirrored_surfaces)
            if getattr(mirrored_surface, "_geometry_body_group_id", None) == body_group_id
        ]
        for mirrored_surface in mirrored_surfaces_to_remove:
            if mirrored_surface in self._mirror_status.mirrored_surfaces:
                self._mirror_status.mirrored_surfaces.remove(mirrored_surface)
            self._entity_registry.remove(mirrored_surface)

    @staticmethod
    def _generate_mirror_status(
        entity_registry: EntityRegistry,
        body_group_id_to_mirror_id: dict[str, str],
        face_group_to_body_group: dict[str, str] | None,
        mirror_planes: list[MirrorPlane],
    ) -> MirrorStatus:
        """Build a serializable status snapshot."""
        existing_body_group_ids = {
            _require_entity_id(entity)
            for entity in entity_registry.find_by_type(GeometryBodyGroup)
            if is_exact_instance(entity, GeometryBodyGroup)
        }

        filtered_actions: dict[str, str] = {}
        for body_group_id, mirror_plane_id in body_group_id_to_mirror_id.items():
            if body_group_id not in existing_body_group_ids:
                logger.warning(
                    "GeometryBodyGroup '%s' assigned to mirror plane '%s' is not in the draft registry; "
                    "skipping this mirror action.",
                    body_group_id,
                    mirror_plane_id,
                )
                continue
            filtered_actions[body_group_id] = mirror_plane_id

        if not filtered_actions:
            return MirrorStatus(
                mirror_planes=[],
                mirrored_geometry_body_groups=[],
                mirrored_surfaces=[],
            )

        mirrored_geometry_groups, mirrored_surfaces = _derive_mirrored_entities_from_actions(
            body_group_id_to_mirror_id=filtered_actions,
            face_group_to_body_group=face_group_to_body_group,
            entity_registry=entity_registry,
            mirror_planes=mirror_planes,
        )

        mirror_planes_by_id = {mirror_plane.private_attribute_id: mirror_plane for mirror_plane in mirror_planes}
        used_plane_ids = {
            mirror_plane_id for mirror_plane_id in filtered_actions.values() if mirror_plane_id in mirror_planes_by_id
        }
        mirror_planes_for_status = [
            mirror_plane for mirror_plane in mirror_planes if mirror_plane.private_attribute_id in used_plane_ids
        ]

        return MirrorStatus(
            mirror_planes=mirror_planes_for_status,
            mirrored_geometry_body_groups=mirrored_geometry_groups,
            mirrored_surfaces=mirrored_surfaces,
        )

    @classmethod
    def _from_status(
        cls,
        *,
        status: MirrorStatus | None,
        face_group_to_body_group: dict[str, str] | None,
        entity_registry: EntityRegistry,
    ) -> MirrorState:
        """Restore mirror state from a status snapshot."""
        state = cls(
            face_group_to_body_group=face_group_to_body_group,
            entity_registry=entity_registry,
        )

        body_groups = cast(list[GeometryBodyGroup], entity_registry.view(GeometryBodyGroup)._entities)
        valid_body_group_ids = {_require_entity_id(body_group) for body_group in body_groups}

        state._body_group_id_to_mirror_id = _extract_body_group_id_to_mirror_id_from_status(
            mirror_status=status,
            valid_body_group_ids=valid_body_group_ids,
        )

        state._mirror_status.mirror_planes = status.mirror_planes if status is not None else []
        state._mirror_status = cls._generate_mirror_status(
            entity_registry=entity_registry,
            body_group_id_to_mirror_id=state._body_group_id_to_mirror_id,
            face_group_to_body_group=state._face_group_to_body_group,
            mirror_planes=state._mirror_status.mirror_planes,
        )

        for mirrored_group in list(state._mirror_status.mirrored_geometry_body_groups):
            state._entity_registry.register(mirrored_group)
        for mirrored_surface in list(state._mirror_status.mirrored_surfaces):
            state._entity_registry.register(mirrored_surface)
        for mirror_plane in list(state._mirror_status.mirror_planes):
            state._entity_registry.register(mirror_plane)

        return state
