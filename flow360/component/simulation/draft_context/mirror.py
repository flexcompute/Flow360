"""Mirror plane, mirrored entities and helpers."""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pydantic as pd

from flow360.component.simulation.entity_operation import (
    _transform_direction,
    _transform_point,
)
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.entity_registry import (
    EntityRegistry,
    EntityRegistryView,
)
from flow360.component.simulation.framework.entity_utils import generate_uuid
from flow360.component.simulation.primitives import (
    GeometryBodyGroup,
    MirroredGeometryBodyGroup,
    MirroredSurface,
    Surface,
)
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.utils import is_exact_instance
from flow360.component.types import Axis
from flow360.exceptions import Flow360RuntimeError
from flow360.log import log


class MirrorPlane(EntityBase):
    """
    :class:`MirrorPlane` class for defining a mirror plane for mirroring entities.

     Example
     -------

     >>> fl.MirrorPlane(
     ...     name="MirrorPlane",
     ...     normal=(0, 1, 0),
     ...     center=(0, 0, 0)*fl.u.m
     ... )
    """

    name: str = pd.Field()
    normal: Axis = pd.Field(description="Normal direction of the plane.")
    # pylint: disable=no-member
    center: LengthType.Point = pd.Field(description="Center point of the plane.")

    private_attribute_entity_type_name: Literal["MirrorPlane"] = pd.Field(
        "MirrorPlane", frozen=True
    )
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    def _apply_transformation(self, matrix: np.ndarray) -> "MirrorPlane":
        """Apply 3x4 transformation matrix, returning new transformed instance."""
        # Transform the center point
        center_array = np.asarray(self.center.value)
        new_center_array = _transform_point(center_array, matrix)
        new_center = type(self.center)(new_center_array, self.center.units)

        # Transform and normalize the normal direction
        normal_array = np.asarray(self.normal)
        transformed_normal = _transform_direction(normal_array, matrix)
        new_normal = tuple(transformed_normal / np.linalg.norm(transformed_normal))

        return self.model_copy(update={"center": new_center, "normal": new_normal})


# region -----------------------------Internal Model Below-------------------------------------
class MirrorStatus(Flow360BaseModel):
    """
    Internal data model for storing the mirror status.
    """

    # Note: We can do similar thing as entityList to support mirroring with EntitySelectors.
    mirror_planes: List[MirrorPlane] = pd.Field(description="List of mirror planes to mirror.")
    mirrored_geometry_body_groups: List[MirroredGeometryBodyGroup] = pd.Field(
        description="List of mirrored geometry body groups."
    )
    mirrored_surfaces: List[MirroredSurface] = pd.Field(description="List of mirrored surfaces.")

    @pd.model_validator(mode="after")
    def _validate_unique_mirror_plane_names(self):
        """Validate that all mirror plane names are unique."""
        seen_names = set()
        for plane in self.mirror_planes:
            if plane.name in seen_names:
                raise ValueError(
                    f"Duplicate mirror plane name '{plane.name}' found in mirror status."
                )
            seen_names.add(plane.name)
        return self

    def is_empty(self) -> bool:
        """Check if the mirror status is empty."""
        return (
            not self.mirror_planes
            and not self.mirrored_geometry_body_groups
            and not self.mirrored_surfaces
        )


# endregion -------------------------------------------------------------------------------------

MIRROR_SUFFIX = "_<mirror>"

# region -----------------------------Internal Functions Below-------------------------------------


def _build_mirrored_geometry_groups(
    *,
    body_group_id_to_mirror_id: Dict[str, str],
    body_groups_by_id: Dict[str, GeometryBodyGroup],
    mirror_planes_by_id: Dict[str, MirrorPlane],
) -> List[MirroredGeometryBodyGroup]:
    """Create mirrored geometry body groups for valid mirror actions."""

    mirrored_groups: List[MirroredGeometryBodyGroup] = []

    for body_group_id, mirror_plane_id in body_group_id_to_mirror_id.items():
        body_group = body_groups_by_id.get(body_group_id)
        if body_group is None:
            log.warning(
                "Mirror action references unknown GeometryBodyGroup id '%s'; skipping.",
                body_group_id,
            )
            continue

        mirror_plane = mirror_planes_by_id.get(mirror_plane_id)
        if mirror_plane is None:
            log.warning(
                "Mirror action references unknown MirrorPlane id '%s'; skipping.",
                mirror_plane_id,
            )
            continue

        mirrored_groups.append(
            MirroredGeometryBodyGroup(
                name=f"{body_group.name}{MIRROR_SUFFIX}",
                geometry_body_group_id=body_group_id,
                mirror_plane_id=mirror_plane_id,
            )
        )

    return mirrored_groups


def _build_mirrored_surfaces(
    *,
    body_group_id_to_mirror_id: Dict[str, str],
    face_group_to_body_group: Optional[Dict[str, str]],
    surfaces: List[Surface],
    mirror_planes_by_id: Dict[str, MirrorPlane],
) -> List[MirroredSurface]:
    """Create mirrored surfaces for the requested body groups."""

    if not body_group_id_to_mirror_id or face_group_to_body_group is None:
        return []

    surfaces_by_name: Dict[str, Surface] = {surface.name: surface for surface in surfaces}
    requested_body_group_ids = set(body_group_id_to_mirror_id.keys())
    mirrored_surfaces: List[MirroredSurface] = []

    for surface_name, owning_body_group_id in face_group_to_body_group.items():
        if owning_body_group_id not in requested_body_group_ids:
            continue

        surface = surfaces_by_name.get(surface_name)
        if surface is None:
            log.warning(
                "Surface '%s' referenced in GeometryEntityInfo was not found in draft registry; "
                "skipping mirroring for this surface.",
                surface_name,
            )
            continue

        mirror_plane_id = body_group_id_to_mirror_id.get(owning_body_group_id)
        mirror_plane = mirror_planes_by_id.get(mirror_plane_id)
        if mirror_plane is None:
            log.warning(
                "Mirror action references unknown MirrorPlane id '%s' for body group '%s'; "
                "skipping mirroring for surface '%s'.",
                mirror_plane_id,
                owning_body_group_id,
                surface_name,
            )
            continue

        mirrored_surface = MirroredSurface(
            name=f"{surface.name}{MIRROR_SUFFIX}",
            surface_id=surface.private_attribute_id,
            mirror_plane_id=mirror_plane_id,
        )
        # Draft-only bookkeeping: record which body group generated this mirrored surface.
        mirrored_surface._geometry_body_group_id = (  # pylint: disable=protected-access
            owning_body_group_id
        )
        mirrored_surfaces.append(mirrored_surface)

    return mirrored_surfaces


def _derive_mirrored_entities_from_actions(
    *,
    body_group_id_to_mirror_id: Dict[str, str],
    face_group_to_body_group: Optional[Dict[str, str]],
    entity_registry: EntityRegistry,
    mirror_planes: List[MirrorPlane],
) -> Tuple[List[MirroredGeometryBodyGroup], List[MirroredSurface]]:
    """
    Derive mirrored entities (MirroredGeometryBodyGroup + MirroredSurface)
    based on the given ``body_group_id_to_mirror_id`` mapping.

    The ``body_group_id_to_mirror_id`` schema is::

        {geometry_body_group_id: mirror_plane_id}

    Parameters
    ----------
    body_group_id_to_mirror_id : Dict[str, str]
        Mapping from geometry body group ID to mirror plane ID.
    face_group_to_body_group : Optional[Dict[str, str]]
        Mapping from surface name to owning body group ID. If None, no surfaces will be mirrored.
    entity_registry : EntityRegistry
        Entity registry containing body groups and surfaces.
    mirror_planes : List[MirrorPlane]
        List of all mirror planes.

    Returns
    -------
    Tuple[List[MirroredGeometryBodyGroup], List[MirroredSurface]]
        Mirrored body groups and surfaces.

    This helper is intended to be reusable both from within the draft context
    (for incremental updates) and before submission (for generating the full list
    of mirrored entities from the stored mirror status).
    """

    if not body_group_id_to_mirror_id:
        return [], []

    # Extract body groups and surfaces from the entity registry.
    body_groups = entity_registry.view(  # pylint: disable=protected-access
        GeometryBodyGroup
    )._entities
    surfaces = entity_registry.view(Surface)._entities  # pylint: disable=protected-access

    # Lookup tables for body groups and mirror planes.
    body_groups_by_id: Dict[str, GeometryBodyGroup] = {
        body_group.private_attribute_id: body_group for body_group in body_groups
    }
    mirror_planes_by_id: Dict[str, MirrorPlane] = {
        plane.private_attribute_id: plane for plane in mirror_planes
    }

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
    mirror_status: Optional[MirrorStatus],
    valid_body_group_ids: Optional[set[str]],
) -> Dict[str, str]:
    """
    Deserialize mirror actions from a :class:`MirrorStatus` instance.

    Parameters
    ----------
    mirror_status : MirrorStatus
        The mirror status to deserialize.
    valid_body_group_ids : Optional[set[str]]
        Set of valid body group IDs. If provided, any mirror actions referencing
        body groups not in this set will be skipped.

    Returns
    -------
    Dict[str, str]
        - ``body_group_id_to_mirror_id``: mapping from geometry body group ID to mirror plane ID.
    """

    if mirror_status is None:
        # No mirror feature used in the asset.
        log.debug("Mirror status not provided; no mirroring actions to restore.")
        return {}

    mirror_planes_by_id: Dict[str, MirrorPlane] = {
        plane.private_attribute_id: plane for plane in mirror_status.mirror_planes
    }

    body_group_id_to_mirror_id: Dict[str, str] = {}
    for mirrored_group in mirror_status.mirrored_geometry_body_groups:
        body_group_id = mirrored_group.geometry_body_group_id
        mirror_plane_id = mirrored_group.mirror_plane_id

        if valid_body_group_ids is not None and body_group_id not in valid_body_group_ids:
            # Skip body groups that no longer exist.
            log.debug(
                "Ignoring mirroring of GeometryBodyGroup (ID:'%s') because it no longer exists.",
                body_group_id,
            )
            continue

        if mirror_plane_id not in mirror_planes_by_id:
            # Skip if the referenced mirror plane is no longer present.
            log.debug(
                "Ignoring mirroring of GeometryBodyGroup (ID:'%s') because the referenced"
                " mirror plane (ID:'%s') no longer exists.",
                body_group_id,
                mirror_plane_id,
            )
            continue

        body_group_id_to_mirror_id[body_group_id] = mirror_plane_id

    return body_group_id_to_mirror_id


# endregion -------------------------------------------------------------------------------------


class MirrorManager:
    """Encapsulates mirror plane registry and entity mirroring operations."""

    __slots__ = (
        "_mirror_status",  # MirrorManager owns the single mirror status instance.
        "_body_group_id_to_mirror_id",
        "_face_group_to_body_group",
        "_entity_registry",  # A link to the full picture.
    )
    _mirror_status: MirrorStatus
    _body_group_id_to_mirror_id: Dict[str, str]
    _face_group_to_body_group: Optional[Dict[str, str]]
    _entity_registry: EntityRegistry

    def __init__(
        self,
        *,
        face_group_to_body_group: Optional[Dict[str, str]],
        entity_registry: EntityRegistry,
    ) -> None:
        self._body_group_id_to_mirror_id = {}
        self._face_group_to_body_group = face_group_to_body_group
        self._entity_registry = entity_registry
        self._mirror_status = MirrorStatus(
            mirror_planes=[], mirrored_geometry_body_groups=[], mirrored_surfaces=[]
        )

    # region Public API -------------------------------------------------
    @property
    def mirror_planes(self) -> EntityRegistryView:
        """
        Return all the available mirror planes.
        # TODO: Some docstrings?
        """
        return self._entity_registry.view(MirrorPlane)

    def create_mirror_of(  # pylint: disable=too-many-branches
        self,
        *,
        entities: Union[List[GeometryBodyGroup], GeometryBodyGroup],
        mirror_plane: MirrorPlane,
    ) -> tuple[List[MirroredGeometryBodyGroup], List[MirroredSurface]]:
        """
        Create mirrored GeometryBodyGroup (and its associated surfaces) for the given `MirrorPlane`.
        New entities will have "_<mirror>" in the name as suffix.

        Parameters
        ----------
        entities : Union[List[GeometryBodyGroup], GeometryBodyGroup]
            One or more geometry body groups to mirror.
        mirror_plane : MirrorPlane
            The mirror plane to use for mirroring.

        Returns
        -------
        tuple[List[MirroredGeometryBodyGroup], List[MirroredSurface]]
            Mirrored geometry body groups and surfaces.
        """
        # 1. [Validation] Ensure `entities` are GeometryBodyGroup entities.
        normalized_entities: List[GeometryBodyGroup]
        if isinstance(entities, GeometryBodyGroup):
            normalized_entities = [entities]
        elif isinstance(entities, list):
            normalized_entities = entities
        else:
            raise Flow360RuntimeError(
                f"`entities` accepts a single entity or a list of entities. Received type: {type(entities).__name__}."
            )

        for entity in normalized_entities:
            if not is_exact_instance(entity, GeometryBodyGroup):
                raise Flow360RuntimeError(
                    "Only GeometryBodyGroup entities are supported by `create()` currently. "
                    f"Received: {type(entity).__name__}."
                )

        # 2. [Validation] Ensure `mirror_plane` is a `MirrorPlane` entity.
        if not is_exact_instance(mirror_plane, MirrorPlane):
            raise Flow360RuntimeError(
                f"`mirror_plane` must be a MirrorPlane entity. Instead received: {type(mirror_plane).__name__}."
            )

        # 3. [Validation] Ensure the surface-to-body-group mapping is available.
        if self._face_group_to_body_group is None:
            raise Flow360RuntimeError(
                "Mirroring is not available because the surface-to-body-group mapping could not be derived. "
                "This typically happens when face groupings span across multiple body groups."
            )

        # 4. [Restriction] Each GeometryBodyGroup entity can only be mirrored once.
        #                  If a duplicate request is made, reset to the new one with a warning.
        for body_group in normalized_entities:
            body_group_id = body_group.private_attribute_id
            if body_group_id in self._body_group_id_to_mirror_id:
                log.warning(
                    "GeometryBodyGroup `%s` was already mirrored; resetting to the latest mirror plane request.",
                    body_group.name,
                )

        # If we're updating an existing mirror assignment, remove any previously-derived mirrored entities
        # so that the registry/status always reflect the latest mirror status.
        body_group_ids_to_update = {bg.private_attribute_id for bg in normalized_entities}
        existing_mirrored_groups = [
            mirrored_group
            for mirrored_group in list(self._mirror_status.mirrored_geometry_body_groups)
            if mirrored_group.geometry_body_group_id in body_group_ids_to_update
        ]
        for mirrored_group in existing_mirrored_groups:
            self._remove(mirrored_group)

        # 5. [Validation] Ensure mirror plane has a unique name.
        #
        # Note: We do NOT rely on EntityRegistry.contains(mirror_plane) here because
        # entity equality for registry membership can treat different plane objects
        # with the same name as "contained". We enforce uniqueness by name + id.
        for existing_plane in self._mirror_planes:
            if (
                existing_plane.name == mirror_plane.name
                and existing_plane.private_attribute_id != mirror_plane.private_attribute_id
            ):
                raise Flow360RuntimeError(
                    f"Mirror plane name '{mirror_plane.name}' already exists in the draft."
                )

        # Register the plane if we don't already have the same plane id.
        if not any(
            plane.private_attribute_id == mirror_plane.private_attribute_id
            for plane in self._mirror_planes
        ):
            self._add(mirror_plane)

        # 6. Create/Update the self._body_group_id_to_mirror_id
        body_group_id_to_mirror_id_update: Dict[str, str] = {}
        for body_group in normalized_entities:
            body_group_id = body_group.private_attribute_id
            body_group_id_to_mirror_id_update[body_group_id] = mirror_plane.private_attribute_id
            self._body_group_id_to_mirror_id[body_group_id] = mirror_plane.private_attribute_id

        # 7. Derive the generated mirrored entities (MirroredGeometryBodyGroup + MirroredSurface)
        #    and return to user as tokens of use.
        mirrored_geometry_groups, mirrored_surfaces = _derive_mirrored_entities_from_actions(
            body_group_id_to_mirror_id=body_group_id_to_mirror_id_update,
            face_group_to_body_group=self._face_group_to_body_group,
            entity_registry=self._entity_registry,
            mirror_planes=self._mirror_status.mirror_planes,
        )

        # * (comapred to Box where the creation of it does not use any draft API so no need
        # *  to register/update/reflect it to the draft context).
        for mirrored_geometry_group in mirrored_geometry_groups:
            self._add(mirrored_geometry_group)
        for mirrored_surface in mirrored_surfaces:
            self._add(mirrored_surface)

        return mirrored_geometry_groups, mirrored_surfaces

    def remove_mirror_of(
        self, *, entities: Union[List[GeometryBodyGroup], GeometryBodyGroup]
    ) -> None:
        """
        Remove the mirror of the given entities.

        Parameters
        ----------
        entities : Union[List[GeometryBodyGroup], GeometryBodyGroup]
            One or more geometry body groups to remove mirroring from.
        """
        # 1. [Validation] Ensure `entities` are GeometryBodyGroup entities.
        normalized_entities: List[GeometryBodyGroup]
        if isinstance(entities, GeometryBodyGroup):
            normalized_entities = [entities]
        elif isinstance(entities, list):
            normalized_entities = entities
        else:
            raise Flow360RuntimeError(
                f"`entities` accepts a single entity or a list of entities. Received type: {type(entities).__name__}."
            )

        for entity in normalized_entities:
            if not is_exact_instance(entity, GeometryBodyGroup):
                raise Flow360RuntimeError(
                    "Only GeometryBodyGroup entities are supported by `remove_mirror_of()`. "
                    f"Received: {type(entity).__name__}."
                )

        # 2. Remove mirror assignments for the given entities.
        for body_group in normalized_entities:
            body_group_id = body_group.private_attribute_id
            self._body_group_id_to_mirror_id.pop(body_group_id, None)
            mirrored_groups_to_remove = [
                mirrored_group
                for mirrored_group in list(self._mirror_status.mirrored_geometry_body_groups)
                if mirrored_group.geometry_body_group_id == body_group_id
            ]
            for mirrored_group in mirrored_groups_to_remove:
                self._remove(mirrored_group)

    # endregion ------------------------------------------------------------------------------------
    @property
    def _mirror_planes(self) -> List[MirrorPlane]:
        """Return the list of mirror planes."""
        return self._mirror_status.mirror_planes

    @_mirror_planes.setter
    def _mirror_planes(self, *args, **kwargs):
        """Set the list of mirror planes."""
        raise NotImplementedError(
            "Mirror planes are managed by the mirror manager -> _mirror_status and cannot be assigned directly."
        )

    def _add(self, entity: Union[MirrorPlane, MirroredGeometryBodyGroup, MirroredSurface]) -> None:
        """Add an entity to the mirror status."""
        if self._entity_registry.contains(entity):
            return

        # pylint: disable=no-member
        if is_exact_instance(entity, MirrorPlane):
            self._mirror_status.mirror_planes.append(entity)
            self._entity_registry.register(entity)
        elif is_exact_instance(entity, MirroredGeometryBodyGroup):
            self._mirror_status.mirrored_geometry_body_groups.append(entity)
            self._entity_registry.register(entity)
        elif is_exact_instance(entity, MirroredSurface):
            self._mirror_status.mirrored_surfaces.append(entity)
            self._entity_registry.register(entity)
        else:
            raise Flow360RuntimeError(
                f"[Internal] Unsupported entity type: {type(entity).__name__}."
            )

    def _remove(self, entity: MirroredGeometryBodyGroup) -> None:
        """Remove an MirroredGeometryBodyGroup from the mirror status."""
        # pylint: disable=no-member

        if entity in self._mirror_status.mirrored_geometry_body_groups:
            self._mirror_status.mirrored_geometry_body_groups.remove(entity)
        self._entity_registry.remove(entity)

        # Now remove the mirrored surfaces that are associated with this body group.
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

    def _to_status(self) -> MirrorStatus:
        """Build a serializable status snapshot.

        Parameters
        ----------
        entity_registry : EntityRegistry
            The entity registry to validate entity references against.

        Returns
        -------
        Optional[MirrorStatus]
            The serialized mirror status, or None if no valid mirror actions exist.
        """

        entity_registry = self._entity_registry

        # Build a set of existing GeometryBodyGroup IDs in the registry for validation.
        existing_body_group_ids = set()
        for entity in entity_registry.find_by_type(GeometryBodyGroup):
            if is_exact_instance(entity, GeometryBodyGroup):
                existing_body_group_ids.add(entity.private_attribute_id)

        # Filter out actions that refer to body groups that no longer exist in the registry.
        filtered_actions: Dict[str, str] = {}
        for body_group_id, mirror_plane_id in self._body_group_id_to_mirror_id.items():
            if body_group_id not in existing_body_group_ids:
                log.debug(
                    "GeometryBodyGroup '%s' assigned to mirror plane '%s' is not in the draft registry; "
                    "skipping this mirror action.",
                    body_group_id,
                    mirror_plane_id,
                )
                continue
            filtered_actions[body_group_id] = mirror_plane_id

        if not filtered_actions:
            # No valid mirror actions – nothing to serialize.
            return MirrorStatus(
                mirror_planes=[], mirrored_geometry_body_groups=[], mirrored_surfaces=[]
            )

        mirrored_geometry_groups, mirrored_surfaces = _derive_mirrored_entities_from_actions(
            body_group_id_to_mirror_id=filtered_actions,
            face_group_to_body_group=self._face_group_to_body_group,
            entity_registry=entity_registry,
            mirror_planes=self._mirror_planes,
        )

        # Only keep mirror planes that are actually referenced by the filtered actions.
        mirror_planes_by_id: Dict[str, MirrorPlane] = {
            plane.private_attribute_id: plane for plane in self._mirror_planes
        }
        used_plane_ids = {
            mirror_plane_id
            for mirror_plane_id in filtered_actions.values()
            if mirror_plane_id in mirror_planes_by_id
        }
        mirror_planes_for_status: List[MirrorPlane] = [
            plane for plane in self._mirror_planes if plane.private_attribute_id in used_plane_ids
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
        status: Optional[MirrorStatus],
        face_group_to_body_group: Optional[Dict[str, str]],
        entity_registry: EntityRegistry,
    ) -> "MirrorManager":
        """Restore manager from a status snapshot.

        Parameters
        ----------
        status : Optional[MirrorStatus]
            The mirror status to restore from.
        face_group_to_body_group : Optional[Dict[str, str]]
            Mapping from surface name to owning body group ID.
        entity_registry : EntityRegistry
            Entity registry containing body groups and surfaces.

        Returns
        -------
        MirrorManager
            Restored mirror manager.
        """
        mgr = cls(
            face_group_to_body_group=face_group_to_body_group,
            entity_registry=entity_registry,
        )

        body_groups = entity_registry.view(  # pylint: disable=protected-access
            GeometryBodyGroup
        )._entities
        valid_body_group_ids = {body_group.private_attribute_id for body_group in body_groups}

        body_group_id_to_mirror_id = _extract_body_group_id_to_mirror_id_from_status(
            mirror_status=status,
            valid_body_group_ids=valid_body_group_ids,
        )

        mgr._body_group_id_to_mirror_id = body_group_id_to_mirror_id

        # Initialize with external mirror planes.
        # These are not tightly coupled with the persistent entities therefore can be initialized separately.
        mgr._mirror_status.mirror_planes = status.mirror_planes if status is not None else []

        mgr._mirror_status = mgr._to_status()

        for mirrored_group in mgr._mirror_status.mirrored_geometry_body_groups:
            mgr._add(mirrored_group)
        for mirrored_surface in mgr._mirror_status.mirrored_surfaces:
            mgr._add(mirrored_surface)
        for mirror_plane in mgr._mirror_status.mirror_planes:
            mgr._add(mirror_plane)

        return mgr

    # endregion ------------------------------------------------------------------------------------
