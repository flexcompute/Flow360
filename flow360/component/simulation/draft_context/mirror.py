"""Mirror plane, mirrored entities and helpers."""

from typing import ClassVar, Dict, List, Literal, Optional, Tuple, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.entity_utils import generate_uuid
from flow360.component.simulation.primitives import GeometryBodyGroup, Surface
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
    entity_bucket: ClassVar[str] = "MirrorPlaneType"
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)


class MirroredGeometryBodyGroup(EntityBase):
    """
    :class:`MirroredGeometryBodyGroup` class for representing a mirrored geometry body group.
    """

    name: str = pd.Field()
    geometry_body_group_id: str = pd.Field(description="ID of the geometry body group to mirror.")
    mirror_plane_id: str = pd.Field(
        description="ID of the mirror plane to mirror the geometry body group."
    )

    private_attribute_entity_type_name: Literal["MirroredGeometryBodyGroup"] = pd.Field(
        "MirroredGeometryBodyGroup", frozen=True
    )
    entity_bucket: ClassVar[str] = "MirroredGeometryBodyGroupType"
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)


class MirroredSurface(EntityBase):
    """
    :class:`MirroredSurface` class for representing a mirrored surface.
    """

    name: str = pd.Field()
    surface_id: str = pd.Field(
        description="ID of the original surface being mirrored.", frozen=True
    )
    mirror_plane_id: str = pd.Field(description="ID of the mirror plane to mirror the surface.")

    private_attribute_entity_type_name: Literal["MirroredSurface"] = pd.Field(
        "MirroredSurface", frozen=True
    )
    entity_bucket: ClassVar[str] = "MirroredSurfaceType"
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)


# region -----------------------------Internal Model Below-------------------------------------
class MirrorStatus(Flow360BaseModel):
    """
    Internal model for storing the mirror status.
    """

    # Note: We can do similar thing as entityList to support mirroring with EntitySelectors.
    mirror_planes: List[MirrorPlane] = pd.Field(description="List of mirror planes to mirror.")
    mirrored_geometry_body_groups: List[MirroredGeometryBodyGroup] = pd.Field(
        description="List of mirrored geometry body groups."
    )
    mirrored_surfaces: List[MirroredSurface] = pd.Field(description="List of mirrored surfaces.")


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

        mirrored_surfaces.append(
            MirroredSurface(
                name=f"{surface.name}{MIRROR_SUFFIX}",
                surface_id=surface.private_attribute_id,
                mirror_plane_id=mirror_plane_id,
            )
        )

    return mirrored_surfaces


def _derive_mirrored_entities_from_actions(
    *,
    body_group_id_to_mirror_id: Dict[str, str],
    face_group_to_body_group: Optional[Dict[str, str]],
    body_groups: List[GeometryBodyGroup],
    surfaces: List[Surface],
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
    body_groups : List[GeometryBodyGroup]
        List of all body groups.
    surfaces : List[Surface]
        List of all surfaces.
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
    mirror_status: MirrorStatus,
    valid_body_group_ids: Optional[set[str]],
) -> Tuple[Dict[str, str], List[MirrorPlane]]:
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
    Tuple[Dict[str, str], List[MirrorPlane]]
        A tuple of:
        - ``body_group_id_to_mirror_id``: mapping from geometry body group ID to mirror plane ID.
        - ``mirror_planes``: list of :class:`MirrorPlane` instances referenced by those actions.
    """

    if mirror_status is None:
        # No mirror feature used in the asset.
        log.debug("Mirror status not provided; no mirroring actions to restore.")
        return {}, []

    mirror_planes_by_id: Dict[str, MirrorPlane] = {
        plane.private_attribute_id: plane for plane in mirror_status.mirror_planes
    }

    body_group_id_to_mirror_id: Dict[str, str] = {}
    for mirrored_group in mirror_status.mirrored_geometry_body_groups:
        body_group_id = mirrored_group.geometry_body_group_id
        mirror_plane_id = mirrored_group.mirror_plane_id

        if valid_body_group_ids is not None and body_group_id not in valid_body_group_ids:
            # Skip body groups that no longer exist.
            continue

        if mirror_plane_id not in mirror_planes_by_id:
            # Skip if the referenced mirror plane is no longer present.
            continue

        body_group_id_to_mirror_id[body_group_id] = mirror_plane_id

    used_plane_ids = set(body_group_id_to_mirror_id.values())
    mirror_planes: List[MirrorPlane] = [
        plane
        for plane in mirror_status.mirror_planes
        if plane.private_attribute_id in used_plane_ids
    ]

    return body_group_id_to_mirror_id, mirror_planes


# endregion -------------------------------------------------------------------------------------


class MirrorManager:
    """Encapsulates mirror plane registry and entity mirroring operations."""

    __slots__ = (
        "_mirror_planes",
        "_body_group_id_to_mirror_id",
        "_face_group_to_body_group",
        "_body_groups",
        "_surfaces",
    )

    def __init__(
        self,
        *,
        face_group_to_body_group: Optional[Dict[str, str]],
        body_groups: List[GeometryBodyGroup],
        surfaces: List[Surface],
    ) -> None:
        self._mirror_planes: List[MirrorPlane] = []
        self._body_group_id_to_mirror_id: Dict[str, str] = {}
        self._face_group_to_body_group = face_group_to_body_group
        # _body_groups and _surfaces are needed to return token entities. But I prefer not having these as members.
        self._body_groups = body_groups
        self._surfaces = surfaces

    # region Public API -------------------------------------------------

    def get_mirror_plane(self, name: str) -> MirrorPlane:
        """
        Retrieve a mirror plane by name.

        Parameters
        ----------
        name : str
            Name of the mirror plane.

        Returns
        -------
        MirrorPlane
            The mirror plane with the specified name.

        Raises
        ------
        Flow360RuntimeError
            If no mirror plane with the given name exists.
        """
        for plane in self._mirror_planes:
            if plane.name == name:
                return plane
        raise Flow360RuntimeError(f"Mirror plane '{name}' not found in the draft.")

    def create(
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

        # 3. [Restriction] Each GeometryBodyGroup entity can only be mirrored once.
        #                  If a duplicate request is made, reset to the new one with a warning.
        for body_group in normalized_entities:
            body_group_id = body_group.private_attribute_id
            if body_group_id in self._body_group_id_to_mirror_id:
                log.warning(
                    "GeometryBodyGroup `%s` was already mirrored; resetting to the latest mirror plane request.",
                    body_group.name,
                )

        # 4. Create/Update the self._body_group_id_to_mirror_id
        #    and also capture the MirrorPlane into the manager.
        body_group_id_to_mirror_id_update: Dict[str, str] = {}
        for body_group in normalized_entities:
            body_group_id = body_group.private_attribute_id
            body_group_id_to_mirror_id_update[body_group_id] = mirror_plane.private_attribute_id
            self._body_group_id_to_mirror_id[body_group_id] = mirror_plane.private_attribute_id

        existing_plane_ids = {plane.private_attribute_id for plane in self._mirror_planes}
        if mirror_plane.private_attribute_id not in existing_plane_ids:
            self._mirror_planes.append(mirror_plane)

        # 5. Derive the generated mirrored entities (MirroredGeometryBodyGroup + MirroredSurface)
        #    and return to user as tokens of use.
        return _derive_mirrored_entities_from_actions(
            body_group_id_to_mirror_id=body_group_id_to_mirror_id_update,
            face_group_to_body_group=self._face_group_to_body_group,
            body_groups=self._body_groups,
            surfaces=self._surfaces,
            mirror_planes=self._mirror_planes,
        )

    # endregion ------------------------------------------------------------------------------------

    def _to_status(self) -> Optional[MirrorStatus]:
        """Build a serializable status snapshot."""
        # Filter out actions that refer to body groups that no longer exist.
        valid_body_group_ids = {body_group.private_attribute_id for body_group in self._body_groups}
        filtered_actions: Dict[str, str] = {
            body_group_id: mirror_plane_id
            for body_group_id, mirror_plane_id in self._body_group_id_to_mirror_id.items()
            if body_group_id in valid_body_group_ids
        }

        if not filtered_actions:
            # No valid mirror actions – nothing to serialize.
            return None

        mirrored_geometry_groups, mirrored_surfaces = _derive_mirrored_entities_from_actions(
            body_group_id_to_mirror_id=filtered_actions,
            face_group_to_body_group=self._face_group_to_body_group,
            body_groups=self._body_groups,
            surfaces=self._surfaces,
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
    def _from_status(  # pylint: disable=too-many-arguments
        cls,
        *,
        status: Optional[MirrorStatus],
        face_group_to_body_group: Optional[Dict[str, str]],
        valid_body_group_ids: Optional[set[str]],
        body_groups: List[GeometryBodyGroup],
        surfaces: List[Surface],
    ) -> "MirrorManager":
        """Restore manager from a status snapshot.

        Parameters
        ----------
        status : Optional[MirrorStatus]
            The mirror status to restore from.
        face_group_to_body_group : Optional[Dict[str, str]]
            Mapping from surface name to owning body group ID.
        valid_body_group_ids : Optional[set[str]]
            Set of valid body group IDs. Used to filter out stale mirror actions.
        body_groups : List[GeometryBodyGroup]
            List of all body groups.
        surfaces : List[Surface]
            List of all surfaces.

        Returns
        -------
        MirrorManager
            Restored mirror manager.
        """
        mgr = cls(
            face_group_to_body_group=face_group_to_body_group,
            body_groups=body_groups,
            surfaces=surfaces,
        )

        if status is None:
            return mgr

        body_group_id_to_mirror_id, mirror_planes = _extract_body_group_id_to_mirror_id_from_status(
            mirror_status=status,
            valid_body_group_ids=valid_body_group_ids,
        )

        mgr._body_group_id_to_mirror_id = body_group_id_to_mirror_id
        mgr._mirror_planes = mirror_planes

        return mgr

    # endregion ------------------------------------------------------------------------------------
