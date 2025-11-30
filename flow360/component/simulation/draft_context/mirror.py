"""Mirror plane, mirrored entities and helpers."""

from typing import ClassVar, Dict, List, Literal, Optional, Tuple

import pydantic as pd

from flow360.component.simulation.entity_info import EntityInfoModel, GeometryEntityInfo
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.entity_utils import generate_uuid
from flow360.component.simulation.primitives import GeometryBodyGroup, Surface
from flow360.component.simulation.unit_system import LengthType
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
    surface_id: str = pd.Field(description="ID of the surface to mirror.")
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
    entity_info: GeometryEntityInfo,
    surfaces: List[Surface],
    mirror_planes_by_id: Dict[str, MirrorPlane],
) -> List[MirroredSurface]:
    """Create mirrored surfaces for the requested body groups."""

    if not body_group_id_to_mirror_id:
        return []

    try:
        face_group_to_body_group = entity_info.get_face_group_to_body_group_id_map()
    except ValueError as exc:
        raise Flow360RuntimeError(
            "[Internal] Failed to derive surface-to-body-group mapping for mirroring."
        ) from exc

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
    entity_info: EntityInfoModel,
    body_groups: List[GeometryBodyGroup],
    surfaces: List[Surface],
    mirror_planes: List[MirrorPlane],
) -> Tuple[List[MirroredGeometryBodyGroup], List[MirroredSurface]]:
    """
    Derive mirrored entities (MirroredGeometryBodyGroup + MirroredSurface)
    based on the given ``body_group_id_to_mirror_id`` mapping.

    The ``body_group_id_to_mirror_id`` schema is::

        {geometry_body_group_id: mirror_plane_id}

    This helper is intended to be reusable both from within the draft context
    (for incremental updates) and before submission (for generating the full list
    of mirrored entities from the stored mirror status).
    """

    if not body_group_id_to_mirror_id or not isinstance(entity_info, GeometryEntityInfo):
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
        entity_info=entity_info,
        surfaces=surfaces,
        mirror_planes_by_id=mirror_planes_by_id,
    )

    return mirrored_geometry_groups, mirrored_surfaces


def _extract_body_group_id_to_mirror_id_from_status(
    *,
    mirror_status: MirrorStatus,
    entity_info: EntityInfoModel,
) -> Tuple[Dict[str, str], List[MirrorPlane]]:
    """
    Deserialize mirror actions from a :class:`MirrorStatus` instance.

    Returns a tuple of:
    - ``body_group_id_to_mirror_id``: mapping from geometry body group ID to mirror plane ID.
    - ``mirror_planes``: list of :class:`MirrorPlane` instances referenced by those actions.

    Any entries referencing geometry body groups that no longer exist in ``entity_info``
    are ignored.
    """

    if mirror_status is None:
        # No mirror feature used in the asset.
        log.debug("Mirror status not provided; no mirroring actions to restore.")
        return {}, []

    # Determine valid body group IDs based on current entity info.
    valid_body_group_ids: Optional[set[str]] = None
    if isinstance(entity_info, GeometryEntityInfo):
        valid_body_group_ids = {
            body_group.private_attribute_id
            for group in entity_info.grouped_bodies
            for body_group in group
        }

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

# region -----------------------------Public Functions Below-------------------------------------


def build_mirror_status(
    *,
    body_group_id_to_mirror_id: Dict[str, str],
    entity_info: EntityInfoModel,
    body_groups: List[GeometryBodyGroup],
    surfaces: List[Surface],
    mirror_planes: List[MirrorPlane],
) -> Optional[MirrorStatus]:
    """
    Construct a :class:`MirrorStatus` instance from the given mirror actions and entities.

    This will:
    - Drop any mirror actions that reference geometry body groups that no longer exist.
    - Derive the corresponding mirrored entities.
    - Restrict the stored mirror planes to those actually referenced.
    """

    # Filter out actions that refer to body groups that no longer exist.
    valid_body_group_ids = {body_group.private_attribute_id for body_group in body_groups}
    filtered_actions: Dict[str, str] = {
        body_group_id: mirror_plane_id
        for body_group_id, mirror_plane_id in body_group_id_to_mirror_id.items()
        if body_group_id in valid_body_group_ids
    }

    if not filtered_actions:
        # No valid mirror actions – nothing to serialize.
        return None

    mirrored_geometry_groups, mirrored_surfaces = _derive_mirrored_entities_from_actions(
        body_group_id_to_mirror_id=filtered_actions,
        entity_info=entity_info,
        body_groups=body_groups,
        surfaces=surfaces,
        mirror_planes=mirror_planes,
    )

    # Only keep mirror planes that are actually referenced by the filtered actions.
    mirror_planes_by_id: Dict[str, MirrorPlane] = {
        plane.private_attribute_id: plane for plane in mirror_planes
    }
    used_plane_ids = {
        mirror_plane_id
        for mirror_plane_id in filtered_actions.values()
        if mirror_plane_id in mirror_planes_by_id
    }
    mirror_planes_for_status: List[MirrorPlane] = [
        plane for plane in mirror_planes if plane.private_attribute_id in used_plane_ids
    ]

    return MirrorStatus(
        mirror_planes=mirror_planes_for_status,
        mirrored_geometry_body_groups=mirrored_geometry_groups,
        mirrored_surfaces=mirrored_surfaces,
    )


# endregion -------------------------------------------------------------------------------------
