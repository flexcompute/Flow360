"""Mirror plane, mirrored entities and helpers."""

from __future__ import annotations

from flow360_schema.framework.entity.mirror_state import (
    MIRROR_SUFFIX,
    MirrorState,
    _derive_mirrored_entities_from_actions,
)
from flow360_schema.models.asset_cache import MirrorStatus
from flow360_schema.models.entities.geometry_entities import MirrorPlane

from flow360.component.simulation.framework.entity_registry import (
    EntityRegistry,
    EntityRegistryView,
)
from flow360.component.simulation.primitives import (
    GeometryBodyGroup,
    MirroredGeometryBodyGroup,
    MirroredSurface,
)
from flow360.component.simulation.utils import is_exact_instance
from flow360.exceptions import Flow360ValueError

# pylint: disable=protected-access


__all__ = [
    "MIRROR_SUFFIX",
    "MirrorManager",
    "MirrorPlane",
    "MirrorStatus",
    "_derive_mirrored_entities_from_actions",
]


class MirrorManager:
    """
    Manage mirror planes and mirrored draft entities inside a `DraftContext`.

    This manager provides:

    - Storage/registration of `MirrorPlane` entities.
    - Creation/removal of mirror actions for `GeometryBodyGroup` entities.
    - Derivation and registration of draft-only entities:
      `MirroredGeometryBodyGroup` and (when possible) `MirroredSurface`.

    Notes
    -----
    Surface mirroring requires a surface-to-body-group mapping derived from `GeometryEntityInfo`.
    If that mapping cannot be derived (e.g. face grouping spans multiple body groups), mirroring
    is disabled and mirror operations will raise.
    """

    # Keep the manager as a thin client-facing facade. We intentionally use composition
    # instead of inheriting from MirrorState so the client does not implicitly expose
    # every schema-core helper as part of its API surface.
    __slots__ = ("_state",)

    def __init__(
        self,
        *,
        face_group_to_body_group: dict[str, str] | None = None,
        entity_registry: EntityRegistry | None = None,
        state: MirrorState | None = None,
    ) -> None:
        if state is None:
            if entity_registry is None:
                raise Flow360ValueError(
                    "[Internal] MirrorManager requires `entity_registry` when `state` is not provided."
                )
            state = MirrorState(
                face_group_to_body_group=face_group_to_body_group,
                entity_registry=entity_registry,
            )
        self._state = state

    @property
    def mirror_planes(self) -> EntityRegistryView:
        """
        Return all the available mirror planes.

        Returns
        -------
        EntityRegistryView
            A registry view of `MirrorPlane` entities available in this draft.
        """
        return self._state.mirror_planes

    @property
    def _mirror_status(self) -> MirrorStatus:
        """Read-only compatibility surface for existing draft plumbing and tests."""
        return self._state._mirror_status

    @property
    def _body_group_id_to_mirror_id(self) -> dict[str, str]:
        """Read-only compatibility surface for existing draft plumbing and tests."""
        return self._state._body_group_id_to_mirror_id

    @property
    def _mirror_planes(self) -> list[MirrorPlane]:
        """Read-only compatibility surface for existing draft plumbing and tests."""
        return self._state._mirror_planes

    def create_mirror_of(
        self,
        *,
        entities: list[GeometryBodyGroup] | GeometryBodyGroup,
        mirror_plane: MirrorPlane,
    ) -> tuple[list[MirroredGeometryBodyGroup], list[MirroredSurface]]:
        """
        Create mirrored entities for one or more geometry body groups.

        This registers mirror actions for the requested body groups and then derives/creates
        draft-only entities:

        - `MirroredGeometryBodyGroup` for each body group
        - `MirroredSurface` for each surface belonging to those body groups
          (when surface ownership mapping is available)

        Newly created mirrored entities use `MIRROR_SUFFIX` (``"_<mirror>"``) as a name suffix.

        Parameters
        ----------
        entities : list[GeometryBodyGroup] | GeometryBodyGroup
            One or more geometry body groups to mirror.
        mirror_plane : MirrorPlane
            The mirror plane to use for mirroring.

        Returns
        -------
        tuple[list[MirroredGeometryBodyGroup], list[MirroredSurface]]
            Mirrored geometry body groups and surfaces.
        """
        normalized_entities = [entities] if isinstance(entities, GeometryBodyGroup) else entities
        for entity in normalized_entities:
            if not is_exact_instance(entity, GeometryBodyGroup):
                raise Flow360ValueError(
                    "Only GeometryBodyGroup entities are supported by `create()` currently. "
                    f"Received: {type(entity).__name__}."
                )
            if entity.private_attribute_id is None:
                raise Flow360ValueError(
                    f"Entity '{entity.name}' ({type(entity).__name__}) is not supported for mirror operations."
                )

        return self._state.create_mirror_of(
            entities=entities,
            mirror_plane=mirror_plane,
        )

    def remove_mirror_of(self, *, entities: list[GeometryBodyGroup] | GeometryBodyGroup) -> None:
        """
        Remove the mirror of the given entities.

        Parameters
        ----------
        entities : list[GeometryBodyGroup] | GeometryBodyGroup
            One or more geometry body groups to remove mirroring from.
        """
        self._state.remove_mirror_of(entities=entities)

    @classmethod
    def _from_status(
        cls,
        *,
        status: MirrorStatus | None,
        face_group_to_body_group: dict[str, str] | None,
        entity_registry: EntityRegistry,
    ) -> "MirrorManager":
        """
        Restore manager from a status snapshot.

        Parameters
        ----------
        status : MirrorStatus | None
            The mirror status to restore from.
        face_group_to_body_group : dict[str, str] | None
            Mapping from surface name to owning body group ID.
        entity_registry : EntityRegistry
            Entity registry containing body groups and surfaces.

        Returns
        -------
        MirrorManager
            Restored mirror manager.
        """
        if status is None or status.is_empty():
            return cls(
                face_group_to_body_group=face_group_to_body_group,
                entity_registry=entity_registry,
            )

        return cls(
            state=MirrorState._from_status(
                status=status,
                face_group_to_body_group=face_group_to_body_group,
                entity_registry=entity_registry,
            )
        )
