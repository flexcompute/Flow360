"""Draft context manager for local entity sandboxing."""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import Optional, get_args

from flow360.component.simulation.draft_context.coordinate_system_manager import (
    CoordinateSystemManager,
    CoordinateSystemStatus,
)
from flow360.component.simulation.draft_context.mirror import (
    MirrorManager,
    MirrorStatus,
)
from flow360.component.simulation.entity_info import (
    DraftEntityTypes,
    EntityInfoModel,
    GeometryEntityInfo,
)
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.entity_registry import (
    EntityRegistry,
    EntityRegistryView,
)
from flow360.component.simulation.primitives import (
    Edge,
    GenericVolume,
    GeometryBodyGroup,
    Surface,
)
from flow360.exceptions import Flow360RuntimeError
from flow360.log import log

__all__ = [
    "DraftContext",
    "get_active_draft",
]


_ACTIVE_DRAFT: ContextVar[DraftContext | None] = ContextVar("_ACTIVE_DRAFT", default=None)

_DRAFT_ENTITY_TYPE_TUPLE: tuple[type[EntityBase], ...] = tuple(
    get_args(get_args(DraftEntityTypes)[0])
)


def get_active_draft() -> DraftContext | None:
    """Return the current active draft context if any."""
    return _ACTIVE_DRAFT.get()


class DraftContext(  # pylint: disable=too-many-instance-attributes
    AbstractContextManager["DraftContext"]
):
    """
    Context manager that tracks locally modified simulation entities/status.
    This should (eventually, not right now) be replacement of accessing entities directly from assets.
    """

    __slots__ = (
        "_entity_info",
        "_entity_registry",
        "_mirror_manager",
        "_coordinate_system_manager",
        "_token",
    )

    def __init__(
        self,
        *,
        entity_info: EntityInfoModel,
        mirror_status: Optional[MirrorStatus] = None,
        coordinate_system_status: Optional[CoordinateSystemStatus] = None,
    ) -> None:
        """
        Data members:
        - _token: Token to track the active draft context.

        - _mirror_manager: Manager for mirror planes and mirrored entities.

        - _entity_registry: Registry of entities of self._entity_info.
                            This provides interface for user to access the entities in the draft.

        """

        if entity_info is None:
            raise Flow360RuntimeError(
                "[Internal] DraftContext requires `entity_info` to initialize."
            )
        self._token: Optional[Token] = None

        # DraftContext owns a deep copy of entity_info (created by create_draft()).
        # This ensures modifications in the draft don't affect the original asset.
        self._entity_info = entity_info

        # Use EntityRegistry.from_entity_info() for the new DraftContext workflow.
        # This builds the registry by referencing entities from our copied entity_info.
        self._entity_registry: EntityRegistry = EntityRegistry.from_entity_info(entity_info)

        # Pre-compute face_group_to_body_group map for mirror operations.
        # This is only available for GeometryEntityInfo.
        face_group_to_body_group = None

        if isinstance(self._entity_info, GeometryEntityInfo):
            try:
                face_group_to_body_group = self._entity_info.get_face_group_to_body_group_id_map()
            except ValueError as exc:
                # Face grouping spans across body groups.
                # TODO: Are you sure just impacts the surface?
                log.warning(
                    "Failed to derive surface-to-body-group mapping for mirroring: %s. "
                    "Mirroring surfaces will be disabled.",
                    exc,
                )

        self._mirror_manager = MirrorManager._from_status(
            status=mirror_status,
            face_group_to_body_group=face_group_to_body_group,
            body_groups=self._entity_registry.view(GeometryBodyGroup)._entities,
            surfaces=self._entity_registry.view(Surface)._entities,
        )

        self._coordinate_system_manager = CoordinateSystemManager._from_status(
            status=coordinate_system_status,
        )

    def __enter__(self) -> DraftContext:
        if get_active_draft() is not None:
            raise Flow360RuntimeError("Nested draft contexts are not allowed.")
        self._token = _ACTIVE_DRAFT.set(self)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._token is None:
            raise Flow360RuntimeError(
                "[Internal] DraftContext exit called without a matching enter."
            )
        _ACTIVE_DRAFT.reset(self._token)
        self._token = None
        return False

    # region -----------------------------Private implementations Below-----------------------------

    # endregion ------------------------------------------------------------------------------------

    # region -----------------------------Public properties Below-------------------------------------

    # Persistent entities
    @property
    def body_groups(self) -> EntityRegistryView:
        """
        Return the list of body groups in the draft.


        Example
        -------
          >>> with fl.create_draft(new_run_from=geometry) as draft:
          >>>     draft.body_groups["body_group_1"]
          >>>     draft.body_groups["body_group*"]

        ====
        """
        return self._entity_registry.view(GeometryBodyGroup)

    @property
    def surfaces(self) -> EntityRegistryView:
        """
        Return the list of surfaces in the draft.
        """
        return self._entity_registry.view(Surface)

    @property
    def edges(self) -> EntityRegistryView:
        """
        Return the list of edges in the draft.
        """
        return self._entity_registry.view(Edge)

    @property
    def volumes(self) -> EntityRegistryView:
        """
        Return the list of volumes (volume zones) in the draft.
        """
        return self._entity_registry.view(GenericVolume)

    # Non-persistent entities
    @property
    def boxes(self) -> EntityRegistryView:
        """
        Return the list of boxes in the draft.
        """
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.primitives import Box

        return self._entity_registry.view(Box)

    @property
    def cylinders(self) -> EntityRegistryView:
        """
        Return the list of cylinders in the draft.
        """
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.primitives import Cylinder

        return self._entity_registry.view(Cylinder)

    @property
    def coordinate_systems(self) -> CoordinateSystemManager:
        """Coordinate system manager."""
        return self._coordinate_system_manager

    @property
    def mirror(self) -> MirrorManager:
        """Mirror manager."""
        return self._mirror_manager

    # endregion ------------------------------------------------------------------------------------
