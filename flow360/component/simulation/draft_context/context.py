"""Draft context manager for local entity sandboxing."""

from __future__ import annotations

from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import Optional, get_args

from flow360.component.simulation.entity_info import DraftEntityTypes, EntityInfoModel
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.entity_utils import compile_glob_cached
from flow360.component.simulation.primitives import (
    Edge,
    GenericVolume,
    GeometryBodyGroup,
    Surface,
)
from flow360.component.simulation.utils import is_exact_instance
from flow360.exceptions import Flow360RuntimeError, Flow360ValueError
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


class _SingleTypeEntityRegistry:
    """
    A thin view over `EntityRegistry` restricted to a single entity type.
    """

    def __init__(self, *, registry: EntityRegistry, entity_type: type[EntityBase]) -> None:
        self._registry = registry
        self._entity_type = entity_type

    def __iter__(self):
        return iter(self._entities)

    def __len__(self):
        return len(self._entities)

    @property
    def _entities(self) -> list[EntityBase]:
        """Entities of the target type."""
        return self._registry.view(self._entity_type)._entities

    def __getitem__(self, key: str) -> EntityBase | list[EntityBase]:
        """
        Support syntax like `draft.body_groups['body_group_1']`
        and `draft.body_groups['body_group*']` (glob only).
        """
        if not isinstance(key, str):
            raise Flow360ValueError(f"Entity naming pattern: {key} is not a string.")

        matcher = compile_glob_cached(key)
        matched = [entity for entity in self._entities if matcher.match(entity.name)]

        if not matched:
            raise ValueError(
                f"No entity found in registry with given name/naming pattern: '{key}'."
            )
        if len(matched) == 1:
            return matched[0]
        return matched


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
        "_body_groups",
        "_surfaces",
        "_edges",
        "_volumes",
        "_coordinate_system_manager",
        "_mirror_manager",
        "_token",
    )

    def __init__(
        self,
        *,
        entity_info: EntityInfoModel,
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

        # Direct reference to entity_info
        # Modifications to entities will be reflected in the asset's entity_info,
        # mimicking web UI behavior where users can directly edit entity properties.
        self._entity_info = entity_info
        self._entity_registry: EntityRegistry = self._entity_info.get_persistent_entity_registry(
            None
        )

        # Persistent entities (referencing objects in the _entity_info)
        self._body_groups = _SingleTypeEntityRegistry(
            registry=self._entity_registry, entity_type=GeometryBodyGroup
        )
        self._surfaces = _SingleTypeEntityRegistry(
            registry=self._entity_registry, entity_type=Surface
        )
        self._edges = _SingleTypeEntityRegistry(registry=self._entity_registry, entity_type=Edge)
        self._volumes = _SingleTypeEntityRegistry(
            registry=self._entity_registry, entity_type=GenericVolume
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
    def body_groups(self) -> _SingleTypeEntityRegistry:
        """
        Return the list of body groups in the draft.


        Example
        -------
          >>> with fl.create_draft(new_run_from=geometry) as draft:
          >>>     draft.body_groups["body_group_1"]
          >>>     draft.body_groups["body_group*"]

        ====
        """
        return self._body_groups

    @property
    def surfaces(self) -> _SingleTypeEntityRegistry:
        """
        Return the list of surfaces in the draft.
        """
        return self._surfaces

    @property
    def edges(self) -> _SingleTypeEntityRegistry:
        """
        Return the list of edges in the draft.
        """
        return self._edges

    @property
    def volumes(self) -> _SingleTypeEntityRegistry:
        """
        Return the list of volumes (volume zones) in the draft.
        """
        # If volume zone as root asset.
        return self._volumes

    # Non-persistent entities
    @property
    def boxes(self) -> _SingleTypeEntityRegistry:
        """
        Return the list of boxes in the draft.
        """

    @property
    def cylinders(self) -> _SingleTypeEntityRegistry:
        """
        Return the list of cylinders in the draft.
        """

    # endregion ------------------------------------------------------------------------------------

    # endregion -------------------------------------------------------------------------------------
