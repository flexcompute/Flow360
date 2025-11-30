"""Draft context manager for local entity sandboxing."""

from __future__ import annotations

import copy
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Dict, List, Optional, get_args

from flow360.component.simulation.draft_context.mirror import (
    MirroredGeometryBodyGroup,
    MirroredSurface,
    MirrorPlane,
    _derive_mirrored_entities_from_actions,
)
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


def capture_into_draft(entity: EntityBase) -> EntityBase:
    """
    Capture an entity into the active draft context.
    """

    # TODO: so For this function the only usage that I can think of is capturing draft entities that are
    # created under the draft context So is it possible that we have this function in the constructor of these
    # models for example in the box or cylinder etc? Tell me how this design sounds to you and what's your
    # concern about this design?
    draft = get_active_draft()
    if draft is None:
        raise Flow360RuntimeError("Cannot capture entity because no draft context is active.")
    draft._capture_entity(entity)  # pylint: disable=protected-access
    return entity


class _SingleTypeEntityRegistry:
    """
    A thin view over `EntityRegistry` restricted to a single entity type.
    """

    def __init__(self, *, registry: EntityRegistry, entity_type: type[EntityBase]) -> None:
        self._registry = registry
        self._entity_type = entity_type

    def __iter__(self):
        return iter(self.entities)

    def __len__(self):
        return len(self.entities)

    def __dir__(self):
        # Limit tab-completion to read-only accessors.
        return ["__iter__", "__len__", "__getitem__"]

    @property
    def entities(self) -> list[EntityBase]:
        """Entities of the target type."""
        bucket = self._registry.get_bucket(by_type=self._entity_type)

        return [item for item in bucket.entities if is_exact_instance(item, self._entity_type)]

    def __getitem__(self, key: str) -> EntityBase | list[EntityBase]:
        """
        Support syntax like `draft.body_groups['body_group_1']`
        and `draft.body_groups['body_group*']` (glob only).
        """
        if not isinstance(key, str):
            raise Flow360ValueError(f"Entity naming pattern: {key} is not a string.")

        matcher = compile_glob_cached(key)
        matched = [entity for entity in self.entities if matcher.match(entity.name)]

        if not matched:
            raise ValueError(
                f"No entity found in registry with given name/naming pattern: '{key}'."
            )
        if len(matched) == 1:
            return matched[0]
        return matched


class DraftContext(AbstractContextManager["DraftContext"]):
    """Context manager that tracks locally modified simulation entities."""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, *, entity_info: EntityInfoModel) -> None:
        """
        Data members:
        - _token: Token to track the active draft context.

        - _mirror_status: Dictionary to track the mirror actions.
        The key is the GeometryBodyGroup ID and the value is MirrorPlane ID to mirror.

        - _mirror_planes: List to track the MirrorPlane entities.

        - _entity_registry: Registry of entities of self._entity_info.
                            This provides interface for user to access the entities in the draft.

        """

        if entity_info is None:
            raise Flow360RuntimeError(
                "[Internal] DraftContext requires `entity_info` to initialize."
            )
        self._token: Optional[Token] = None

        self._mirror_status: Dict[str, str] = {}
        self._mirror_planes: List[MirrorPlane] = []

        self._entity_info = copy.deepcopy(entity_info)
        self._entity_registry: EntityRegistry = self._entity_info.get_registry(None)

        # TODO: Handle draft entities here.
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

    # Non persistent entities
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

    @property
    def mirror_planes(self) -> List[MirrorPlane]:
        """
        Return the list of mirror planes in the draft.
        """
        return self._mirror_planes

    # endregion ------------------------------------------------------------------------------------

    # region -----------------------------Public Methods Below-------------------------------------
    def mirror(
        self, *, entities: List[EntityBase], mirror_plane: MirrorPlane
    ) -> tuple[list[MirroredGeometryBodyGroup], list[MirroredSurface]]:
        """
        Create mirrored GeometryBodyGroup (and its associated surfaces) for the given `MirrorPlane`.
        New entities will have "_<mirror>" in the name as suffix.

        Example
        -------
          >>> with fl.create_draft() as draft:
          >>>     mirror_plane = fl.MirrorPlane(center=(0, 0, 0)*fl.u.m, normal=(1, 0, 0))
          >>>     draft.mirror(entities=geometry["body1"], mirror_plane=mirror_plane)

        ====
        """

        # pylint: disable=fixme
        # TODO: Support EntitySelector for specifying the GeometryBodyGroup in the future?

        # 1. [Validation] Ensure `entities` are GeometryBodyGroup entities.
        # Note:We could in the future just move these into pd.validate_call but for first
        # roll out let's keep the error message clear and readable.
        normalized_entities: list[EntityBase]
        if isinstance(entities, EntityBase):
            normalized_entities = [entities]
        elif isinstance(entities, list):
            normalized_entities = entities
        else:
            raise Flow360RuntimeError(
                f"`entities` accepts a single entity or a list of entities. Received type: {type(entities).__name__}."
            )

        geometry_body_groups: list[GeometryBodyGroup] = []
        for entity in normalized_entities:
            if not is_exact_instance(entity, GeometryBodyGroup):
                raise Flow360RuntimeError(
                    "Only GeometryBodyGroup entities are supported by `mirror()` currently. "
                    f"Received: {type(entity).__name__}."
                )
            geometry_body_groups.append(entity)

        # 2. [Validation] Ensure `mirror_plane` is a `MirrorPlane` entity.
        if not is_exact_instance(mirror_plane, MirrorPlane):
            raise Flow360RuntimeError(
                f"`mirror_plane` must be a MirrorPlane entity. Instead received: {type(mirror_plane).__name__}."
            )

        # 3. [Restriction] Each GeometryBodyGroup entity can only be mirrored once.
        #                  If a duplicate request is made, reset to the new one with a warning.
        for body_group in geometry_body_groups:
            body_group_id = body_group.private_attribute_id
            if body_group_id in self._mirror_status:
                log.warning(
                    "GeometryBodyGroup `%s` was already mirrored; resetting to the latest mirror plane request.",
                    body_group.name,
                )

        # 4. Create/Update the self._mirror_status
        #    and also capture the MirrorPlane into the `draft`.
        mirror_actions_update: Dict[str, str] = {}
        for body_group in geometry_body_groups:
            body_group_id = body_group.private_attribute_id
            mirror_actions_update[body_group_id] = mirror_plane.private_attribute_id
            self._mirror_status[body_group_id] = mirror_plane.private_attribute_id

        existing_plane_ids = {plane.private_attribute_id for plane in self._mirror_planes}
        if mirror_plane.private_attribute_id not in existing_plane_ids:
            self._mirror_planes.append(mirror_plane)

        # 5. Derive the generated mirrored entities (MirroredGeometryBodyGroup + MirroredSurface)
        #    and return to user as tokens of use.
        return _derive_mirrored_entities_from_actions(
            mirror_actions=mirror_actions_update,
            entity_info=self._entity_info,
            body_groups=self._body_groups.entities,
            surfaces=self._surfaces.entities,
            mirror_planes=self._mirror_planes,
        )

    # endregion -------------------------------------------------------------------------------------
