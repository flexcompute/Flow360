"""Coordinate system management for DraftContext."""

from __future__ import annotations

from typing import List, Optional, Union

from flow360_schema.framework.entity.coordinate_system_state import (
    CoordinateSystemState,
)
from flow360_schema.models.asset_cache import (
    CoordinateSystemAssignmentGroup,
    CoordinateSystemEntityRef,
    CoordinateSystemParent,
    CoordinateSystemStatus,
)

from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.utils import is_exact_instance
from flow360.exceptions import Flow360ValueError
from flow360.log import log

# pylint: disable=protected-access,unused-import


__all__ = [
    "CoordinateSystemAssignmentGroup",
    "CoordinateSystemEntityRef",
    "CoordinateSystemManager",
    "CoordinateSystemParent",
    "CoordinateSystemStatus",
]


class CoordinateSystemManager:
    """
    Manage coordinate systems, hierarchy, and entity assignments inside a `DraftContext`.

    This manager provides:
    - Registration of coordinate systems with optional parent relationships (inheritance).
    - Validation of the parent graph (no missing parents, no cycles).
    - Assignment of coordinate systems to draft entities.

    Notes
    -----
    Coordinate systems are treated as draft-local objects. They are registered and validated
    here, and can be serialized to a `CoordinateSystemStatus` snapshot for persistence in the
    asset cache.
    """

    # Keep the manager as a thin client-facing facade. We intentionally use composition
    # instead of inheriting from CoordinateSystemState so the client does not implicitly
    # expose every schema-core helper as part of its API surface.
    __slots__ = ("_state",)

    def __init__(self, *, state: Optional[CoordinateSystemState] = None) -> None:
        self._state = CoordinateSystemState() if state is None else state

    @property
    def _coordinate_systems(self) -> list[CoordinateSystem]:
        """Read-only compatibility surface for existing tests and draft internals."""
        return self._state._coordinate_systems

    @property
    def _coordinate_system_parents(self) -> dict[str, Optional[str]]:
        """Read-only compatibility surface for existing tests and draft internals."""
        return self._state._coordinate_system_parents

    @property
    def _entity_key_to_coordinate_system_id(self) -> dict[tuple[str, str], str]:
        """Read-only compatibility surface for existing tests and draft internals."""
        return self._state._entity_key_to_coordinate_system_id

    def _get_coordinate_system_by_id(self, coordinate_system_id: str) -> Optional[CoordinateSystem]:
        return self._state._get_coordinate_system_by_id(coordinate_system_id)

    def _get_coordinate_system_matrix(self, *, coordinate_system: CoordinateSystem):
        return self._state._get_coordinate_system_matrix(coordinate_system=coordinate_system)

    def _get_coordinate_system_for_entity(self, *, entity: EntityBase):
        return self._state._get_coordinate_system_for_entity(entity=entity)

    def _get_matrix_for_entity(self, *, entity: EntityBase):
        return self._state._get_matrix_for_entity(entity=entity)

    def _get_matrix_for_entity_key(self, *, entity_type: str, entity_id: str):
        return self._state._get_matrix_for_entity_key(entity_type=entity_type, entity_id=entity_id)

    def _to_status(self) -> CoordinateSystemStatus:
        return self._state._to_status()

    def add(
        self, coordinate_system: CoordinateSystem, *, parent: Optional[CoordinateSystem] = None
    ) -> CoordinateSystem:
        """
        Register a coordinate system in this draft, optionally with a parent.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            The coordinate system instance to register.
        parent : CoordinateSystem, optional
            Optional parent coordinate system. If provided and not yet registered, the parent
            will be auto-registered as a root coordinate system first.

        Returns
        -------
        CoordinateSystem
            The registered coordinate system (same instance as `coordinate_system`).

        Raises
        ------
        Flow360ValueError
            If `coordinate_system` is not an exact `CoordinateSystem` instance, if the id or
            name is already registered, or if the resulting parent graph is invalid (e.g. cycle).
        """
        if not is_exact_instance(coordinate_system, CoordinateSystem):
            raise Flow360ValueError(
                f"coordinate_system must be a CoordinateSystem. Received: {type(coordinate_system).__name__}."
            )
        cs = coordinate_system

        if parent is not None:
            self._state._register_coordinate_system(coordinate_system=parent, parent_id=None)

        if self._state._contains(cs):
            raise Flow360ValueError(
                f"Coordinate system id '{cs.private_attribute_id}' already registered."
            )
        if any(existing.name == cs.name for existing in self._state._coordinate_systems):
            raise Flow360ValueError(f"Coordinate system name '{cs.name}' already registered.")

        self._state._coordinate_systems.append(cs)
        self._state._coordinate_system_parents[cs.private_attribute_id] = (
            parent.private_attribute_id if parent is not None else None
        )
        self._state._validate_coordinate_system_graph()
        return cs

    def update_parent(
        self, *, coordinate_system: CoordinateSystem, parent: Optional[CoordinateSystem]
    ) -> None:
        """
        Update the parent of a registered coordinate system.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            A coordinate system that is already registered in this manager.
        parent : CoordinateSystem, optional
            The new parent. If provided and not yet registered, it will be auto-registered as
            a root coordinate system first.

        Raises
        ------
        Flow360ValueError
            If `coordinate_system` is not registered, or if updating the parent would make the
            parent graph invalid (e.g. introduce a cycle). In that case, the change is rolled back.
        """
        if not self._state._contains(coordinate_system):
            raise Flow360ValueError("Coordinate system must be part of the draft to be updated.")

        if parent is not None:
            self._state._register_coordinate_system(coordinate_system=parent, parent_id=None)

        coordinate_system_id = coordinate_system.private_attribute_id
        original_parent = self._state._coordinate_system_parents.get(coordinate_system_id)
        self._state._coordinate_system_parents[coordinate_system_id] = (
            parent.private_attribute_id if parent else None
        )

        try:
            self._state._validate_coordinate_system_graph()
        except Flow360ValueError:
            self._state._coordinate_system_parents[coordinate_system_id] = original_parent
            raise

    def remove(self, coordinate_system: CoordinateSystem) -> None:
        """
        Remove a coordinate system if it has no dependents.

        This also removes any entity assignments that reference the removed coordinate system.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            The coordinate system to remove.

        Raises
        ------
        Flow360ValueError
            If the coordinate system is not registered, or if other registered coordinate systems
            depend on it (i.e. it is a parent).
        """
        if not self._state._contains(coordinate_system):
            raise Flow360ValueError("Coordinate system is not registered in this draft.")

        coordinate_system_id = coordinate_system.private_attribute_id
        dependents = [
            child_id
            for child_id, parent_id in self._state._coordinate_system_parents.items()
            if parent_id == coordinate_system_id
        ]
        if dependents:
            names = ", ".join(
                coordinate_system.name
                for coordinate_system in self._state._coordinate_systems
                if coordinate_system.private_attribute_id in dependents
            )
            raise Flow360ValueError(
                f"Cannot remove coordinate system '{coordinate_system.name}' because dependents exist: {names}"
            )

        self._state._coordinate_systems = [
            existing
            for existing in self._state._coordinate_systems
            if existing.private_attribute_id != coordinate_system.private_attribute_id
        ]
        self._state._coordinate_system_parents.pop(coordinate_system_id, None)
        self._state._entity_key_to_coordinate_system_id = {
            entity_id: assigned_id
            for entity_id, assigned_id in self._state._entity_key_to_coordinate_system_id.items()
            if assigned_id != coordinate_system_id
        }

    def assign(
        self,
        *,
        entities: Union[List[EntityBase], EntityBase],
        coordinate_system: CoordinateSystem,
    ) -> None:
        """
        Assign a coordinate system to one or more draft entities.

        Parameters
        ----------
        entities : EntityBase or list[EntityBase]
            One entity or a list of entities to assign.
        coordinate_system : CoordinateSystem
            Coordinate system to assign. If not yet registered, it is auto-registered as a
            root coordinate system.

        Raises
        ------
        Flow360ValueError
            If `coordinate_system` is not an exact `CoordinateSystem` instance or if any item
            in `entities` is not an entity instance.

        Notes
        -----
        If an entity already has a different coordinate system assigned, the assignment is
        overwritten and a warning is logged.
        """
        if not is_exact_instance(coordinate_system, CoordinateSystem):
            raise Flow360ValueError(
                f"`coordinate_system` must be a CoordinateSystem. Received: {type(coordinate_system).__name__}."
            )

        if isinstance(entities, list):
            normalized_entities = entities
        else:
            normalized_entities = [entities]

        for entity in normalized_entities:
            if not isinstance(entity, EntityBase):
                raise Flow360ValueError(
                    f"Only entities can be assigned a coordinate system. Received: {type(entity).__name__}."
                )
            if entity.private_attribute_id is None:
                raise Flow360ValueError(
                    f"Entity '{entity.name}' ({type(entity).__name__}) is not supported "
                    f"for coordinate system assignment."
                )

        self._state._register_coordinate_system(coordinate_system=coordinate_system, parent_id=None)

        for entity in normalized_entities:
            entity_key = self._state._entity_key(entity)
            previous = self._state._entity_key_to_coordinate_system_id.get(entity_key)
            if previous is not None and previous != coordinate_system.private_attribute_id:
                log.warning(
                    "Entity '%s' already had a coordinate system '%s'; overwriting to '%s'.",
                    entity.name,
                    previous,
                    coordinate_system.private_attribute_id,
                )
            self._state._entity_key_to_coordinate_system_id[entity_key] = (
                coordinate_system.private_attribute_id
            )

    def clear_assignment(self, *, entity: EntityBase) -> None:
        """
        Remove any coordinate system assignment for the given entity.

        Parameters
        ----------
        entity : EntityBase
            The entity whose coordinate system assignment should be cleared.
        """
        self._state._entity_key_to_coordinate_system_id.pop(self._state._entity_key(entity), None)

    def get_by_name(self, name: str) -> CoordinateSystem:
        """
        Retrieve a registered coordinate system by name.

        Parameters
        ----------
        name : str
            Coordinate system name.

        Returns
        -------
        CoordinateSystem
            The matching coordinate system.

        Raises
        ------
        Flow360ValueError
            If no coordinate system with the given name exists in this draft.
        """
        return self._state.get_by_name(name)

    @classmethod
    def _from_status(
        cls,
        *,
        status: Optional[CoordinateSystemStatus],
        entity_registry: Optional[EntityRegistry] = None,
    ) -> CoordinateSystemManager:
        """
        Restore manager from a status snapshot.

        Parameters
        ----------
        status : CoordinateSystemStatus, optional
            The status to restore from.
        entity_registry : EntityRegistry, optional
            Entity registry containing entities to validate against. If None, no validation is performed.
        Returns
        -------
        CoordinateSystemManager
            The restored manager.
        """
        return cls(
            state=CoordinateSystemState._from_status(
                status=status,
                entity_registry=entity_registry,
            )
        )
