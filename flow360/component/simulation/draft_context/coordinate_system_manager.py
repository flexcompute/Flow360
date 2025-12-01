"""Coordinate system management for DraftContext."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from flow360.component.simulation.entity_operation import (
    CoordinateSystem,
    _compose_transformation_matrices,
)
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.utils import is_exact_instance
from flow360.exceptions import Flow360RuntimeError
from flow360.log import log


class CoordinateSystemManager:
    """Encapsulates coordinate system registry, hierarchy, and entity assignments.

    - _coordinate_systems: List of coordinate systems tracked in the draft.
    - _coordinate_system_parents: Mapping from coordinate system id to parent coordinate system id.
    - _entity_id_to_coordinate_system_id: Mapping from entity id to coordinate system id.
    - _entity_registry: Supplied entity registry to track the entities in the draft.
    """

    __slots__ = (
        "_coordinate_systems",
        "_coordinate_system_parents",
        "_entity_id_to_coordinate_system_id",
        "_entity_registry",
    )

    def __init__(self, *, entity_registry: EntityRegistry) -> None:
        self._coordinate_systems: list[CoordinateSystem] = []
        self._coordinate_system_parents: dict[str, Optional[str]] = {}
        self._entity_id_to_coordinate_system_id: dict[str, str] = {}
        self._entity_registry = entity_registry

    # region Registration and hierarchy -------------------------------------------------
    def add(
        self, *, coordinate_system: CoordinateSystem, parent: CoordinateSystem | None = None
    ) -> CoordinateSystem:
        """Register a coordinate system and optional parent."""
        if not is_exact_instance(coordinate_system, CoordinateSystem):
            raise Flow360RuntimeError(
                f"coordinate_system must be a CoordinateSystem. Received: {type(coordinate_system).__name__}."
            )
        cs = coordinate_system

        if parent is not None and parent not in self._coordinate_systems:
            raise Flow360RuntimeError(
                "Parent coordinate system must be registered in the draft before being referenced."
            )

        if any(
            existing.private_attribute_id == cs.private_attribute_id
            for existing in self._coordinate_systems
        ):
            raise Flow360RuntimeError(
                f"Coordinate system id '{cs.private_attribute_id}' already registered."
            )
        if any(existing.name == cs.name for existing in self._coordinate_systems):
            raise Flow360RuntimeError(f"Coordinate system name '{cs.name}' already registered.")

        self._coordinate_systems.append(cs)
        self._coordinate_system_parents[cs.private_attribute_id] = (
            parent.private_attribute_id if parent is not None else None
        )
        self._validate_coordinate_system_graph()
        return cs

    def update_parent(
        self, *, coordinate_system: CoordinateSystem, parent: CoordinateSystem | None
    ) -> None:
        """Update parent of a registered coordinate system."""
        if coordinate_system not in self._coordinate_systems:
            raise Flow360RuntimeError("Coordinate system must be part of the draft to be updated.")
        if parent is not None and parent not in self._coordinate_systems:
            raise Flow360RuntimeError(
                "Parent coordinate system must be registered in the draft before being referenced."
            )

        cs_id = coordinate_system.private_attribute_id
        original_parent = self._coordinate_system_parents.get(cs_id)
        self._coordinate_system_parents[cs_id] = parent.private_attribute_id if parent else None

        try:
            self._validate_coordinate_system_graph()
        except Exception:
            self._coordinate_system_parents[cs_id] = original_parent
            raise

    def remove(self, *, coordinate_system: CoordinateSystem) -> None:
        """Remove a coordinate system if no dependents reference it."""
        if coordinate_system not in self._coordinate_systems:
            raise Flow360RuntimeError("Coordinate system is not registered in this draft.")

        cs_id = coordinate_system.private_attribute_id
        dependents = [
            child_id
            for child_id, parent_id in self._coordinate_system_parents.items()
            if parent_id == cs_id
        ]
        if dependents:
            names = ", ".join(
                cs.name for cs in self._coordinate_systems if cs.private_attribute_id in dependents
            )
            raise Flow360RuntimeError(
                f"Cannot remove coordinate system '{coordinate_system.name}' because dependents exist: {names}"
            )

        self._coordinate_systems = [
            cs for cs in self._coordinate_systems if cs is not coordinate_system
        ]
        self._coordinate_system_parents.pop(cs_id, None)
        # Drop assignments referencing this coordinate system.
        self._entity_id_to_coordinate_system_id = {
            entity_id: assigned_id
            for entity_id, assigned_id in self._entity_id_to_coordinate_system_id.items()
            if assigned_id != cs_id
        }

    # endregion ------------------------------------------------------------------------------------

    def _validate_coordinate_system_graph(self) -> None:
        """Validate parent references and detect cycles."""
        id_to_cs: Dict[str, CoordinateSystem] = {}
        for cs in self._coordinate_systems:
            if cs.private_attribute_id in id_to_cs:
                raise Flow360RuntimeError(
                    f"Duplicate coordinate system id '{cs.private_attribute_id}' detected."
                )
            if cs.name in (existing.name for existing in id_to_cs.values()):
                raise Flow360RuntimeError(f"Coordinate system name '{cs.name}' already registered.")
            id_to_cs[cs.private_attribute_id] = cs

        for cs_id, cs in id_to_cs.items():
            parent_id = self._coordinate_system_parents.get(cs_id)
            if parent_id is None:
                continue
            if parent_id not in id_to_cs:
                raise Flow360RuntimeError(
                    f"Parent coordinate system '{parent_id}' not found for '{cs.name}'."
                )

        for cs_id, cs in id_to_cs.items():
            visited: set[str] = set()
            parent_id = self._coordinate_system_parents.get(cs_id)
            while parent_id is not None:
                if parent_id in visited:
                    raise Flow360RuntimeError("Cycle detected in coordinate system inheritance")
                visited.add(parent_id)
                if parent_id not in id_to_cs:
                    raise Flow360RuntimeError(
                        f"Parent coordinate system '{parent_id}' not found for '{cs.name}'."
                    )
                parent_id = self._coordinate_system_parents.get(parent_id)

    # Lookup --------------------------------------------------------------------

    def get_by_name(self, name: str) -> CoordinateSystem:
        """Retrieve a coordinate system by name."""
        for cs in self._coordinate_systems:
            if cs.name == name:
                return cs
        raise Flow360RuntimeError(f"Coordinate system '{name}' not found in the draft.")

    def _get_coordinate_system_by_id(self, cs_id: str) -> CoordinateSystem | None:
        for cs in self._coordinate_systems:
            if cs.private_attribute_id == cs_id:
                return cs
        return None

    # Assignment ----------------------------------------------------------------
    def assign(
        self,
        *,
        entities: List[EntityBase] | EntityBase,
        coordinate_system: CoordinateSystem,
    ) -> None:
        """Assign a coordinate system to one or more draft entities."""
        if not is_exact_instance(coordinate_system, CoordinateSystem):
            raise Flow360RuntimeError(
                f"`coordinate_system` must be a CoordinateSystem. Received: {type(coordinate_system).__name__}."
            )

        # Normalize to list for uniform validation.
        if isinstance(entities, list):
            normalized_entities = entities
        else:
            normalized_entities = [entities]

        for entity in normalized_entities:
            if not isinstance(entity, EntityBase):
                raise Flow360RuntimeError(
                    f"Only entities can be assigned a coordinate system. Received: {type(entity).__name__}."
                )
            if not self._entity_registry.contains(entity):
                raise Flow360RuntimeError(
                    f"Entity '{entity.name}' is not part of the draft registry and cannot be assigned."
                )

        if coordinate_system not in self._coordinate_systems:
            self.add(coordinate_system=coordinate_system, parent=None)

        for entity in normalized_entities:
            entity_id = entity.private_attribute_id
            previous = self._entity_id_to_coordinate_system_id.get(entity_id)
            if previous is not None and previous != coordinate_system.private_attribute_id:
                log.warning(
                    "Entity '%s' already had a coordinate system '%s'; overwriting to '%s'.",
                    entity.name,
                    previous,
                    coordinate_system.private_attribute_id,
                )
            self._entity_id_to_coordinate_system_id[entity_id] = (
                coordinate_system.private_attribute_id
            )

    def clear_assignment(self, *, entity: EntityBase) -> None:
        """Remove any coordinate system assignment for the given entity."""
        if not self._entity_registry.contains(entity):
            raise Flow360RuntimeError(f"Entity '{entity.name}' is not part of the draft registry.")
        self._entity_id_to_coordinate_system_id.pop(entity.private_attribute_id, None)

    def get_for_entity(self, *, entity: EntityBase) -> CoordinateSystem | None:
        """Return the coordinate system assigned to the entity, if any."""
        if not self._entity_registry.contains(entity):
            raise Flow360RuntimeError(f"Entity '{entity.name}' is not part of the draft registry.")
        cs_id = self._entity_id_to_coordinate_system_id.get(entity.private_attribute_id)
        if cs_id is None:
            return None
        cs = self._get_coordinate_system_by_id(cs_id)
        if cs is None:
            raise Flow360RuntimeError(
                f"Coordinate system id '{cs_id}' assigned to entity '{entity.name}' is not registered."
            )
        return cs

    def get_coordinate_system_matrix(self, *, coordinate_system: CoordinateSystem) -> np.ndarray:
        """Return the composed matrix for a registered coordinate system (parents applied)."""
        if coordinate_system not in self._coordinate_systems:
            raise Flow360RuntimeError("Coordinate system must be registered to compute its matrix.")

        cs_id = coordinate_system.private_attribute_id
        combined_matrix = coordinate_system._get_local_matrix()  # pylint:disable=protected-access

        visited: set[str] = set()
        parent_id = self._coordinate_system_parents.get(cs_id)
        while parent_id is not None:
            if parent_id in visited:
                raise Flow360RuntimeError("Cycle detected in coordinate system inheritance")
            visited.add(parent_id)

            parent = self._get_coordinate_system_by_id(parent_id)
            if parent is None:
                raise Flow360RuntimeError(
                    f"Parent coordinate system '{parent_id}' not found for '{coordinate_system.name}'."
                )
            combined_matrix = _compose_transformation_matrices(
                parent=parent._get_local_matrix(),  # pylint:disable=protected-access
                child=combined_matrix,
            )
            parent_id = self._coordinate_system_parents.get(parent_id)

        return combined_matrix
