"""Coordinate system management for DraftContext."""

from __future__ import annotations

import collections
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pydantic as pd

from flow360.component.simulation.entity_operation import (
    CoordinateSystem,
    _compose_transformation_matrices,
)
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.utils import is_exact_instance
from flow360.exceptions import Flow360RuntimeError
from flow360.log import log


class CoordinateSystemParent(Flow360BaseModel):
    """Parent relationship for a coordinate system."""

    type_name: Literal["CoordinateSystemParent"] = pd.Field("CoordinateSystemParent", frozen=True)
    coordinate_system_id: str
    parent_id: Optional[str] = pd.Field(None)


class CoordinateSystemEntityRef(Flow360BaseModel):
    """Entity reference used in assignment serialization."""

    type_name: Literal["CoordinateSystemEntityRef"] = pd.Field(
        "CoordinateSystemEntityRef", frozen=True
    )
    entity_type: str
    entity_id: str


class CoordinateSystemAssignmentGroup(Flow360BaseModel):
    """Grouped entity assignments for a coordinate system."""

    type_name: Literal["CoordinateSystemAssignmentGroup"] = pd.Field(
        "CoordinateSystemAssignmentGroup", frozen=True
    )
    coordinate_system_id: str
    entities: List[CoordinateSystemEntityRef]


class CoordinateSystemStatus(Flow360BaseModel):
    """Serializable snapshot for front end/asset cache."""

    type_name: Literal["CoordinateSystemStatus"] = pd.Field("CoordinateSystemStatus", frozen=True)
    coordinate_systems: List[CoordinateSystem]
    parents: List[CoordinateSystemParent]
    assignments: List[CoordinateSystemAssignmentGroup]

    @pd.model_validator(mode="after")
    def _validate_unique_coordinate_system_ids_and_names(self):
        """Validate that all coordinate system IDs and names are unique."""
        seen_ids = set()
        seen_names = set()
        for cs in self.coordinate_systems:
            # Check IDs first to match the order of validation in _from_status
            if cs.private_attribute_id in seen_ids:
                raise ValueError(
                    f"[Internal] Duplicate coordinate system id '{cs.private_attribute_id}' in status."
                )
            if cs.name in seen_names:
                raise ValueError(
                    f"[Internal] Duplicate coordinate system name '{cs.name}' in status."
                )
            seen_ids.add(cs.private_attribute_id)
            seen_names.add(cs.name)
        return self


class CoordinateSystemManager:
    """Encapsulates coordinate system registry, hierarchy, and entity assignments."""

    __slots__ = (
        "_coordinate_systems",
        "_coordinate_system_parents",
        "_entity_key_to_coordinate_system_id",
    )

    def __init__(self) -> None:
        self._coordinate_systems: list[CoordinateSystem] = []
        self._coordinate_system_parents: dict[str, Optional[str]] = {}
        self._entity_key_to_coordinate_system_id: dict[Tuple[str, str], str] = {}

    @property
    def _known_ids(self) -> set[str]:
        """Return set of registered coordinate system IDs for O(1) lookups."""
        return {cs.private_attribute_id for cs in self._coordinate_systems}

    def _register_coordinate_system(
        self, *, coordinate_system: CoordinateSystem, parent_id: str | None
    ) -> None:
        """Internal helper to register a coordinate system without graph validation."""
        if coordinate_system.private_attribute_id in self._known_ids:
            return  # Already registered, skip
        if any(existing.name == coordinate_system.name for existing in self._coordinate_systems):
            raise Flow360RuntimeError(
                f"Coordinate system name '{coordinate_system.name}' already registered."
            )
        self._coordinate_systems.append(coordinate_system)
        self._coordinate_system_parents[coordinate_system.private_attribute_id] = parent_id

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

        # Auto-register parent as root if not already registered
        if parent is not None:
            self._register_coordinate_system(coordinate_system=parent, parent_id=None)

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
        self, *, coordinate_system: CoordinateSystem, parent: Optional[CoordinateSystem]
    ) -> None:
        """Update parent of a registered coordinate system."""
        if coordinate_system not in self._coordinate_systems:
            raise Flow360RuntimeError("Coordinate system must be part of the draft to be updated.")
        # Auto-register parent as root if not already registered
        if parent is not None:
            self._register_coordinate_system(coordinate_system=parent, parent_id=None)

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
        self._entity_key_to_coordinate_system_id = {
            entity_id: assigned_id
            for entity_id, assigned_id in self._entity_key_to_coordinate_system_id.items()
            if assigned_id != cs_id
        }

    # endregion ------------------------------------------------------------------------------------

    def _validate_coordinate_system_graph(self) -> None:
        """Validate parent references exist and detect cycles using Kahn's algorithm."""
        id_to_cs: Dict[str, CoordinateSystem] = {}
        for cs in self._coordinate_systems:
            if cs.private_attribute_id in id_to_cs:
                raise Flow360RuntimeError(
                    f"Duplicate coordinate system id '{cs.private_attribute_id}' detected."
                )
            if cs.name in (existing.name for existing in id_to_cs.values()):
                raise Flow360RuntimeError(f"Coordinate system name '{cs.name}' already registered.")
            id_to_cs[cs.private_attribute_id] = cs

        # Validate all parent references exist
        for cs_id, cs in id_to_cs.items():
            parent_id = self._coordinate_system_parents.get(cs_id)
            if parent_id is not None and parent_id not in id_to_cs:
                raise Flow360RuntimeError(
                    f"Parent coordinate system '{parent_id}' not found for '{cs.name}'."
                )

        # Kahn's algorithm for cycle detection
        in_degree = {cs_id: 0 for cs_id in id_to_cs}
        for cs_id in id_to_cs:
            parent_id = self._coordinate_system_parents.get(cs_id)
            if parent_id is not None:
                in_degree[cs_id] += 1

        queue = collections.deque([cs_id for cs_id, degree in in_degree.items() if degree == 0])
        processed = 0

        while queue:
            current = queue.popleft()
            processed += 1
            for cs_id, parent_id in self._coordinate_system_parents.items():
                if parent_id == current:
                    in_degree[cs_id] -= 1
                    if in_degree[cs_id] == 0:
                        queue.append(cs_id)

        if processed != len(id_to_cs):
            cycle_nodes = [cs_id for cs_id, degree in in_degree.items() if degree > 0]
            raise Flow360RuntimeError(
                f"Cycle detected in coordinate system inheritance among: {sorted(cycle_nodes)}"
            )

    def _get_coordinate_system_matrix(self, *, coordinate_system: CoordinateSystem) -> np.ndarray:
        """Return the composed matrix for a registered coordinate system (parents applied)."""
        if coordinate_system not in self._coordinate_systems:
            raise Flow360RuntimeError("Coordinate system must be registered to compute its matrix.")

        cs_id = coordinate_system.private_attribute_id
        combined_matrix = coordinate_system._get_local_matrix()  # pylint:disable=protected-access

        # Graph is validated, parent guaranteed to exist and no cycles
        parent_id = self._coordinate_system_parents.get(cs_id)
        while parent_id is not None:
            parent = self._get_coordinate_system_by_id(parent_id)
            combined_matrix = _compose_transformation_matrices(
                parent=parent._get_local_matrix(),  # pylint:disable=protected-access
                child=combined_matrix,
            )
            parent_id = self._coordinate_system_parents.get(parent_id)

        return combined_matrix

    # --------------------------------------------------------------------
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

    @staticmethod
    def _entity_key(entity: EntityBase) -> tuple[str, str]:
        return (entity.private_attribute_entity_type_name, entity.private_attribute_id)

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

        self._register_coordinate_system(coordinate_system=coordinate_system, parent_id=None)

        for entity in normalized_entities:
            entity_key = self._entity_key(entity)
            previous = self._entity_key_to_coordinate_system_id.get(entity_key)
            if previous is not None and previous != coordinate_system.private_attribute_id:
                log.warning(
                    "Entity '%s' already had a coordinate system '%s'; overwriting to '%s'.",
                    entity.name,
                    previous,
                    coordinate_system.private_attribute_id,
                )
            self._entity_key_to_coordinate_system_id[entity_key] = (
                coordinate_system.private_attribute_id
            )

    def clear_assignment(self, *, entity: EntityBase) -> None:
        """Remove any coordinate system assignment for the given entity."""
        self._entity_key_to_coordinate_system_id.pop(self._entity_key(entity), None)

    def _get_coordinate_system_for_entity(self, *, entity: EntityBase) -> CoordinateSystem | None:
        """Return the coordinate system assigned to the entity, if any."""
        cs_id = self._entity_key_to_coordinate_system_id.get(self._entity_key(entity))
        if cs_id is None:
            return None
        cs = self._get_coordinate_system_by_id(cs_id)
        if cs is None:
            raise Flow360RuntimeError(
                f"Coordinate system id '{cs_id}' assigned to entity '{entity.name}' is not registered."
            )
        return cs

    def _get_matrix_for_entity(self, *, entity: EntityBase) -> Optional[np.ndarray]:
        """Return the composed 3x4 transformation matrix for an entity, if assigned."""
        cs = self._get_coordinate_system_for_entity(entity=entity)
        if cs is None:
            return None
        return self._get_coordinate_system_matrix(coordinate_system=cs)

    def _get_matrix_for_entity_key(
        self, *, entity_type: str, entity_id: str
    ) -> Optional[np.ndarray]:
        """Return the composed 3x4 matrix for an entity reference, if assigned."""
        cs_id = self._entity_key_to_coordinate_system_id.get((entity_type, entity_id))
        if cs_id is None:
            return None
        cs = self._get_coordinate_system_by_id(cs_id)
        if cs is None:
            raise Flow360RuntimeError(
                f"Coordinate system id '{cs_id}' assigned to entity '{entity_type}:{entity_id}' is not registered."
            )
        return self._get_coordinate_system_matrix(coordinate_system=cs)

    # Serialization ----------------------------------------------------------------
    def _to_status(self) -> CoordinateSystemStatus:
        """Build a serializable status snapshot.

        Returns
        -------
        CoordinateSystemStatus
            The serialized status.
        """
        parents = [
            CoordinateSystemParent(coordinate_system_id=cs_id, parent_id=parent_id)
            for cs_id, parent_id in self._coordinate_system_parents.items()
        ]

        grouped: Dict[str, List[CoordinateSystemEntityRef]] = {}
        for (entity_type, entity_id), cs_id in self._entity_key_to_coordinate_system_id.items():
            grouped.setdefault(cs_id, []).append(
                CoordinateSystemEntityRef(entity_type=entity_type, entity_id=entity_id)
            )

        assignments = [
            CoordinateSystemAssignmentGroup(coordinate_system_id=cs_id, entities=entities)
            for cs_id, entities in grouped.items()
        ]
        return CoordinateSystemStatus(
            coordinate_systems=self._coordinate_systems,
            parents=parents,
            assignments=assignments,
        )

    @classmethod
    def _from_status(
        cls,
        *,
        status: Optional[CoordinateSystemStatus],
        entity_registry: Optional[EntityRegistry] = None,
    ) -> CoordinateSystemManager:
        """Restore manager from a status snapshot.

        Parameters
        ----------
        status : CoordinateSystemStatus | None
            The status to restore from.
        entity_registry : EntityRegistry | None
            Entity registry containing entities to validate against. If None, no validation is performed.
        Returns
        -------
        CoordinateSystemManager
            The restored manager.
        """
        mgr = cls()
        if status is None:
            return mgr

        # Build set of IDs for validation of parent/assignment references.
        existing_ids: set[str] = set()
        for cs in status.coordinate_systems:
            mgr._coordinate_systems.append(cs)
            existing_ids.add(cs.private_attribute_id)

        for parent in status.parents:
            if parent.coordinate_system_id not in existing_ids:
                raise Flow360RuntimeError(
                    f"Parent record references unknown coordinate system '{parent.coordinate_system_id}'."
                )
            if parent.parent_id is not None and parent.parent_id not in existing_ids:
                raise Flow360RuntimeError(
                    f"Parent coordinate system '{parent.parent_id}' not found for '{parent.coordinate_system_id}'."
                )
            mgr._coordinate_system_parents[parent.coordinate_system_id] = parent.parent_id

        seen_entity_keys: set[Tuple[str, str]] = set()
        for assignment in status.assignments:
            if assignment.coordinate_system_id not in existing_ids:
                raise Flow360RuntimeError(
                    f"Assignment references unknown coordinate system '{assignment.coordinate_system_id}'."
                )
            for entity in assignment.entities:
                key = (entity.entity_type, entity.entity_id)
                # Sanitize invalid assignments due to entity not being in scope anymore.
                if (
                    entity_registry  # Fast lane: entity_registry None indicates that no validation is needed
                    and (
                        entity_registry.find_by_type_name_and_id(
                            entity_type=entity.entity_type, entity_id=entity.entity_id
                        )
                        is None
                    )
                ):
                    log.warning(
                        "Entity '%s:%s' assigned to coordinate system '%s' is not in the draft registry; "
                        "skipping this coordinate system assignment.",
                        entity.entity_type,
                        entity.entity_id,
                        assignment.coordinate_system_id,
                    )
                    continue
                if key in seen_entity_keys:
                    raise Flow360RuntimeError(
                        f"Duplicate entity assignment for entity '{entity.entity_type}:{entity.entity_id}'."
                    )
                seen_entity_keys.add(key)
                mgr._entity_key_to_coordinate_system_id[key] = assignment.coordinate_system_id

        mgr._validate_coordinate_system_graph()
        return mgr
