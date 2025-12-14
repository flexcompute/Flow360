"""Coordinate system management for DraftContext."""

from __future__ import annotations

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
        self._entity_key_to_coordinate_system_id = {
            entity_id: assigned_id
            for entity_id, assigned_id in self._entity_key_to_coordinate_system_id.items()
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

    def _get_coordinate_system_matrix(self, *, coordinate_system: CoordinateSystem) -> np.ndarray:
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

        if coordinate_system not in self._coordinate_systems:
            self.add(coordinate_system=coordinate_system, parent=None)

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
    def _to_status(self, *, entity_registry: EntityRegistry) -> CoordinateSystemStatus:
        """Build a serializable status snapshot.

        Parameters
        ----------
        entity_registry : EntityRegistry
            The entity registry to validate entity references against.

        Returns
        -------
        CoordinateSystemStatus
            The serialized status.

        Raises
        ------
        Flow360RuntimeError
            If any assigned entity is not in the registry.
        """
        parents = [
            CoordinateSystemParent(coordinate_system_id=cs_id, parent_id=parent_id)
            for cs_id, parent_id in self._coordinate_system_parents.items()
        ]

        # Validate entity existence before serialization.
        # Build a set of all existing entity keys in the registry for efficient lookup.
        existing_entity_keys = set()
        for entity_list in entity_registry.internal_registry.values():
            for entity in entity_list:
                existing_entity_keys.add(
                    (entity.private_attribute_entity_type_name, entity.private_attribute_id)
                )

        grouped: Dict[str, List[CoordinateSystemEntityRef]] = {}
        for (entity_type, entity_id), cs_id in self._entity_key_to_coordinate_system_id.items():
            # Check if entity exists in registry.
            # A missing entity is possible if the entity was removed from the draft context.
            # For example by deleting a geometry body group.
            entity_key = (entity_type, entity_id)
            if entity_key not in existing_entity_keys:
                log.debug(
                    "Entity '%s:%s' assigned to coordinate system '%s' is not in the draft registry; "
                    "skipping this assignment.",
                    entity_type,
                    entity_id,
                    cs_id,
                )
                continue

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
    def _from_status(cls, *, status: CoordinateSystemStatus | None) -> "CoordinateSystemManager":
        """Restore manager from a status snapshot.

        Parameters
        ----------
        status : CoordinateSystemStatus | None
            The status to restore from.

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
                if key in seen_entity_keys:
                    raise Flow360RuntimeError(
                        f"Duplicate entity assignment for entity '{entity.entity_type}:{entity.entity_id}'."
                    )
                seen_entity_keys.add(key)
                mgr._entity_key_to_coordinate_system_id[key] = assignment.coordinate_system_id

        mgr._validate_coordinate_system_graph()
        return mgr
