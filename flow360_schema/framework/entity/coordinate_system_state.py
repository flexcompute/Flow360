"""Coordinate system graph/state helpers shared by schema and client."""

from __future__ import annotations

import collections
import logging
from typing import cast

import numpy as np
import numpy.typing as npt

from flow360_schema.exceptions import Flow360ValueError
from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_operation import (
    CoordinateSystem,
    _compose_transformation_matrices,
)
from flow360_schema.framework.entity.entity_registry import EntityRegistry
from flow360_schema.models.asset_cache import (
    CoordinateSystemAssignmentGroup,
    CoordinateSystemEntityRef,
    CoordinateSystemParent,
    CoordinateSystemStatus,
)

logger = logging.getLogger(__name__)
Float64Matrix = npt.NDArray[np.float64]


class CoordinateSystemState:
    """Read-only coordinate-system graph and assignment state."""

    __slots__ = (
        "_coordinate_systems",
        "_coordinate_system_parents",
        "_entity_key_to_coordinate_system_id",
    )

    def __init__(self) -> None:
        self._coordinate_systems: list[CoordinateSystem] = []
        self._coordinate_system_parents: dict[str, str | None] = {}
        self._entity_key_to_coordinate_system_id: dict[tuple[str, str], str] = {}

    # Public read API ----------------------------------------------------
    def get_by_name(self, name: str) -> CoordinateSystem:
        """Retrieve a registered coordinate system by name."""
        for coordinate_system in self._coordinate_systems:
            if coordinate_system.name == name:
                return coordinate_system
        raise Flow360ValueError(f"Coordinate system '{name}' not found in the draft.")

    # Status bridge API --------------------------------------------------
    def _to_status(self) -> CoordinateSystemStatus:
        """Build a serializable status snapshot."""
        parents = [
            CoordinateSystemParent(
                type_name="CoordinateSystemParent",
                coordinate_system_id=coordinate_system_id,
                parent_id=parent_id,
            )
            for coordinate_system_id, parent_id in self._coordinate_system_parents.items()
        ]

        grouped: dict[str, list[CoordinateSystemEntityRef]] = {}
        for (entity_type, entity_id), coordinate_system_id in self._entity_key_to_coordinate_system_id.items():
            grouped.setdefault(coordinate_system_id, []).append(
                CoordinateSystemEntityRef(
                    type_name="CoordinateSystemEntityRef",
                    entity_type=entity_type,
                    entity_id=entity_id,
                )
            )

        assignments = [
            CoordinateSystemAssignmentGroup(
                type_name="CoordinateSystemAssignmentGroup",
                coordinate_system_id=coordinate_system_id,
                entities=entities,
            )
            for coordinate_system_id, entities in grouped.items()
        ]
        return CoordinateSystemStatus(
            type_name="CoordinateSystemStatus",
            coordinate_systems=self._coordinate_systems,
            parents=parents,
            assignments=assignments,
        )

    @classmethod
    def _from_status(
        cls,
        *,
        status: CoordinateSystemStatus | None,
        entity_registry: EntityRegistry | None = None,
    ) -> CoordinateSystemState:
        """Restore state from a status snapshot."""
        state = cls()
        if status is None:
            return state

        existing_ids: set[str] = set()
        for coordinate_system in status.coordinate_systems:
            state._coordinate_systems.append(coordinate_system)
            existing_ids.add(coordinate_system.private_attribute_id)

        for parent in status.parents:
            if parent.coordinate_system_id not in existing_ids:
                raise Flow360ValueError(
                    f"Parent record references unknown coordinate system '{parent.coordinate_system_id}'."
                )
            if parent.parent_id is not None and parent.parent_id not in existing_ids:
                raise Flow360ValueError(
                    f"Parent coordinate system '{parent.parent_id}' not found for '{parent.coordinate_system_id}'."
                )
            state._coordinate_system_parents[parent.coordinate_system_id] = parent.parent_id

        seen_entity_keys: set[tuple[str, str]] = set()
        for assignment in status.assignments:
            if assignment.coordinate_system_id not in existing_ids:
                raise Flow360ValueError(
                    f"Assignment references unknown coordinate system '{assignment.coordinate_system_id}'."
                )
            for entity in assignment.entities:
                entity_key = (entity.entity_type, entity.entity_id)
                if entity_registry and (
                    entity_registry.find_by_type_name_and_id(
                        entity_type=entity.entity_type,
                        entity_id=entity.entity_id,
                    )
                    is None
                ):
                    logger.warning(
                        "Entity '%s:%s' assigned to coordinate system '%s' is not in the draft registry; "
                        "skipping this coordinate system assignment.",
                        entity.entity_type,
                        entity.entity_id,
                        assignment.coordinate_system_id,
                    )
                    continue
                if entity_key in seen_entity_keys:
                    raise Flow360ValueError(
                        f"Duplicate entity assignment for entity '{entity.entity_type}:{entity.entity_id}'."
                    )
                seen_entity_keys.add(entity_key)
                state._entity_key_to_coordinate_system_id[entity_key] = assignment.coordinate_system_id

        state._validate_coordinate_system_graph()
        return state

    # Internal graph helpers ---------------------------------------------
    @property
    def _known_ids(self) -> set[str]:
        """Return set of registered coordinate system IDs for O(1) lookups."""
        return {coordinate_system.private_attribute_id for coordinate_system in self._coordinate_systems}

    def _contains(self, coordinate_system: CoordinateSystem) -> bool:
        return coordinate_system.private_attribute_id in self._known_ids

    def _register_coordinate_system(self, *, coordinate_system: CoordinateSystem, parent_id: str | None) -> None:
        """Register a coordinate system without validating the full graph."""
        if self._contains(coordinate_system):
            return
        if any(existing.name == coordinate_system.name for existing in self._coordinate_systems):
            raise Flow360ValueError(f"Coordinate system name '{coordinate_system.name}' already registered.")
        self._coordinate_systems.append(coordinate_system)
        self._coordinate_system_parents[coordinate_system.private_attribute_id] = parent_id

    def _validate_coordinate_system_graph(self) -> None:
        """Validate parent references exist and detect cycles using Kahn's algorithm."""
        id_to_coordinate_system: dict[str, CoordinateSystem] = {}
        for coordinate_system in self._coordinate_systems:
            if coordinate_system.private_attribute_id in id_to_coordinate_system:
                raise Flow360ValueError(
                    f"Duplicate coordinate system id '{coordinate_system.private_attribute_id}' detected."
                )
            if coordinate_system.name in (existing.name for existing in id_to_coordinate_system.values()):
                raise Flow360ValueError(f"Coordinate system name '{coordinate_system.name}' already registered.")
            id_to_coordinate_system[coordinate_system.private_attribute_id] = coordinate_system

        for coordinate_system_id, coordinate_system in id_to_coordinate_system.items():
            parent_id = self._coordinate_system_parents.get(coordinate_system_id)
            if parent_id is not None and parent_id not in id_to_coordinate_system:
                raise Flow360ValueError(
                    f"Parent coordinate system '{parent_id}' not found for '{coordinate_system.name}'."
                )

        in_degree = {coordinate_system_id: 0 for coordinate_system_id in id_to_coordinate_system}
        for coordinate_system_id in id_to_coordinate_system:
            parent_id = self._coordinate_system_parents.get(coordinate_system_id)
            if parent_id is not None:
                in_degree[coordinate_system_id] += 1

        queue = collections.deque(
            coordinate_system_id for coordinate_system_id, degree in in_degree.items() if degree == 0
        )
        processed = 0

        while queue:
            current = queue.popleft()
            processed += 1
            for coordinate_system_id, parent_id in self._coordinate_system_parents.items():
                if parent_id == current:
                    in_degree[coordinate_system_id] -= 1
                    if in_degree[coordinate_system_id] == 0:
                        queue.append(coordinate_system_id)

        if processed != len(id_to_coordinate_system):
            cycle_nodes = sorted(
                coordinate_system_id for coordinate_system_id, degree in in_degree.items() if degree > 0
            )
            raise Flow360ValueError(f"Cycle detected in coordinate system inheritance among: {cycle_nodes}")

    def _get_coordinate_system_matrix(self, *, coordinate_system: CoordinateSystem) -> Float64Matrix:
        """Return the composed matrix for a registered coordinate system."""
        if not self._contains(coordinate_system):
            raise Flow360ValueError("Coordinate system must be registered to compute its matrix.")

        coordinate_system_id = coordinate_system.private_attribute_id
        combined_matrix = cast(Float64Matrix, coordinate_system._get_local_matrix())

        parent_id = self._coordinate_system_parents.get(coordinate_system_id)
        while parent_id is not None:
            parent = self._get_coordinate_system_by_id(parent_id)
            if parent is None:
                raise Flow360ValueError(
                    f"Parent coordinate system '{parent_id}' not found for '{coordinate_system.name}'."
                )
            combined_matrix = cast(
                Float64Matrix,
                _compose_transformation_matrices(
                    parent=parent._get_local_matrix(),
                    child=combined_matrix,
                ),
            )
            parent_id = self._coordinate_system_parents.get(parent_id)

        return combined_matrix

    def _get_coordinate_system_by_id(self, coordinate_system_id: str) -> CoordinateSystem | None:
        for coordinate_system in self._coordinate_systems:
            if coordinate_system.private_attribute_id == coordinate_system_id:
                return coordinate_system
        return None

    # Internal assignment helpers ----------------------------------------
    @staticmethod
    def _require_entity_id(entity: EntityBase) -> str:
        if entity.private_attribute_id is None:
            raise Flow360ValueError(
                f"Entity '{entity.name}' ({type(entity).__name__}) must have an id for coordinate-system assignment."
            )
        return entity.private_attribute_id

    @classmethod
    def _entity_key(cls, entity: EntityBase) -> tuple[str, str]:
        return (entity.private_attribute_entity_type_name, cls._require_entity_id(entity))

    def _get_coordinate_system_for_entity(self, *, entity: EntityBase) -> CoordinateSystem | None:
        """Return the coordinate system assigned to the entity, if any."""
        coordinate_system_id = self._entity_key_to_coordinate_system_id.get(self._entity_key(entity))
        if coordinate_system_id is None:
            return None
        coordinate_system = self._get_coordinate_system_by_id(coordinate_system_id)
        if coordinate_system is None:
            raise Flow360ValueError(
                f"Coordinate system id '{coordinate_system_id}' assigned to entity '{entity.name}' is not registered."
            )
        return coordinate_system

    def _get_matrix_for_entity(self, *, entity: EntityBase) -> Float64Matrix | None:
        """Return the composed transformation matrix for an entity, if assigned."""
        coordinate_system = self._get_coordinate_system_for_entity(entity=entity)
        if coordinate_system is None:
            return None
        return self._get_coordinate_system_matrix(coordinate_system=coordinate_system)

    def _get_matrix_for_entity_key(self, *, entity_type: str, entity_id: str) -> Float64Matrix | None:
        """Return the composed transformation matrix for an entity reference, if assigned."""
        coordinate_system_id = self._entity_key_to_coordinate_system_id.get((entity_type, entity_id))
        if coordinate_system_id is None:
            return None
        coordinate_system = self._get_coordinate_system_by_id(coordinate_system_id)
        if coordinate_system is None:
            raise Flow360ValueError(
                "Coordinate system id "
                f"'{coordinate_system_id}' assigned to entity "
                f"'{entity_type}:{entity_id}' is not registered."
            )
        return self._get_coordinate_system_matrix(coordinate_system=coordinate_system)
