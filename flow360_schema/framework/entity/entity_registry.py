"""Registry for managing and storing instances of various entity types."""

from collections.abc import Iterator
from typing import Any

import pydantic as pd

from flow360_schema.exceptions import Flow360ValueError
from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_utils import (
    compile_glob_cached,
    get_combined_subclasses,
    is_exact_instance,
    naming_pattern_handler,
)


class EntityRegistryView:
    """
    Type-filtered view over EntityRegistry with glob pattern support.

    Provides a simplified interface for accessing entities of **only** a specific type,
    supporting both direct name lookup and glob pattern matching.
    """

    def __init__(self, registry: "EntityRegistry", entity_type: type[EntityBase]) -> None:
        self._registry = registry
        self._entity_type = entity_type

    def __iter__(self) -> Iterator[EntityBase]:
        """Iterate over all entities of this type."""
        return iter(self._entities)

    def __len__(self) -> int:
        """Return the number of entities of this type."""
        return len(self._entities)

    @property
    def _entities(self) -> list[EntityBase]:
        """Get all entities of the target type (exact type match only)."""
        entities_of_type = self._registry.internal_registry.get(self._entity_type, [])
        return [item for item in entities_of_type if is_exact_instance(item, self._entity_type)]

    def __getitem__(self, key: str) -> EntityBase | list[EntityBase]:
        """
        Support syntax like `registry.view(Surface)['wing']` and glob patterns `registry.view(Surface)['wing*']`.

        Parameters:
            key (str): Entity name or glob pattern (e.g., 'wing', 'wing*', '*tail').

        Returns:
            EntityBase or list[EntityBase]: Single entity if exactly one match, otherwise list of matches.

        Raises:
            Flow360ValueError: If key is not a string.
            ValueError: If no entities match the pattern.
        """
        if not isinstance(key, str):
            raise Flow360ValueError(f"Entity naming pattern: {key} is not a string.")

        matcher = compile_glob_cached(pattern=key)
        matched = [entity for entity in self._entities if matcher.fullmatch(entity.name) is not None]

        if not matched:
            raise ValueError(f"No entity found in registry with given name/naming pattern: '{key}'.")
        if len(matched) == 1:
            return matched[0]
        return matched


class EntityRegistry(Flow360BaseModel):
    """
    A registry for managing references to instances of various entity types.

    This class provides methods to register entities, retrieve entities by their type,
    and find entities by name patterns using regular expressions.

    Attributes:
        internal_registry (Dict[type[EntityBase], List[EntityBase]]): A dictionary that maps entity
        types to lists of instances.

    #Known Issues:
    frozen=True do not stop the user from changing the internal_registry
    """

    internal_registry: dict[type[EntityBase], list[Any]] = pd.Field({})

    def fast_register(self, entity: EntityBase, known_frozen_hashes: set[str]) -> set[str]:
        """
        Registers an entity in the registry under its type. Suitable for registering a large number of entities.

        Parameters:
            entity (EntityBase): The entity instance to register.
            known_frozen_hashes (Optional[set[str]]): A set of hashes of frozen entities.
                This is used to speed up checking if the has is already in the registry by avoiding O(N^2) complexity.
                This can be provided when registering a large number of entities.

        Returns:
            known_frozen_hashes (set[str])
        """
        entity_type = type(entity)
        if entity_type not in self.internal_registry:
            self.internal_registry[entity_type] = []

        if entity._get_hash() in known_frozen_hashes:
            return known_frozen_hashes
        known_frozen_hashes.add(entity._get_hash())

        self.internal_registry[entity_type].append(entity)
        return known_frozen_hashes

    def register(self, entity: EntityBase) -> None:
        """
        Registers an entity in the registry under its type.

        Parameters:
            entity (EntityBase): The entity instance to register.
        """
        entity_type = type(entity)
        if entity_type not in self.internal_registry:
            self.internal_registry[entity_type] = []

        for existing_entity in self.internal_registry[entity_type]:
            if existing_entity._get_hash() == entity._get_hash():
                return

        self.internal_registry[entity_type].append(entity)

    def remove(self, entity: EntityBase) -> None:
        """Remove an entity from the registry."""
        entity_type = type(entity)
        if entity_type not in self.internal_registry or entity not in self.internal_registry[entity_type]:
            return
        self.internal_registry[entity_type].remove(entity)

    def view(self, entity_type: type[EntityBase]) -> EntityRegistryView:
        """
        Create a filtered view for a specific entity type with glob pattern support.

        Parameters:
            entity_type (type[EntityBase]): The entity type to filter by (exact type match).

        Returns:
            EntityRegistryView: A view providing filtered access to entities of the specified type.

        Example:
            >>> surfaces = registry.view(Surface)
            >>> wing = surfaces['wing']
            >>> tails = surfaces['*tail']
        """
        return EntityRegistryView(registry=self, entity_type=entity_type)

    def view_subclasses(self, parent_type: type[EntityBase]) -> list[EntityRegistryView]:
        """
        Create views for all subclasses of a parent entity type.

        Parameters:
            parent_type (type[EntityBase]): The parent entity type.

        Returns:
            list[EntityRegistryView]: A list of views, one for each subclass found in the registry.

        Example:
            >>> # Get views for all Surface subclasses
            >>> surface_views = registry.view_subclasses(Surface)
            >>> for view in surface_views:
            >>>     print(f"Found {len(view)} entities")
        """
        subclasses = get_combined_subclasses(parent_type)
        views = []
        for subclass in subclasses:
            if subclass in self.internal_registry:
                views.append(EntityRegistryView(registry=self, entity_type=subclass))
        return views

    def find_by_naming_pattern(
        self, pattern: str, enforce_output_as_list: bool = True, error_when_no_match: bool = False
    ) -> list[EntityBase]:
        """
        Finds all registered entities whose names match a given pattern.

        Parameters:
            pattern (str): A naming pattern, which can include '*' as a wildcard.

        Returns:
            List[EntityBase]: A list of entities whose names match the pattern.
        """
        matched_entities: list[EntityBase] = []
        matcher = naming_pattern_handler(pattern=pattern)
        for entity_list in self.internal_registry.values():
            matched_entities.extend(filter(lambda x: matcher.fullmatch(x.name) is not None, entity_list))

        if not matched_entities and error_when_no_match is True:
            raise ValueError(f"No entity found in registry with given name/naming pattern: '{pattern}'.")
        if enforce_output_as_list is False and len(matched_entities) == 1:
            return matched_entities[0]  # type: ignore[return-value]

        return matched_entities

    def __str__(self) -> str:
        """
        Returns a string representation of all registered entities, grouped by type.
        """
        index = 0
        result = "---- Content of the registry ----\n"
        for entity_type, entities in self.internal_registry.items():
            result += f"\n    Entities of type '{entity_type.__name__}':\n"
            for entity in entities:
                result += f"    - [{index:05d}]\n{entity}\n"
                index += 1
        result += "---- End of content ----"
        return result

    def clear(self, entity_type: None | type[EntityBase] = None) -> None:
        """
        Clears all entities from the registry.
        """
        if entity_type is not None:
            if entity_type in self.internal_registry:
                self.internal_registry[entity_type].clear()
        else:
            self.internal_registry.clear()

    def contains(self, entity: EntityBase) -> bool:
        """
        Returns True if the registry contains any entities, False otherwise.
        """
        entity_type = type(entity)
        if entity_type in self.internal_registry and entity in self.internal_registry[entity_type]:
            return True
        return False

    def entity_count(self) -> int:
        """Return total number of entities in the registry."""
        count = 0
        for list_of_entities in self.internal_registry.values():
            count += len(list_of_entities)
        return count

    def replace_existing_with(self, new_entity: EntityBase) -> None:
        """
        Replaces an entity in the registry with a new entity (searched by name across all types).

        This searches for an entity with the same name across all registered types.
        If found, removes the old entity and registers the new one.
        If not found, simply registers the new entity.

        Parameters:
            new_entity (EntityBase): The new entity to replace the existing entity with.
        """
        for entity_type, entity_list in self.internal_registry.items():
            for i, entity in enumerate(entity_list):
                if entity.name == new_entity.name:
                    self.internal_registry[entity_type].pop(i)
                    self.register(new_entity)
                    return

        self.register(new_entity)

    def find_by_asset_id(self, *, entity_id: str, entity_class: type[EntityBase]) -> EntityBase | None:
        """
        Find the entity with matching asset id and the same type as the input entity class.
        Return None if no such entity is found.
        """
        entities = self.view(entity_class)._entities
        matched_entities = [item for item in entities if item.private_attribute_id == entity_id]

        if len(matched_entities) > 1:
            raise ValueError(
                f"[INTERNAL] Multiple entities with the same asset id ({entity_id}) found." " Data is likely corrupted."
            )
        if not matched_entities:
            return None
        return matched_entities[0]

    @property
    def is_empty(self) -> bool:
        """Return True if the registry is empty, False otherwise."""
        return not self.internal_registry

    def find_by_type(self, entity_class: type[EntityBase]) -> list[EntityBase]:
        """Find all registered entities of a given type (including subclasses).

        Parameters:
            entity_class (type[EntityBase]): The entity class to search for.

        Returns:
            list[EntityBase]: All entities that are instances of the given class.
        """
        matched_entities = []
        for entity_type, entity_list in self.internal_registry.items():
            if issubclass(entity_type, entity_class):
                matched_entities.extend(entity_list)
        return matched_entities

    def find_by_type_name(self, type_name: str | list[str]) -> list[EntityBase]:
        """Find all registered entities with matching private_attribute_entity_type_name.

        This is useful for matching entities by their serialized type name (e.g., "Surface", "Edge").
        Supports both single type name and multiple type names for efficient batch lookup.

        Parameters:
            type_name: Single type name string or list of type name strings to search for.

        Returns:
            list[EntityBase]: All entities with matching type names.

        Examples:
            >>> registry.find_by_type_name("Surface")
            >>> registry.find_by_type_name(["Surface", "MirroredSurface"])
        """
        type_names_to_find = [type_name] if isinstance(type_name, str) else type_name
        type_name_set = set(type_names_to_find)

        matched_entities = []
        for entity_list in self.internal_registry.values():
            for entity in entity_list:
                if entity.private_attribute_entity_type_name in type_name_set:
                    matched_entities.append(entity)
        return matched_entities

    def find_by_type_name_and_id(self, *, entity_type: str, entity_id: str) -> EntityBase | None:
        """Find entity by serialized type name and asset id.

        Parameters
        ----------
        entity_type : str
            Serialized type name (EntityBase.private_attribute_entity_type_name), e.g. "Surface".
        entity_id : str
            Asset id (EntityBase.private_attribute_id).

        Returns
        -------
        Optional[EntityBase]
            Matched entity if found, otherwise None.
        """
        if not isinstance(entity_type, str):
            raise Flow360ValueError(f"[Internal] entity_type must be a string. Received: {type(entity_type).__name__}.")
        if not isinstance(entity_id, str):
            raise Flow360ValueError(f"[Internal] entity_id must be a string. Received: {type(entity_id).__name__}.")

        matched_entities: list[EntityBase] = []
        for entity_list in self.internal_registry.values():
            for entity in entity_list:
                if not isinstance(entity, EntityBase):
                    continue
                if (
                    entity.private_attribute_entity_type_name == entity_type
                    and entity.private_attribute_id == entity_id
                ):
                    matched_entities.append(entity)

        if len(matched_entities) > 1:
            raise ValueError(
                f"[INTERNAL] Multiple entities with the same type/id ({entity_type}:{entity_id}) found."
                " Data is likely corrupted."
            )
        if not matched_entities:
            return None
        return matched_entities[0]

    @classmethod
    def from_entity_info(cls, entity_info: Any) -> "EntityRegistry":
        """Build registry by referencing entities from entity_info.

        This is for the DraftContext workflow only. Legacy asset code
        continues to use entity_info.get_persistent_entity_registry().

        Parameters:
            entity_info: One of GeometryEntityInfo, VolumeMeshEntityInfo, or SurfaceMeshEntityInfo.
                Must be a deserialized object instance, not a dictionary.

        Returns:
            EntityRegistry: A registry populated with references to entities from entity_info.
        """
        registry = cls()  # type: ignore[call-arg]
        registry._register_from_entity_info(entity_info)
        return registry

    def _register_from_entity_info(self, entity_info: Any) -> None:
        """Populate internal_registry with references to entity_info entities.

        This method extracts all entities from the given entity_info and registers
        them in this registry. It handles all three entity info types:
        - GeometryEntityInfo: grouped_faces, grouped_edges, grouped_bodies
        - VolumeMeshEntityInfo: boundaries, zones
        - SurfaceMeshEntityInfo: boundaries

        All entity info types also have draft_entities and ghost_entities which are
        registered as well.

        Note:
        self is supposed to be an empty registry.
        """
        known_frozen_hashes: set[str] = set()

        def _register_selected_grouping(group_tag: Any, attribute_names: Any, grouped_items: Any) -> None:
            """Helper to register entities from the selected grouping."""
            if group_tag and group_tag in attribute_names:
                idx = attribute_names.index(group_tag)
                if idx < len(grouped_items):
                    for entity in grouped_items[idx]:
                        nonlocal known_frozen_hashes
                        known_frozen_hashes = self.fast_register(entity, known_frozen_hashes)
                else:
                    raise Flow360ValueError(
                        f"Group tag {group_tag} maps to index {idx}, but grouped_items only has "
                        f"{len(grouped_items)} entries."
                    )

        if entity_info.type_name == "GeometryEntityInfo":
            _register_selected_grouping(
                entity_info.face_group_tag,
                entity_info.face_attribute_names,
                entity_info.grouped_faces,
            )
            _register_selected_grouping(
                entity_info.edge_group_tag,
                entity_info.edge_attribute_names,
                entity_info.grouped_edges,
            )
            _register_selected_grouping(
                entity_info.body_group_tag,
                entity_info.body_attribute_names,
                entity_info.grouped_bodies,
            )

        elif entity_info.type_name == "VolumeMeshEntityInfo":
            for boundary in entity_info.boundaries:
                known_frozen_hashes = self.fast_register(boundary, known_frozen_hashes)
            for zone in entity_info.zones:
                known_frozen_hashes = self.fast_register(zone, known_frozen_hashes)

        elif entity_info.type_name == "SurfaceMeshEntityInfo":
            for boundary in entity_info.boundaries:
                known_frozen_hashes = self.fast_register(boundary, known_frozen_hashes)

        # Common to all: draft_entities and ghost_entities
        for entity in entity_info.draft_entities:
            known_frozen_hashes = self.fast_register(entity, known_frozen_hashes)
        for entity in entity_info.ghost_entities:
            known_frozen_hashes = self.fast_register(entity, known_frozen_hashes)
