"""Registry for managing and storing instances of various entity types."""

from typing import Any, Dict, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.utils import _naming_pattern_handler
from flow360.exceptions import Flow360ValueError


class StringIndexableList(list):
    """
    An extension of a list that allows accessing elements inside it through a string key.
    """

    def __getitem__(self, key: Union[str, slice, int]):
        if isinstance(key, str):
            returned_items = []
            for item in self:
                try:
                    item_ret_value = item[key]
                except KeyError:
                    item_ret_value = []
                except Exception as e:
                    raise ValueError(
                        f"Trying to access something in {item} through string indexing, which is not allowed."
                    ) from e
                if isinstance(item_ret_value, list):
                    returned_items += item_ret_value
                else:
                    returned_items.append(item_ret_value)
            if not returned_items:
                raise ValueError(
                    "No entity found in registry for parent entities: "
                    + f"{', '.join([f'{entity.name}' for entity in self])} with given name/naming pattern: '{key}'."
                )
            return returned_items
        return super().__getitem__(key)


class EntityRegistryView:
    """
    Type-filtered view over EntityRegistry with glob pattern support.

    Provides a simplified interface for accessing entities of **only** a specific type,
    supporting both direct name lookup and glob pattern matching.
    """

    def __init__(self, registry: "EntityRegistry", entity_type: type[EntityBase]) -> None:
        self._registry = registry
        self._entity_type = entity_type

    def __iter__(self):
        """Iterate over all entities of this type."""
        return iter(self._entities)

    def __len__(self):
        """Return the number of entities of this type."""
        return len(self._entities)

    @property
    def _entities(self) -> list[EntityBase]:
        """Get all entities of the target type (exact type match only)."""
        # Direct lookup in internal_registry for exact type
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.utils import is_exact_instance

        entities_of_type = self._registry.internal_registry.get(self._entity_type, [])
        # Filter to ensure exact type match (not subclasses)
        return [item for item in entities_of_type if is_exact_instance(item, self._entity_type)]

    def __getitem__(self, key: str) -> Union[EntityBase, list[EntityBase]]:
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
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.framework.entity_utils import (
            compile_glob_cached,
        )

        matcher = compile_glob_cached(pattern=key)
        matched = [entity for entity in self._entities if matcher.match(entity.name)]

        if not matched:
            raise ValueError(
                f"No entity found in registry with given name/naming pattern: '{key}'."
            )
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

    internal_registry: Dict[type[EntityBase], list[Any]] = pd.Field({})

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
            # pylint: disable=unsupported-assignment-operation
            self.internal_registry[entity_type] = []

        # pylint: disable=protected-access
        if entity._get_hash() in known_frozen_hashes:
            return known_frozen_hashes
        known_frozen_hashes.add(entity._get_hash())

        # pylint: disable=unsubscriptable-object
        self.internal_registry[entity_type].append(entity)
        return known_frozen_hashes

    def register(self, entity: EntityBase):
        """
        Registers an entity in the registry under its type.

        Parameters:
            entity (EntityBase): The entity instance to register.
        """
        entity_type = type(entity)
        # pylint: disable=unsupported-membership-test
        if entity_type not in self.internal_registry:
            # pylint: disable=unsupported-assignment-operation
            self.internal_registry[entity_type] = []

        # pylint: disable=unsubscriptable-object
        for existing_entity in self.internal_registry[entity_type]:
            # pylint: disable=protected-access
            if existing_entity._get_hash() == entity._get_hash():
                # Identical entities. Just ignore
                return

        # pylint: disable=unsubscriptable-object
        self.internal_registry[entity_type].append(entity)

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
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.utils import get_combined_subclasses

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
        matched_entities = []
        regex = _naming_pattern_handler(pattern=pattern)
        # pylint: disable=no-member
        for entity_list in self.internal_registry.values():
            matched_entities.extend(filter(lambda x: regex.match(x.name), entity_list))

        if not matched_entities and error_when_no_match is True:
            raise ValueError(
                f"No entity found in registry with given name/naming pattern: '{pattern}'."
            )
        if enforce_output_as_list is False and len(matched_entities) == 1:
            return matched_entities[0]

        return matched_entities

    def __str__(self):
        """
        Returns a string representation of all registered entities, grouped by type.
        """
        index = 0
        result = "---- Content of the registry ----\n"
        # pylint: disable=no-member
        for entity_type, entities in self.internal_registry.items():
            result += f"\n    Entities of type '{entity_type.__name__}':\n"
            for entity in entities:
                result += f"    - [{index:05d}]\n{entity}\n"
                index += 1
        result += "---- End of content ----"
        return result

    def clear(self, entity_type: Union[None, type[EntityBase]] = None):
        """
        Clears all entities from the registry.
        """
        # pylint: disable=no-member
        if entity_type is not None:
            if entity_type in self.internal_registry.keys():
                # pylint: disable=unsubscriptable-object
                self.internal_registry[entity_type].clear()
        else:
            self.internal_registry.clear()

    def contains(self, entity: EntityBase) -> bool:
        """
        Returns True if the registry contains any entities, False otherwise.
        """
        entity_type = type(entity)
        # pylint: disable=unsupported-membership-test
        if entity_type in self.internal_registry:
            # pylint: disable=unsubscriptable-object
            if entity in self.internal_registry[entity_type]:
                return True
        return False

    def entity_count(self) -> int:
        """Return total number of entities in the registry."""
        count = 0
        # pylint: disable=no-member
        for list_of_entities in self.internal_registry.values():
            count += len(list_of_entities)
        return count

    def replace_existing_with(self, new_entity: EntityBase):
        """
        Replaces an entity in the registry with a new entity (searched by name across all types).

        This searches for an entity with the same name across all registered types.
        If found, removes the old entity and registers the new one.
        If not found, simply registers the new entity.

        Parameters:
            new_entity (EntityBase): The new entity to replace the existing entity with.
        """
        # Search by name across all types to find entity to replace
        # pylint: disable=no-member
        for entity_type, entity_list in self.internal_registry.items():
            for i, entity in enumerate(entity_list):
                if entity.name == new_entity.name:
                    # Found entity with matching name - remove it
                    self.internal_registry[entity_type].pop(i)
                    # Register new entity under its own type
                    self.register(new_entity)
                    return

        # No matching entity found, just register the new one
        self.register(new_entity)

    def find_by_asset_id(self, *, entity_id: str, entity_class: type[EntityBase]):
        """
        Find the entity with matching asset id and the same type as the input entity class.
        Return None if no such entity is found.
        """
        # Get entities of the specific type (including subclasses)
        entities = self.view(entity_class)._entities  # pylint: disable=protected-access
        matched_entities = [item for item in entities if item.private_attribute_id == entity_id]

        if len(matched_entities) > 1:
            raise ValueError(
                f"[INTERNAL] Multiple entities with the same asset id ({entity_id}) found."
                " Data is likely corrupted."
            )
        if len(matched_entities) == 0:
            return None
        return matched_entities[0]

    @property
    def is_empty(self):
        """Return True if the registry is empty, False otherwise."""
        return not self.internal_registry

    def find_by_name(self, name: str) -> Optional[EntityBase]:
        """Find entity by exact name match.

        Parameters:
            name (str): The exact name to search for.

        Returns:
            EntityBase or None: The entity if found, None otherwise.
        """
        # pylint: disable=no-member
        for entity_list in self.internal_registry.values():
            for entity in entity_list:
                if entity.name == name:
                    return entity
        return None

    def find_by_type(self, entity_class: type[EntityBase]) -> list[EntityBase]:
        """Find all registered entities of a given type (including subclasses).

        Parameters:
            entity_class (type[EntityBase]): The entity class to search for.

        Returns:
            list[EntityBase]: All entities that are instances of the given class.
        """
        matched_entities = []
        # pylint: disable=no-member
        for entity_type, entity_list in self.internal_registry.items():
            # Check if entity_type is the target class or a subclass
            if issubclass(entity_type, entity_class):
                matched_entities.extend(entity_list)
        return matched_entities

    def find_by_type_name(self, type_name: str) -> list[EntityBase]:
        """Find all registered entities with a given private_attribute_entity_type_name.

        This is useful for matching entities by their serialized type name (e.g., "Surface", "Edge").

        Parameters:
            type_name (str): The entity type name to search for.

        Returns:
            list[EntityBase]: All entities with matching type name.
        """
        matched_entities = []
        # pylint: disable=no-member
        for entity_list in self.internal_registry.values():
            for entity in entity_list:
                if entity.private_attribute_entity_type_name == type_name:
                    matched_entities.append(entity)
        return matched_entities

    @classmethod
    def from_entity_info(cls, entity_info) -> "EntityRegistry":
        """Build registry by referencing entities from entity_info.

        This is for the DraftContext workflow only. Legacy asset code
        continues to use entity_info.get_persistent_entity_registry().

        Parameters:
            entity_info: One of GeometryEntityInfo, VolumeMeshEntityInfo, or SurfaceMeshEntityInfo.
                Must be a deserialized object instance, not a dictionary.

        Returns:
            EntityRegistry: A registry populated with references to entities from entity_info.
        """
        registry = cls()
        registry._register_from_entity_info(entity_info)
        return registry

    def _register_from_entity_info(self, entity_info):  # pylint: disable=too-many-branches
        """Populate internal_registry with references to entity_info entities.

        This method extracts all entities from the given entity_info and registers
        them in this registry. It handles all three entity info types:
        - GeometryEntityInfo: grouped_faces, grouped_edges, grouped_bodies
        - VolumeMeshEntityInfo: boundaries, zones
        - SurfaceMeshEntityInfo: boundaries

        All entity info types also have draft_entities and ghost_entities which are
        registered as well.
        """
        known_frozen_hashes = set()

        def _register_selected_grouping(group_tag, attribute_names, grouped_items):
            """Helper to register entities from the selected grouping."""
            if group_tag and group_tag in attribute_names:
                idx = attribute_names.index(group_tag)
                if idx < len(grouped_items):
                    for entity in grouped_items[idx]:
                        nonlocal known_frozen_hashes
                        known_frozen_hashes = self.fast_register(entity, known_frozen_hashes)
                else:
                    raise Flow360ValueError(
                        f"Group tag {group_tag} not found in attribute names {attribute_names}."
                    )

        if entity_info.type_name == "GeometryEntityInfo":
            # Register only entities from the selected groupings
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


class SnappyBodyRegistry(EntityRegistry):
    """
    Extension of :class:`EntityRegistry` to be used with :class:`SnappyBody`, allows double indexing
    for accessing the boundaries under certain :class:`SnappyBody`.
    """

    def find_by_naming_pattern(
        self, pattern: str, enforce_output_as_list: bool = True, error_when_no_match: bool = False
    ) -> StringIndexableList[EntityBase]:
        """
        Finds all registered entities whose names match a given pattern.

        Parameters:
            pattern (str): A naming pattern, which can include '*' as a wildcard.

        Returns:
            List[EntityBase]: A list of entities whose names match the pattern.
        """
        matched_entities = StringIndexableList()
        regex = _naming_pattern_handler(pattern=pattern)
        # pylint: disable=no-member
        for entity_list in self.internal_registry.values():
            matched_entities.extend(filter(lambda x: regex.match(x.name), entity_list))

        if not matched_entities and error_when_no_match is True:
            raise ValueError(
                f"No entity found in registry with given name/naming pattern: '{pattern}'."
            )
        if enforce_output_as_list is False and len(matched_entities) == 1:
            return matched_entities[0]

        return matched_entities

    def __getitem__(self, key):
        """
        Get the entity by name.
        `key` is the name of the entity or the naming pattern if wildcard is used.
        """
        if isinstance(key, str) is False:
            raise Flow360ValueError(f"Entity naming pattern: {key} is not a string.")

        return self.find_by_naming_pattern(
            key, enforce_output_as_list=False, error_when_no_match=True
        )
