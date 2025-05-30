"""Registry for managing and storing instances of various entity types."""

from typing import Any, Dict, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.utils import _naming_pattern_handler


class EntityRegistryBucket:
    """By reference, a snippet of certain collection of a EntityRegistry instance that is inside the same bucket."""

    # pylint: disable=too-few-public-methods
    def __init__(self, source_dict: dict, key: str):
        self._source = source_dict
        self._key = key

    @property
    def entities(self):
        """Return all entities in the bucket."""
        return self._source.get(self._key, [])

    def _get_property_values(self, property_name: str) -> list:
        """Get the given property of all the entities in the bucket as a list"""
        return [getattr(entity, property_name) for entity in self.entities]


class EntityRegistry(Flow360BaseModel):
    """
    A registry for managing and storing instances of various entity types.

    This class provides methods to register entities, retrieve entities by their type,
    and find entities by name patterns using regular expressions.

    Attributes:
        internal_registry (Dict[str, List[EntityBase]]): A dictionary that maps entity types to lists of instances.

    #Known Issues:
    frozen=True do not stop the user from changing the internal_registry
    """

    internal_registry: Dict[str, list[Any]] = pd.Field({})

    def register(self, entity: EntityBase):
        """
        Registers an entity in the registry under its type.

        Parameters:
            entity (EntityBase): The entity instance to register.
        """
        # pylint: disable=unsupported-membership-test
        if entity.entity_bucket not in self.internal_registry:
            # pylint: disable=unsupported-assignment-operation
            self.internal_registry[entity.entity_bucket] = []

        # pylint: disable=unsubscriptable-object
        for existing_entity in self.internal_registry[entity.entity_bucket]:
            # pylint: disable=protected-access
            if existing_entity._get_hash() == entity._get_hash():
                # Identical entities. Just ignore
                return

        # pylint: disable=unsubscriptable-object
        self.internal_registry[entity.entity_bucket].append(entity)

    def get_bucket(self, by_type: type[EntityBase]) -> EntityRegistryBucket:
        """Get the bucket of a certain type of entity."""
        return EntityRegistryBucket(
            self.internal_registry,
            by_type.model_fields["private_attribute_registry_bucket_name"].default,
        )

    def find_by_type(self, entity_class: type[EntityBase]) -> list[EntityBase]:
        """
        Finds all registered entities of a given type.
        """
        matched_entities = []
        # pylint: disable=no-member
        for entity_list in self.internal_registry.values():
            matched_entities.extend(filter(lambda x: isinstance(x, entity_class), entity_list))

        return matched_entities

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
        for entity_bucket, entities in self.internal_registry.items():
            result += f"\n    Entities of type '{entity_bucket}':\n"
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
            bucket_name = entity_type.model_fields["private_attribute_registry_bucket_name"].default
            if bucket_name in self.internal_registry.keys():
                # pylint: disable=unsubscriptable-object
                self.internal_registry[bucket_name].clear()
        else:
            self.internal_registry.clear()

    def contains(self, entity: EntityBase) -> bool:
        """
        Returns True if the registry contains any entities, False otherwise.
        """
        # pylint: disable=unsupported-membership-test
        if entity.entity_bucket in self.internal_registry:
            # pylint: disable=unsubscriptable-object
            if entity in self.internal_registry[entity.entity_bucket]:
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
        Replaces an entity in the registry with a new entity.

        Parameters:
            new_entity (EntityBase): The new entity to replace the existing entity with.
        """
        bucket_to_find = new_entity.entity_bucket
        # pylint: disable=unsupported-membership-test
        if bucket_to_find not in self.internal_registry:
            return

        # pylint: disable=unsubscriptable-object
        for entity in self.internal_registry[bucket_to_find]:
            if entity.name == new_entity.name:
                self.internal_registry[bucket_to_find].remove(entity)
                self.internal_registry[bucket_to_find].append(new_entity)
                return

        self.register(new_entity)

    def find_by_asset_id(self, *, entity_id: str, entity_class: type[EntityBase]):
        """
        Find the entity with matching asset id and the same entity bucket as the input entity.
        Return None if no such entity is found.
        """
        bucket = self.get_bucket(by_type=entity_class)
        matched_entities = [
            item for item in bucket.entities if item.private_attribute_id == entity_id
        ]

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
