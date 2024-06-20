import re
from typing import Any

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import (
    EntityBase,
    MergeConflictError,
    _merge_objects,
)
from flow360.log import log


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

    internal_registry: dict[str, list[Any]] = pd.Field({})

    def register(self, entity: EntityBase):
        """
        Registers an entity in the registry under its type.

        Parameters:
            entity (EntityBase): The entity instance to register.
        """
        if entity.entity_bucket not in self.internal_registry:
            self.internal_registry[entity.entity_bucket] = []

        for existing_entity in self.internal_registry[entity.entity_bucket]:
            if existing_entity.name == entity.name:
                # Same type and same name. Now we try to update existing entity with new values.
                try:
                    existing_entity = _merge_objects(existing_entity, entity)
                    return
                except MergeConflictError as e:
                    log.debug("Merge conflict: %s", e)
                    raise ValueError(
                        f"Entity with name '{entity.name}' and type '{entity.entity_bucket}' already exists and have different definition."
                    )
                except Exception as e:
                    log.debug("Merge failed unexpectly: %s", e)
                    raise

        self.internal_registry[entity.entity_bucket].append(entity)

    def get_all_entities_of_given_bucket(self, entity_bucket):
        """
        Retrieves all entities in a specified bucket.

        Parameters:
            entity_bucket (Type[EntityBase]): The class of the entities to retrieve.

        Returns:
            List[EntityBase]: A list of registered entities of the specified type.
        """
        return self.internal_registry.get(
            entity_bucket.model_fields["private_attribute_registry_bucket_name"].default, []
        )

    def find_by_naming_pattern(self, pattern: str) -> list[EntityBase]:
        """
        Finds all registered entities whose names match a given pattern.

        Parameters:
            pattern (str): A naming pattern, which can include '*' as a wildcard.

        Returns:
            List[EntityBase]: A list of entities whose names match the pattern.
        """
        matched_entities = []
        if "*" in pattern:
            regex_pattern = pattern.replace("*", ".*")
        else:
            regex_pattern = f"^{pattern}$"  # Exact match if no '*'

        regex = re.compile(regex_pattern)
        for entity_list in self.internal_registry.values():
            matched_entities.extend(filter(lambda x: regex.match(x.name), entity_list))
        return matched_entities

    def find_by_name(self, name: str):
        """Retrieve the entity with the given name from the registry."""
        entities = self.find_by_naming_pattern(name)
        if entities == []:
            raise ValueError(f"No entity found in registry with given name: '{name}'.")
        if len(entities) > 1:
            raise ValueError(f"Multiple entities found in registry with given name: '{name}'.")
        return entities[0]

    def show(self):
        """
        Prints a list of all registered entities, grouped by type.
        """
        index = 0
        print("---- Content of the registry ----")
        for entity_bucket, entities in self.internal_registry.items():
            print(f"    Entities of type '{entity_bucket}':")
            for entity in entities:
                print(f"    - [{index:03d}]\n{entity}")
                index += 1
        print("---- End of content ----")

    def clear(self):
        """
        Clears all entities from the registry.
        """
        self.internal_registry.clear()

    def contains(self, entity: EntityBase) -> bool:
        """
        Returns True if the registry contains any entities, False otherwise.
        """
        if entity.entity_bucket in self.internal_registry:
            if entity in self.internal_registry[entity.entity_bucket]:
                return True
        return False

    def entity_count(self) -> int:
        """Return total number of entities in the registry."""
        count = 0
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
        if bucket_to_find not in self.internal_registry:
            return

        for entity in self.internal_registry[bucket_to_find]:
            if entity.name == new_entity.name:
                self.internal_registry[bucket_to_find].remove(entity)
                self.internal_registry[bucket_to_find].append(new_entity)
                return

        self.register(new_entity)

    @property
    def is_empty(self):
        return self.internal_registry == {}
