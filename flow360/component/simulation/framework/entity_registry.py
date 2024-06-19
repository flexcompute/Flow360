import re
from typing import Any

import numpy as np
import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.log import log
from tests.utils import compare_to_ref


class MergeConflictError(Exception):
    pass


def __combine_bools(input_data):
    # If the input is a single boolean, return it directly
    if isinstance(input_data, bool):
        return input_data
    # If the input is a numpy ndarray, flatten it
    elif isinstance(input_data, np.ndarray):
        input_data = input_data.ravel()
    # If the input is not a boolean or an ndarray, assume it's an iterable of booleans
    return all(input_data)


def _merge_objects(obj_old, obj_new, overwrite_existing: bool = False):
    """Merges two objects of the same type and same name, raising an exception if there are conflicts.
    Parameters:
        obj_old: The original object to merge into.
        obj_new: The new object to merge into the original object.
        overwrite_existing: when true we just overwrite the existing ones with the new ones.
    """

    should_overwrite = False
    for attr, value in obj_new.__dict__.items():
        if attr in obj_old.__dict__:
            diff_result = __combine_bools(obj_old.__dict__[attr] != value)
            if obj_old.__dict__[attr] is not None and diff_result:
                if overwrite_existing:
                    should_overwrite = True
                else:
                    raise MergeConflictError(
                        f"Conflict on attribute '{attr}': {obj_old.__dict__[attr]} != {value}"
                    )
        # for new attr from new object, we just add it to the old object.
        obj_old.__dict__[attr] = value
    # if we found conflict and overwrite_existing is true then we overwrite the existing object with the new one.
    if should_overwrite:
        obj_old = obj_new


class EntityRegistry(Flow360BaseModel):
    """
    A registry for managing and storing instances of various entity types.

    This class provides methods to register entities, retrieve entities by their type,
    and find entities by name patterns using regular expressions.

    Attributes:
        internal_registry (Dict[str, List[EntityBase]]): A dictionary that maps entity types to lists of instances.
    """

    internal_registry: dict[str, list[Any]] = pd.Field({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def register(self, entity, overwrite_existing: bool = False):
        """
        Registers an entity in the registry under its type.

        Parameters:
            entity (EntityBase): The entity instance to register.
        """
        if entity.entity_type not in self.internal_registry:
            self.internal_registry[entity.entity_type] = []

        for existing_entity in self.internal_registry[entity.entity_type]:
            if existing_entity.name == entity.name:
                # Same type and same name. Now we try to update existing entity with new values.
                try:
                    _merge_objects(existing_entity, entity, overwrite_existing)
                    return
                except MergeConflictError as e:
                    log.debug("Merge conflict: %s", e)
                    raise ValueError(
                        f"Entity with name '{entity.name}' and type '{entity.entity_type}' already exists and have different definition."
                    )
                except Exception as e:
                    log.debug("Merge failed unexpectly: %s", e)
                    raise

        self.internal_registry[entity.entity_type].append(entity)

    def get_all_entities_of_given_type(self, entity_type):
        """
        Retrieves all entities of a specified type.

        Parameters:
            entity_type (Type[EntityBase]): The class of the entities to retrieve.

        Returns:
            List[EntityBase]: A list of registered entities of the specified type.
        """
        return self.internal_registry.get(
            entity_type.model_fields["private_attribute_registry_bucket_name"].default, []
        )

    def find_by_name_pattern(self, pattern: str):
        """
        Finds all registered entities whose names match a given pattern.

        Parameters:
            pattern (str): A naming pattern, which can include '*' as a wildcard.

        Returns:
            List[EntityBase]: A list of entities whose names match the pattern.
        """
        matched_entities = []
        regex = re.compile(pattern.replace("*", ".*"))
        for entity_list in self.internal_registry.values():
            matched_entities.extend(filter(lambda x: regex.match(x.name), entity_list))
        return matched_entities

    def show(self):
        """
        Prints a list of all registered entities, grouped by type.
        """
        index = 0
        print("---- Content of the registry ----")
        for entity_type, entities in self.internal_registry.items():
            print(f"   Entities of type '{entity_type}':")
            for entity in entities:
                print(f"     - [{index:03d}] {entity}")
                index += 1
        print("---- End of content ----")

    def clear(self):
        """
        Clears all entities from the registry.
        """
        self.internal_registry.clear()

    def contains(self) -> bool:
        """
        Returns True if the registry contains any entities, False otherwise.
        """
        return bool(self.internal_registry)

    def entity_count(self) -> int:
        """Return total number of entities in the registry."""
        count = 0
        for list_of_entities in self.internal_registry.values():
            count += len(list_of_entities)
        return count

    @property
    def is_empty(self):
        return self.internal_registry == {}
