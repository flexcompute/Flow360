import re

from flow360.log import log


class MergeConflictError(Exception):
    pass


def _merge_objects(obj_old, obj_new):
    """Merges two objects of the same type and same name, raising an exception if there are conflicts.
    Parameters:
        obj_old: The original object to merge into.
        obj_new: The new object to merge into the original object.
    """
    for attr, value in obj_new.__dict__.items():
        if attr in obj_old.__dict__:
            if obj_old.__dict__[attr] is not None and obj_old.__dict__[attr] != value:
                raise MergeConflictError(
                    f"Conflict on attribute '{attr}': {obj_old.__dict__[attr]} != {value}"
                )
        obj_old.__dict__[attr] = value


class EntityRegistry:
    """
    A registry for managing and storing instances of various entity types.

    This class provides methods to register entities, retrieve entities by their type,
    and find entities by name patterns using regular expressions.

    Attributes:
        _registry (Dict[str, List[EntityBase]]): A dictionary that maps entity types to lists of instances.
    """

    _registry = None

    def __init__(self) -> None:
        self._registry = {}

    def register(self, entity):
        """
        Registers an entity in the registry under its type.

        Parameters:
            entity (EntityBase): The entity instance to register.
        """
        if entity.entity_type not in self._registry:
            self._registry[entity.entity_type] = []

        for existing_entity in self._registry[entity.entity_type]:
            if existing_entity.name == entity.name:
                # Same type and same name. Now we try to update existing entity with new values.
                try:
                    _merge_objects(existing_entity, entity)
                    return
                except MergeConflictError as e:
                    log.debug("Merge conflict: %s", e)
                    raise ValueError(
                        f"Entity with name '{entity.name}' and type '{entity.entity_type}' already exists and have different definition."
                    )
                except Exception as e:
                    log.debug("Merge failed unexpectly: %s", e)
                    raise

        self._registry[entity.entity_type].append(entity)

    def get_all_entities_of_given_type(self, entity_type):
        """
        Retrieves all entities of a specified type.

        Parameters:
            entity_type (Type[EntityBase]): The class of the entities to retrieve.

        Returns:
            List[EntityBase]: A list of registered entities of the specified type.
        """
        return self._registry.get(entity_type.entity_type, [])

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
        for entity_list in self._registry.values():
            matched_entities.extend(filter(lambda x: regex.match(x.name), entity_list))
        return matched_entities

    def show(self):
        """
        Prints a list of all registered entities, grouped by type.
        """
        for entity_type, entities in self._registry.items():
            print(f"Entities of type '{entity_type}':")
            for entity in entities:
                print(f"  - {entity}")

    def clear(self):
        """
        Clears all entities from the registry.
        """
        self._registry.clear()
