from abc import ABCMeta

from flow360.component.simulation.framework.entity_base import EntityBase
from flow360.component.simulation.framework.entity_registry import EntityRegistry


class AssetBase(metaclass=ABCMeta):
    _registry: EntityRegistry

    def __init__(self):
        self._registry = EntityRegistry()

    def __getitem__(self, key: str) -> list[EntityBase]:
        """Use [] to access the registry"""
        if isinstance(key, str) == False:
            raise ValueError(f"Entity naming pattern: {key} is not a string.")
        found_entities = self._registry.find_by_name_pattern(key)
        if found_entities == []:
            raise ValueError(
                f"Failed to find any matching entity with {key}. Please check your input."
            )
        return found_entities