from __future__ import annotations

import copy
from abc import ABCMeta
from typing import List, Union, get_args

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_registry import EntityRegistry


class EntityBase(Flow360BaseModel, metaclass=ABCMeta):
    """
    Base class for dynamic entity types with automatic registration upon instantiation.

    Attributes:
        _entity_type (str): A string representing the specific type of the entity.
                            This should be set in subclasses to differentiate between entity types.
                            Note this controls the granularity of the registry.
        name (str): The name of the entity, used for identification and retrieval.
    """

    _entity_type: str = None
    name: str = pd.Field(frozen=True)

    def __init__(self, **data):
        """
        Initializes a new entity and registers it in the global registry.

        Parameters:
            data: Keyword arguments containing initial values for fields declared in the entity.
        """
        super().__init__(**data)
        assert self._entity_type is not None, "_entity_type is not defined in the entity class."
        EntityRegistry.register(self)

    def copy(self, update=None, **kwargs) -> EntityBase:
        """
        Creates a copy of the entity with compulsory updates.

        Parameters:
            update: A dictionary containing the updated attributes to apply to the copied entity.
            **kwargs: Additional arguments to pass to the copy constructor.

        Returns:
            A copy of the entity with the specified updates.
        """
        if update is None:
            raise ValueError(
                "Change is necessary when copying an entity as there cannot be two identical entities at the same time. Please use update parameter to change the entity attributes."
            )
        if "name" not in update or update["name"] == self.name:
            raise ValueError(
                "Copying an entity requires a new name to be specified. Please provide a new name in the update dictionary."
            )
        super().copy(update=update, **kwargs)


class _CombinedMeta(type(Flow360BaseModel), type):
    pass


class _EntitiesListMeta(_CombinedMeta):
    def __getitem__(cls, entity_types):
        """
        Creates a new class with the specified entity types as a list of stored entities.
        """
        if not isinstance(entity_types, tuple):
            entity_types = (entity_types,)
        union_type = Union[entity_types]
        annotations = {"stored_entities": List[union_type]}
        new_cls = type(
            f"{cls.__name__}[{','.join([t.__name__ for t in entity_types])}]",
            (cls,),
            {"__annotations__": annotations},
        )
        return new_cls


class EntityList(Flow360BaseModel, metaclass=_EntitiesListMeta):
    """
    Represents a collection of volume entities in the Flow360 simulation.

    Attributes:
        stored_entities (List[Union[Box, Cylinder, str]]): List of stored entities, which can be
            instances of `Box`, `Cylinder`, or strings representing naming patterns.

    Methods:
        format_input_to_list(cls, input: List) -> dict: Class method that formats the input to a
            dictionary with the key 'stored_entities'.
        check_duplicate_entity_in_list(cls, values): Class method that checks for duplicate entities
            in the list of stored entities.
        get_expanded_entities(self): Method that processes the stored entities to resolve any naming
            patterns into actual entity references, expanding and filtering based on the defined
            entity types.

    """

    stored_entities: List = pd.Field()

    @pd.model_validator(mode="before")
    @classmethod
    def format_input_to_list(cls, input: List) -> dict:
        if isinstance(input, str) or isinstance(input, EntityBase):
            return dict(stored_entities=[input])
        elif isinstance(input, list):
            if input == []:
                raise ValueError("Invalid input type to `entities`, list is empty.")
            return dict(stored_entities=input)
        else:
            raise ValueError(f"Invalid input type to `entities`: {type(input)}")

    @pd.field_validator("stored_entities", mode="after")
    @classmethod
    def check_duplicate_entity_in_list(cls, values):
        seen = []
        for value in values:
            if value in seen:
                if isinstance(value, EntityBase):
                    raise ValueError(f"Duplicate entity found, name: {value.name}")
                raise ValueError(f"Duplicate entity found: {value}")
            seen.append(value)
        return values

    def get_expanded_entities(self):
        """
        Processes `stored_entities` to resolve any naming patterns into actual entity
        references, expanding and filtering based on the defined entity types. This ensures that
        all entities referenced either directly or by pattern are valid and registered.

        Raises:
            TypeError: If an entity does not match the expected type.
        Returns:
            Deep copy of the exapnded entities list.
        """

        entities = getattr(self, "stored_entities", [])

        entity_field_type = self.__annotations__["stored_entities"]
        if hasattr(entity_field_type, "__origin__") and entity_field_type.__origin__ is list:
            valid_types = get_args(entity_field_type)[0]
            if hasattr(valid_types, "__origin__") and valid_types.__origin__ is Union:
                valid_types = get_args(valid_types)
            else:
                valid_types = (valid_types,)
        expanded_entities = []

        for entity in entities:
            if isinstance(entity, str):
                # Expand based on naming pattern registered in the Registry
                pattern_matched_entities = EntityRegistry.find_by_name_pattern(entity)
                # Filter pattern matched entities by valid types
                expanded_entities.extend(
                    [
                        e
                        for e in pattern_matched_entities
                        if isinstance(e, tuple(valid_types)) and e not in expanded_entities
                    ]
                )
            elif entity not in expanded_entities:
                # Direct entity references are simply appended if they are of a valid type
                expanded_entities.append(entity)
        if expanded_entities == []:
            raise ValueError(
                f"Failed to find any matching entity with {entities}. Please check the input to entities."
            )
        return copy.deepcopy(expanded_entities)
