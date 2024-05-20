from __future__ import annotations

import copy
from abc import ABCMeta
from collections import defaultdict
from typing import List, Union, get_args

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.log import log


class EntityBase(Flow360BaseModel, metaclass=ABCMeta):
    """
    Base class for dynamic entity types.

    Attributes:
        _entity_type (str): A string representing the specific type of the entity.
                            This should be set in subclasses to differentiate between entity types.
                            Note this controls the granularity of the registry.
        _is_generic(bool): A flag indicating whether the entity is a generic entity (constructed from metadata).
        name (str): The name of the entity, used for identification and retrieval.
    """

    _entity_type: str = None
    _is_generic = False
    name: str = pd.Field(frozen=True)

    def __init__(self, **data):
        """
        Initializes a new entity and registers it in the global registry.

        Parameters:
            data: Keyword arguments containing initial values for fields declared in the entity.
        """
        super().__init__(**data)
        assert self._entity_type is not None, "_entity_type is not defined in the entity class."

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
                "Change is necessary when calling .copy() as there cannot be two identical entities at the same time. Please use update parameter to change the entity attributes."
            )
        if "name" not in update or update["name"] == self.name:
            raise ValueError(
                "Copying an entity requires a new name to be specified. Please provide a new name in the update dictionary."
            )
        return super().copy(update=update, **kwargs)


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


def _remove_duplicate_entities(expanded_entities: List[EntityBase]):
    """
    In the expanded entity list from `_get_expanded_entities` we will very likely have generic entities
    which comes from asset metadata. These entities may have counterparts given by users. We remove the
    generic ones when they have duplicate counterparts because the counterparts will likely have more info.

    For example `entities = [my_mesh["*"], user_defined_zone]`. We need to remove duplicates from the expanded list.
    """
    all_entities = defaultdict(list)

    for entity in expanded_entities:
        all_entities[entity.name].append(entity)

    for entity_list in all_entities.values():
        if len(entity_list) > 1:
            for entity in entity_list:
                if entity._is_generic and len(entity_list) > 1:
                    entity_list.remove(entity)

        assert len(entity_list) == 1

    return [entity_list[0] for entity_list in all_entities.values()]


class EntityList(Flow360BaseModel, metaclass=_EntitiesListMeta):
    """
    The type accepting a list of entities or (name, registry) pair.

    Attributes:
        stored_entities (List[Union[EntityBase, Tuple[str, registry]]]): List of stored entities, which can be
            instances of `Box`, `Cylinder`, or strings representing naming patterns.

    Methods:
        _format_input_to_list(cls, input: List) -> dict: Class method that formats the input to a
            dictionary with the key 'stored_entities'.
        _check_duplicate_entity_in_list(cls, values): Class method that checks for duplicate entities
            in the list of stored entities.
        _get_expanded_entities(self): Method that processes the stored entities to resolve any naming
            patterns into actual entity references, expanding and filtering based on the defined
            entity types.

    """

    stored_entities: List = pd.Field()

    @classmethod
    def _get_valid_entity_types(cls):
        """Get the list of types that the entity list can accept."""
        entity_field_type = cls.__annotations__.get("stored_entities")
        if (
            entity_field_type is not None
            and hasattr(entity_field_type, "__origin__")
            and entity_field_type.__origin__ is list
        ):
            valid_types = get_args(entity_field_type)[0]
            if hasattr(valid_types, "__origin__") and valid_types.__origin__ is Union:
                valid_types = get_args(valid_types)
            else:
                valid_types = (valid_types,)
            return valid_types
        raise TypeError("Internal error, the metaclass for EntityList is not properly set.")

    @classmethod
    def _valid_individual_input(cls, input):
        """Validate each individual element in a list or as standalone entity."""
        if isinstance(input, str) or isinstance(input, EntityBase):
            return input
        else:
            raise ValueError(
                f"Type({type(input)}) of input to `entities` ({input}) is not valid. Expected str or entity instance."
            )

    @pd.model_validator(mode="before")
    @classmethod
    def _format_input_to_list(cls, input: Union[dict, list]):
        """
        Flatten List[EntityBase] and put into stored_entities.
        """
        # Note:
        # 1. str comes from Param. These will be expanded before submission
        #    as the user may change Param which affects implicit entities (farfield existence patch for example).
        # 2. The List[EntityBase], comes from the Assets.
        # 3. EntityBase comes from direct specification of entity in the list.
        formated_input = []
        valid_types = cls._get_valid_entity_types()
        if isinstance(input, list):
            if input == []:
                raise ValueError("Invalid input type to `entities`, list is empty.")
            for item in input:
                if isinstance(item, list):  # Nested list comes from assets
                    [cls._valid_individual_input(individual) for individual in item]
                    formated_input.extend(
                        [
                            individual
                            for individual in item
                            if isinstance(individual, tuple(valid_types))
                        ]
                    )
                else:
                    cls._valid_individual_input(item)
                    if isinstance(item, tuple(valid_types)):
                        formated_input.append(item)
        elif isinstance(input, dict):
            return dict(stored_entities=input["stored_entities"])
        else:  # Single reference to an entity
            cls._valid_individual_input(input)
            if isinstance(item, tuple(valid_types)):
                formated_input.append(item)
        return dict(stored_entities=formated_input)

    @pd.field_validator("stored_entities", mode="after")
    @classmethod
    def _check_duplicate_entity_in_list(cls, values):
        seen = []
        for value in values:
            if value in seen:
                if isinstance(value, EntityBase):
                    log.warning(f"Duplicate entity found, name: {value.name}")
                else:
                    log.warning(f"Duplicate entity found: {value}")
                continue
            seen.append(value)
        return seen

    def _get_expanded_entities(self, supplied_registry: EntityRegistry = None) -> List[EntityBase]:
        """
        Processes `stored_entities` to resolve any naming patterns into actual entity
        references, expanding and filtering based on the defined entity types. This ensures that
        all entities referenced either directly or by pattern are valid and registered.

        **Warning**:
            This method has to be called during preprocessing stage of Param when all settings have
            been finalized. This ensures that all entities are registered in the registry (by assets or param).
            Maybe we check hash or something to ensure consistency/integrity?

        Raises:
            TypeError: If an entity does not match the expected type.
        Returns:
            Deep copy of the exapnded entities list.
        """

        entities = getattr(self, "stored_entities", [])

        valid_types = self.__class__._get_valid_entity_types()

        expanded_entities = []

        for entity in entities:
            if isinstance(entity, str):
                # Expand from supplied registry
                if supplied_registry is None:
                    raise ValueError(
                        f"Internal error, registry is not supplied for entity ({entity}) expansion. "
                    )
                # Expand based on naming pattern registered in the Registry
                pattern_matched_entities = supplied_registry.find_by_name_pattern(entity)
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

        expanded_entities = _remove_duplicate_entities(expanded_entities)

        if expanded_entities == []:
            raise ValueError(
                f"Failed to find any matching entity with {entities}. Please check the input to entities."
            )
        # TODO:  As suggested by Runda. We better prompt user what entities are actually used/expanded to avoid user input error. We need a switch to turn it on or off.
        return copy.deepcopy(expanded_entities)

    def preprocess(self, supplied_registry: EntityRegistry = None):
        """Expand and overwrite self.stored_entities in preparation for submissin/serialization."""
        self.stored_entities = self._get_expanded_entities(supplied_registry)
        return self
