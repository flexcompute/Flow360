from __future__ import annotations

import copy
from abc import ABCMeta
from typing import List, Optional, Union, get_args, get_origin

import numpy as np
import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.log import log


class MergeConflictError(Exception):
    pass


class EntityBase(Flow360BaseModel, metaclass=ABCMeta):
    """
    Base class for dynamic entity types.

    Attributes:
        private_attribute_registry_bucket_name (str):
            A string representing the specific type of the entity.
            This should be set in subclasses to differentiate between entity types.
            Warning:
            This controls the granularity of the registry and must be unique for each entity type and it is **strongly recommended NOT** to change it as it will bring up compatability problems.

        name (str):
            The name of the entity instance, used for identification and retrieval.
    """

    private_attribute_registry_bucket_name: str = "Invalid"
    private_attribute_entity_type_name: str = "Invalid"
    name: str = pd.Field(frozen=True)

    def __init__(self, **data):
        """
        Initializes a new entity and registers it in the global registry.

        Parameters:
            data: Keyword arguments containing initial values for fields declared in the entity.
        """
        super().__init__(**data)
        if self.entity_bucket == "Invalid":
            raise NotImplementedError(
                f"private_attribute_registry_bucket_name is not defined in the entity class: {self.__class__.__name__}."
            )
        if self.entity_type == "Invalid":
            raise NotImplementedError(
                f"private_attribute_entity_type_name is not defined in the entity class: {self.__class__.__name__}."
            )

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

    def __eq__(self, other):
        """Defines the equality comparison for entities to support usage in UniqueItemList."""
        if isinstance(other, EntityBase):
            return (self.name + "-" + self.__class__.__name__) == (
                other.name + "-" + other.__class__.__name__
            )
        return False

    @property
    def entity_bucket(self) -> str:
        return self.private_attribute_registry_bucket_name

    @entity_bucket.setter
    def entity_bucket(self, value: str):
        raise AttributeError("Cannot modify the bucket to which the entity belongs.")

    @property
    def entity_type(self) -> str:
        return self.private_attribute_entity_type_name

    @entity_type.setter
    def entity_type(self, value: str):
        raise AttributeError("Cannot modify the name of entity class.")

    def __str__(self) -> str:
        return "\n".join([f"        {attr}: {value}" for attr, value in self.__dict__.items()])

    def _is_generic(self):
        return self.__class__.__name__.startswith("Generic")


class _CombinedMeta(type(Flow360BaseModel), type):
    pass


class _EntityListMeta(_CombinedMeta):
    def __getitem__(cls, entity_types):
        """
        Creates a new class with the specified entity types as a list of stored entities.
        """
        if not isinstance(entity_types, tuple):
            entity_types = (entity_types,)
        union_type = Union[entity_types]
        annotations = {
            "stored_entities": Optional[List[union_type]]
        }  # Make sure it is somewhat consistent with the EntityList class
        new_cls = type(
            f"{cls.__name__}[{','.join([t.__name__ for t in entity_types])}]",
            (cls,),
            {"__annotations__": annotations},
        )
        return new_cls


def __combine_bools(input_data):
    # If the input is a single boolean, return it directly
    if isinstance(input_data, bool):
        return input_data
    # If the input is a numpy ndarray, flatten it
    elif isinstance(input_data, np.ndarray):
        input_data = input_data.ravel()
    # If the input is not a boolean or an ndarray, assume it's an iterable of booleans
    return all(input_data)


def _merge_objects(obj_old: EntityBase, obj_new: EntityBase) -> EntityBase:
    """
    Merges obj_new into obj_old, raising an exception if there are conflicts.
    Ideally the obj_old should be a non-generic one.
    Parameters:
        obj_old: The original object to merge into.
        obj_new: The new object to merge into the original object.
    """

    if obj_new.name != obj_old.name:
        raise MergeConflictError(
            "Make sure merge is intended as the names of two entities are different."
        )

    if obj_new._is_generic() == False and obj_old._is_generic() == True:
        # swap so that obj_old is **non-generic** and obj_new is **generic**
        obj_new, obj_old = obj_old, obj_new

    # Check the two objects are mergeable
    if obj_new._is_generic() == False and obj_old._is_generic() == False:
        if obj_new.__class__ != obj_old.__class__:
            raise MergeConflictError(
                f"Cannot merge objects of different class: {obj_old.__class__.__name__} and {obj_new.__class__.__name__}"
            )

    for attr, value in obj_new.__dict__.items():
        if attr in [
            "private_attribute_entity_type_name",
            "private_attribute_registry_bucket_name",
            "name",
        ]:
            continue
        if attr in obj_old.__dict__:
            found_conflict = __combine_bools(obj_old.__dict__[attr] != value)
            if found_conflict:
                if obj_old.__dict__[attr] is None:
                    # Populate obj_old with new info from lower priority object
                    obj_old.__dict__[attr] = value
                elif obj_new.__dict__[attr] is None:
                    # Ignore difference from lower priority object
                    continue
                else:
                    raise MergeConflictError(
                        f"Conflict on attribute '{attr}': {obj_old.__dict__[attr]} != {value}"
                    )
        # for new attr from new object, we just add it to the old object.
        if attr in obj_old.model_fields.keys():
            obj_old.__dict__[attr] = value

    return obj_old


def _remove_duplicate_entities(expanded_entities: List[EntityBase]):
    """
    In the expanded entity list from `_get_expanded_entities` we will very likely have generic entities
    which comes from asset metadata. These entities may have counterparts given by users. We will try to update the
    non-generic entities with the metadata contained within generic ones.
    For example `entities = [my_mesh["*"], user_defined_zone]`. We need to keep the `user_defined_zone` while updating
    it with the boundaries coming from mesh metadata in expanded list.
    """
    all_entities = {}

    for entity in expanded_entities:
        if entity.name not in all_entities:
            all_entities[entity.name] = []
        all_entities[entity.name].append(entity)

    for name, entity_list in all_entities.items():
        if len(entity_list) > 1:
            # step 1: find one instance that is non-generic if any
            for base_index, entity in enumerate(entity_list):
                if entity._is_generic() == False:
                    break
            for index, entity in enumerate(entity_list):
                if index == base_index:
                    continue  # no merging into self
                print("Base: ", entity_list[base_index])
                entity_list[base_index] = _merge_objects(entity_list[base_index], entity)
                entity_list.remove(entity)

        if len(entity_list) != 1:
            error_message = f"Duplicate entities found for {name}."
            for entity in entity_list:
                error_message += f"\n{entity}\n"
            error_message += "Please remove duplicates."
            raise ValueError(error_message)
    return [entity_list[0] for entity_list in all_entities.values()]


class EntityList(Flow360BaseModel, metaclass=_EntityListMeta):
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

    stored_entities: Optional[List] = pd.Field()

    @classmethod
    def _get_valid_entity_types(cls):
        """Get the list of types that the entity list can accept."""
        entity_field_type = cls.__annotations__.get("stored_entities")

        if entity_field_type is not None:
            # Handle Optional[List[Union[xxxx]]]
            origin = get_origin(entity_field_type)
            if origin is Union:
                args = get_args(entity_field_type)
                for arg in args:
                    if get_origin(arg) is list:
                        entity_field_type = arg
                        break

            if hasattr(entity_field_type, "__origin__") and entity_field_type.__origin__ is list:
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
            if input is None:
                return dict(stored_entities=None)
            else:
                cls._valid_individual_input(input)
                if isinstance(input, tuple(valid_types)):
                    formated_input.append(input)
        return dict(stored_entities=formated_input)

    @pd.field_validator("stored_entities", mode="after")
    @classmethod
    def _check_duplicate_entity_in_list(cls, values):
        seen = []
        if values is None:
            return None
        for value in values:
            if value in seen:
                if isinstance(value, EntityBase):
                    log.warning(f"Duplicate entity found, name: {value.name}")
                else:
                    log.warning(f"Duplicate entity found: {value}")
                continue
            seen.append(value)
        return seen

    def _get_expanded_entities(
        self,
        supplied_registry=None,
        expect_supplied_registry: bool = True,
        create_hard_copy: bool = True,
    ) -> List[EntityBase]:
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
            Exapnded entities list.
        """

        entities = getattr(self, "stored_entities", [])

        if entities is None:
            return None

        valid_types = self.__class__._get_valid_entity_types()

        expanded_entities = []

        for entity in entities:
            if isinstance(entity, str):
                # Expand from supplied registry
                if supplied_registry is None:
                    if expect_supplied_registry == False:
                        continue
                    else:
                        raise ValueError(
                            f"Internal error, registry is not supplied for entity ({entity}) expansion."
                        )
                # Expand based on naming pattern registered in the Registry
                pattern_matched_entities = supplied_registry.find_by_naming_pattern(entity)
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
        if create_hard_copy == True:
            return copy.deepcopy(expanded_entities)
        else:
            return expanded_entities

    def preprocess(self, supplied_registry=None, **kwargs):
        """
        Expand and overwrite self.stored_entities in preparation for submissin/serialization.
        Should only be called as late as possible to incoperate all possible changes.
        """
        self.stored_entities = self._get_expanded_entities(supplied_registry)
        return super().preprocess(self, **kwargs)
