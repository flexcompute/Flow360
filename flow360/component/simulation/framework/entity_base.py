"""Base classes for entity types."""

from __future__ import annotations

import copy
import uuid
from abc import ABCMeta
from numbers import Number
from typing import Annotated, List, Optional, Union, get_args, get_origin

import numpy as np
import pydantic as pd
import unyt

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.utils import is_exact_instance


class MergeConflictError(Exception):
    """Raised when a merge conflict is detected."""


def generate_uuid():
    """generate a unique identifier for non-persistent entities. Required by front end."""
    return str(uuid.uuid4())


class EntityBase(Flow360BaseModel, metaclass=ABCMeta):
    """
    Base class for dynamic entity types.

    Attributes:
        private_attribute_registry_bucket_name (str):
            A string representing the specific type of the entity.
            This should be set in subclasses to differentiate between entity types.
            Warning:
            This controls the granularity of the registry and must be unique for each entity type
            and it is **strongly recommended NOT** to change it as it will bring up compatability problems.

        name (str):
            The name of the entity instance, used for identification and retrieval.
    """

    private_attribute_registry_bucket_name: str = "Invalid"
    private_attribute_entity_type_name: str = "Invalid"
    private_attribute_id: Optional[str] = pd.Field(
        None,
        frozen=True,
        description="Unique identifier for the entity. Used by front end to track entities and enable auto update etc.",
    )

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
                "Change is necessary when calling .copy() as there cannot be two identical entities at the same time. "
                "Please use update parameter to change the entity attributes."
            )
        if "name" not in update or update["name"] == self.name:
            raise ValueError(
                "Copying an entity requires a new name to be specified. "
                "Please provide a new name in the update dictionary."
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
        """returns the bucket to which the entity belongs."""
        return self.private_attribute_registry_bucket_name

    @entity_bucket.setter
    def entity_bucket(self, value: str):
        """disallow modification of the bucket to which the entity belongs."""
        raise AttributeError("Cannot modify the bucket to which the entity belongs.")

    @property
    def entity_type(self) -> str:
        """returns the entity class name."""
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
        union_type = Annotated[
            Union[entity_types], pd.Field(discriminator="private_attribute_entity_type_name")
        ]
        annotations = {
            "stored_entities": List[union_type]
        }  # Make sure it is somewhat consistent with the EntityList class
        new_cls = type(
            f"{cls.__name__}[{','.join([t.__name__ for t in entity_types])}]",
            (cls,),
            {"__annotations__": annotations},
        )
        # Note:
        # Printing the stored_entities's discriminator will be None but
        # that FieldInfo->discriminator seems to be just for show.
        # It seems Pydantic use the discriminator inside the annotation
        # instead so the above should trigger the discrimination during deserialization.
        return new_cls


def __combine_bools(input_data):
    # If the input is a single boolean, return it directly
    if isinstance(input_data, bool):
        return input_data
    # If the input is a numpy ndarray, flatten it
    if isinstance(input_data, np.ndarray):
        input_data = input_data.ravel()
    # If the input is not a boolean or an ndarray, assume it's an iterable of booleans
    return all(input_data)


def _merge_fields(field_old, field_new):
    """
    Merges fields from a new object (field_new) into an existing object (field_old) with higher priority.
    It recursively handles nested objects.

    Parameters:
        field_old (EntityBase): The existing object with higher priority.
        field_new (EntityBase): The new object with lower priority.

    Returns:
        EntityBase: The merged object.

    Raises:
        MergeConflictError: If there's a conflict between the objects that can't be resolved.

    Note:
        The function skips merging for 'private_attribute_entity_type_name' and 'private_attribute_registry_bucket_name'
        to handle cases where the objects could be different classes in nature.
    """
    # Define basic types that should not be merged further but directly compared or replaced
    basic_types = (list, Number, str, tuple, unyt.unyt_array, unyt.unyt_quantity)

    # Iterate over all attributes and values of the new object
    for attr, value in field_new.__dict__.items():
        # Ignore certain private attributes meant for entity type definition
        if attr in [
            "private_attribute_entity_type_name",
            "private_attribute_registry_bucket_name",
            "private_attribute_id",
        ]:
            continue

        if field_new.__dict__[attr] is None:
            # Skip merging this attribute since we always want more info.
            continue

        if attr in field_old.__dict__:
            # Check for conflicts between old and new values
            found_conflict = __combine_bools(field_old.__dict__[attr] != value)
            if found_conflict:
                if (
                    not isinstance(field_old.__dict__[attr], basic_types)
                    and field_old.__dict__[attr] is not None
                ):
                    # Recursive merge for nested objects until reaching basic types
                    field_old.__dict__[attr] = _merge_fields(
                        field_old.__dict__[attr], field_new.__dict__[attr]
                    )
                elif field_old.__dict__[attr] is None or (
                    isinstance(field_old.__dict__[attr], list) and not field_old.__dict__[attr]
                ):
                    # Set the old value to the new value if the old value is empty
                    field_old.__dict__[attr] = value
                else:
                    # Raise an error if basic types conflict and cannot be resolved
                    raise MergeConflictError(
                        f"Conflict on attribute '{attr}': {field_old.__dict__[attr]} != {value}"
                    )
        else:
            # Add new attributes from the new object to the old object
            field_old.__dict__[attr] = value

    return field_old


def _merge_objects(obj_old: EntityBase, obj_new: EntityBase) -> EntityBase:
    """
    Merges two entity objects, prioritizing the fields of the original object (obj_old) unless overridden
    by non-generic values in the new object (obj_new). This function ensures that only compatible entities
    are merged, raising exceptions when merge criteria are not met.

    Parameters:
        obj_old (EntityBase): The original object to be preserved with higher priority.
        obj_new (EntityBase): The new object that might override or add to the fields of the original object.

    Returns:
        EntityBase: The merged entity object.

    Raises:
        MergeConflictError: Raised when the entities have different names or when they are of different types
                            and both are non-generic, indicating that they should not be merged.
    """

    # Check if the names of the entities are the same; they must be for merging to make sense
    if obj_new.name != obj_old.name:
        raise MergeConflictError(
            "Merge operation halted as the names of the entities do not match."
            "Please ensure you are merging the correct entities."
        )

    # Swap objects if the new object is non-generic and the old one is generic.
    # This ensures that the `obj_old` always has priority in attribute preservation.
    # pylint: disable=protected-access
    if not obj_new._is_generic() and obj_old._is_generic():
        obj_new, obj_old = obj_old, obj_new

    # Ensure that objects are of the same type if both are non-generic to prevent merging unrelated types
    if not obj_new._is_generic() and not obj_old._is_generic():
        if obj_new.__class__ != obj_old.__class__:
            raise MergeConflictError(
                f"Cannot merge objects of different classes: {obj_old.__class__.__name__}"
                f" and {obj_new.__class__.__name__}."
            )

    # Utilize the _merge_fields function to merge the attributes of the two objects
    merged_object = _merge_fields(obj_old, obj_new)
    return merged_object


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
            base_index = 0
            for base_index, entity in enumerate(entity_list):
                # pylint: disable=protected-access
                if entity._is_generic() is False:
                    break
            for index, entity in enumerate(entity_list):
                if index == base_index:
                    continue  # no merging into self
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

    stored_entities: List = pd.Field()

    @classmethod
    def _get_valid_entity_types(cls):
        entity_field_type = cls.__annotations__.get("stored_entities")

        if entity_field_type is None:
            raise TypeError("Internal error, the metaclass for EntityList is not properly set.")

        # Handle List[...]
        if get_origin(entity_field_type) in (list, List):
            inner_type = get_args(entity_field_type)[0]  # Extract the inner type
        else:
            # Not a List, handle other cases or raise an error
            raise TypeError("Expected 'stored_entities' to be a List.")

        # Handle Annotated[...]
        if get_origin(inner_type) is Annotated:
            annotated_args = get_args(inner_type)
            if len(annotated_args) > 0:
                actual_type = annotated_args[0]  # The actual type inside Annotated
            else:
                raise TypeError("Annotated type has no arguments.")
        else:
            actual_type = inner_type

        # Handle Union[...]
        if get_origin(actual_type) is Union:
            valid_types = [arg for arg in get_args(actual_type) if isinstance(arg, type)]
            return valid_types
        if isinstance(actual_type, type):
            return [actual_type]
        raise TypeError("Cannot extract valid entity types.")

    @classmethod
    def _valid_individual_input(cls, input_data):
        """Validate each individual element in a list or as standalone entity."""
        if isinstance(input_data, (str, EntityBase)):
            return input_data

        raise ValueError(
            f"Type({type(input_data)}) of input to `entities` ({input_data}) is not valid. "
            "Expected str or entity instance."
        )

    @pd.model_validator(mode="before")
    @classmethod
    def _format_input_to_list(cls, input_data: Union[dict, list]):
        """
        Flatten List[EntityBase] and put into stored_entities.
        """
        entities_to_store = []
        valid_types = cls._get_valid_entity_types()

        if isinstance(input_data, list):  # A list of entities
            if input_data == []:
                raise ValueError("Invalid input type to `entities`, list is empty.")
            for item in input_data:
                if isinstance(item, list):  # Nested list comes from assets
                    _ = [cls._valid_individual_input(individual) for individual in item]
                    # pylint: disable=fixme
                    # TODO: Give notice when some of the entities are not selected due to `valid_types`?
                    entities_to_store.extend(
                        [
                            individual
                            for individual in item
                            if is_exact_instance(individual, tuple(valid_types))
                        ]
                    )
                else:
                    cls._valid_individual_input(item)
                    if is_exact_instance(item, tuple(valid_types)):
                        entities_to_store.append(item)
        elif isinstance(input_data, dict):  # Deseralization
            if "stored_entities" not in input_data:
                raise KeyError(
                    f"Invalid input type to `entities`, dict {input_data} is missing the key 'stored_entities'."
                )
            return {"stored_entities": input_data["stored_entities"]}
        # pylint: disable=no-else-return
        else:  # Single entity
            if input_data is None:
                return {"stored_entities": None}
            else:
                cls._valid_individual_input(input_data)
                if is_exact_instance(input_data, tuple(valid_types)):
                    entities_to_store.append(input_data)

        if not entities_to_store:
            raise ValueError(
                f"Can not find any valid entity of type {[valid_type.__name__ for valid_type in valid_types]}"
                f" from the input."
            )

        return {"stored_entities": entities_to_store}

    def _get_expanded_entities(
        self,
        *,
        create_hard_copy: bool,
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

        expanded_entities = []

        # pylint: disable=not-an-iterable
        for entity in entities:
            if entity not in expanded_entities:
                # Direct entity references are simply appended if they are of a valid type
                expanded_entities.append(entity)

        expanded_entities = _remove_duplicate_entities(expanded_entities)

        if expanded_entities == []:
            raise ValueError(
                f"Failed to find any matching entity with {entities}. Please check the input to entities."
            )
        # pylint: disable=fixme
        # TODO: As suggested by Runda. We better prompt user what entities are actually used/expanded to
        # TODO: avoid user input error. We need a switch to turn it on or off.
        if create_hard_copy is True:
            return copy.deepcopy(expanded_entities)
        return expanded_entities

    # pylint: disable=arguments-differ
    def preprocess(self, **kwargs):
        """
        Expand and overwrite self.stored_entities in preparation for submissin/serialization.
        Should only be called as late as possible to incoperate all possible changes.
        """
        self.stored_entities = self._get_expanded_entities(create_hard_copy=False)
        return super().preprocess(**kwargs)
