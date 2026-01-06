"""Base classes for entity types."""

from __future__ import annotations

import hashlib
from abc import ABCMeta
from typing import Annotated, Any, List, Optional, Union, get_args, get_origin

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_selector import EntitySelector
from flow360.component.simulation.validation.validation_context import (
    ParamsValidationInfo,
    contextual_model_validator,
)
from flow360.log import log


class EntityBase(Flow360BaseModel, metaclass=ABCMeta):
    """
    Base class for dynamic entity types.

    Attributes:
        private_attribute_entity_type_name (str):
            A string representing the specific type of the entity.
            This should be set in subclasses to differentiate between entity types.

        name (str):
            The name of the entity instance, used for identification and retrieval.
    """

    private_attribute_entity_type_name: str = "Invalid"
    private_attribute_id: Optional[str] = pd.Field(
        # pylint: disable=fixme
        # TODO: This should not have default value. Everyone is supposed to set it.
        None,
        frozen=True,
        description="Unique identifier for the entity. Used by front end to track entities and enable auto update etc.",
    )

    name: str = pd.Field(frozen=True)

    # Whether the entity is dirty and needs to be re-hashed
    _dirty: bool = pd.PrivateAttr(True)
    # Cached hash of the entity
    _hash_cache: str = pd.PrivateAttr(None)

    def __init_subclass__(cls, **kwargs):  # type: ignore[override]
        """Validate required class-level attributes at subclass creation time.

        This avoids per-instance checks and catches misconfigured subclasses early.

        Rules:
        - If a subclass explicitly defines `private_attribute_entity_type_name` in its own
          class body, it must also be non-"Invalid". Intermediate abstract bases that do not
          set an entity type are allowed.
        """
        super().__init_subclass__(**kwargs)
        if cls is EntityBase:
            return

        # Only enforce entity type when the subclass explicitly sets it.
        if "private_attribute_entity_type_name" in cls.__dict__:
            # entity_type remains a Pydantic field
            def _resolve_field_default(field_name: str):
                for base in cls.__mro__:
                    if field_name in getattr(base, "__dict__", {}):
                        raw_value = base.__dict__[field_name]
                        return getattr(raw_value, "default", raw_value)
                model_fields = getattr(cls, "model_fields", None)
                if isinstance(model_fields, dict) and field_name in model_fields:
                    field_info = model_fields[field_name]
                    return getattr(field_info, "default", None)
                return None

            type_value = _resolve_field_default("private_attribute_entity_type_name")
            if type_value is None or type_value == "Invalid":
                raise NotImplementedError(
                    f"private_attribute_entity_type_name is not defined in the entity class: {cls.__name__}."
                )

    def copy(self, update: dict, **kwargs) -> EntityBase:  # pylint:disable=signature-differs
        """
        Creates a copy of the entity with compulsory updates.

        Parameters:
            update: A dictionary containing the updated attributes to apply to the copied entity.
            **kwargs: Additional arguments to pass to the copy constructor.

        Returns:
            A copy of the entity with the specified updates.
        """
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
    def entity_type(self) -> str:
        """returns the entity class name."""
        return self.private_attribute_entity_type_name

    @entity_type.setter
    def entity_type(self, value: str):
        raise AttributeError("Cannot modify the name of entity class.")

    def __str__(self) -> str:
        return "\n".join([f"        {attr}: {value}" for attr, value in self.__dict__.items()])

    def _recompute_hash(self):
        new_hash = hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()
        # Can further speed up 10% by using `object.__setattr__`
        self._hash_cache = new_hash
        self._dirty = False
        return new_hash

    def _get_hash(self):
        """hash generator to identify if two entities are the same"""
        # Can further speed up 10% by using `object.__getattribute__`
        dirty = self._dirty
        cache = self._hash_cache
        if dirty or cache is None:
            return self._recompute_hash()
        return cache

    def __setattr__(self, name, value):
        """
        [Large model performance]
        Wrapping the __setattr__ to mark the entity as dirty when the attribute is not private
        This enables caching the hash of the entity to avoid re-calculating the hash when the entity is not changed.
        """

        super().__setattr__(name, value)
        if not name.startswith("_") and not self._dirty:
            # Not using self to avoid invoking
            # Can further speed up 10% by using `object.__setattr__`
            self._dirty = True

    @property
    def id(self) -> str:
        """Returns private_attribute_id of the entity."""
        return self.private_attribute_id

    def _manual_assignment_validation(self, _: ParamsValidationInfo) -> EntityBase:
        """
        Pre-expansion contextual validation for the entity.
        This handles validation for the entity manually assigned.
        """
        return self

    def _per_entity_type_validation(self, _: ParamsValidationInfo) -> EntityBase:
        """Contextual validation with validation logic bond with the specific entity type."""
        return self


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


class EntityList(Flow360BaseModel, metaclass=_EntityListMeta):
    """
    The type accepting a list of entities or selectors.

    Attributes:
        stored_entities (List[Union[EntityBase, Tuple[str, registry]]]): List of stored entities, which can be
            instances of `Box`, `Cylinder`, or strings representing naming patterns.
    """

    stored_entities: List = pd.Field(
        description="List of manually picked entities in addition to the ones selected by selectors."
    )
    selectors: Optional[List[EntitySelector]] = pd.Field(
        None, description="Selectors on persistent entities for rule-based selection."
    )

    @pd.field_validator("stored_entities", mode="before")
    @classmethod
    def _filter_entities_by_valid_types(cls, value):
        """
        Centralized entity type filtering.

        This validator runs before discriminator validation and filters out entities
        whose types are not in the EntityList's valid types. This centralizes filtering
        logic that was previously split between the deserializer and selector expansion.
        """
        if not value:
            return value

        # Extract valid entity type names from class annotation
        try:
            valid_types = cls._get_valid_entity_types()
            valid_type_names = set()
            for valid_type in valid_types:
                field = valid_type.model_fields.get("private_attribute_entity_type_name")
                if field and field.default:
                    valid_type_names.add(field.default)
        except (TypeError, AttributeError, KeyError):
            # If we can't extract valid types, skip filtering
            return value

        if not valid_type_names:
            return value

        # Filter entities to only include valid types
        filtered_entities = []
        entity_count = 0
        for entity in value:
            if not isinstance(entity, EntityBase):
                # Not an entity object, keep it (might be dict for deserialization)
                filtered_entities.append(entity)
                continue

            entity_count += 1
            entity_type_name = getattr(entity, "private_attribute_entity_type_name", None)
            if entity_type_name in valid_type_names:
                filtered_entities.append(entity)
            else:
                log.debug(
                    "Entity '%s' (type=%s) filtered out: not in EntityList valid types %s",
                    getattr(entity, "name", "<unknown>"),
                    entity_type_name,
                    valid_type_names,
                )

        # If all entity objects were filtered out, raise an error
        if entity_count > 0 and not any(isinstance(e, EntityBase) for e in filtered_entities):
            valid_type_name_list = [vt.__name__ for vt in valid_types]
            raise ValueError(
                f"Can not find any valid entity of type {valid_type_name_list} from the input."
            )

        return filtered_entities

    @contextual_model_validator(mode="after")
    def _ensure_entities_after_expansion(self, param_info: ParamsValidationInfo):
        """
        Ensure entity selections yielded at least one entity once selectors are expanded.

        With delayed selector expansion, stored_entities may be empty if only selectors
        are defined.
        """
        is_empty = True
        # If stored_entities already has entities (user manual assignment), validation passes
        manual_assignments: List[EntityBase] = self.stored_entities
        # pylint: disable=protected-access
        if manual_assignments:
            filtered_assignments = [
                item
                for item in manual_assignments
                if item._manual_assignment_validation(param_info) is not None
            ]
            # Use object.__setattr__ to bypass validate_on_assignment and avoid recursion
            object.__setattr__(
                self,
                "stored_entities",
                filtered_assignments,
            )

            for item in filtered_assignments:
                item._per_entity_type_validation(param_info)

            if filtered_assignments:
                is_empty = False

        # No stored_entities - check if selectors will produce any entities
        if self.selectors:
            expanded: List[EntityBase] = param_info.expand_entity_list(self)
            if expanded:
                for item in expanded:
                    item._per_entity_type_validation(param_info)
                # Known non-empty
                return self

        # Neither stored_entities nor selectors produced any entities
        if is_empty:
            raise ValueError("No entities were selected.")
        return self

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
    def _validate_selector(cls, selector: EntitySelector, valid_type_names: List[str]) -> dict:
        """Process and validate an EntitySelector object."""
        if selector.target_class not in valid_type_names:
            raise ValueError(
                f"Selector target_class ({selector.target_class}) is incompatible "
                f"with EntityList types {valid_type_names}."
            )
        return selector

    @classmethod
    def _validate_entity(cls, entity: Union[EntityBase, Any]) -> EntityBase:
        """Process and validate an entity object."""
        if isinstance(entity, EntityBase):
            return entity

        raise ValueError(
            f"Type({type(entity)}) of input to `entities` ({entity}) is not valid. "
            "Expected entity instance."
        )

    @classmethod
    def _build_result(
        cls, entities_to_store: List[EntityBase], entity_selectors_to_store: List[dict]
    ) -> dict:
        """Build the final result dictionary."""
        return {
            "stored_entities": entities_to_store,
            "selectors": entity_selectors_to_store if entity_selectors_to_store else None,
        }

    @classmethod
    # pylint: disable=too-many-arguments
    def _process_single_item(
        cls,
        item: Union[EntityBase, EntitySelector],
        valid_type_names: List[str],
        entities_to_store: List[EntityBase],
        entity_selectors_to_store: List[dict],
    ) -> None:
        """Process a single item (entity or selector) and add to appropriate storage lists."""
        if isinstance(item, EntitySelector):
            entity_selectors_to_store.append(cls._validate_selector(item, valid_type_names))
        else:
            processed_entity = cls._validate_entity(item)
            entities_to_store.append(processed_entity)

    @pd.model_validator(mode="before")
    @classmethod
    def deserializer(cls, input_data: Union[dict, list, EntityBase, EntitySelector]):
        """
        Flatten List[EntityBase] and put into stored_entities.

        Note: Type filtering is now handled by the _filter_entities_by_valid_types
        field validator, which runs after deserialization but before discriminator validation.
        """
        entities_to_store = []
        entity_selectors_to_store = []
        valid_types = tuple(cls._get_valid_entity_types())
        valid_type_names = [t.__name__ for t in valid_types]

        if isinstance(input_data, list):
            # -- User input mode. --
            # List content might be entity Python objects or selector Python objects
            if input_data == []:
                raise ValueError("Invalid input type to `entities`, list is empty.")
            for item in input_data:
                if isinstance(item, list):  # Nested list comes from assets __getitem__
                    # Process all entities without filtering
                    processed_entities = [cls._validate_entity(individual) for individual in item]
                    entities_to_store.extend(processed_entities)
                else:
                    # Single entity or selector
                    cls._process_single_item(
                        item,
                        valid_type_names,
                        entities_to_store,
                        entity_selectors_to_store,
                    )
        elif isinstance(input_data, dict):  # Deserialization
            # With delayed selector expansion, stored_entities may be absent if only selectors are defined.
            # We allow empty stored_entities + empty/None selectors here - the model_validator
            # (_ensure_entities_after_expansion) will raise a proper error if no entities are selected.
            stored_entities = input_data.get("stored_entities", [])
            selectors = input_data.get("selectors", [])
            return cls._build_result(stored_entities, selectors)
        else:  # Single entity or selector
            if input_data is None:
                return cls._build_result(None, [])
            cls._process_single_item(
                input_data,
                valid_type_names,
                entities_to_store,
                entity_selectors_to_store,
            )

        if not entities_to_store and not entity_selectors_to_store:
            raise ValueError(
                f"Can not find any valid entity of type {[valid_type.__name__ for valid_type in valid_types]}"
                f" from the input."
            )

        return cls._build_result(entities_to_store, entity_selectors_to_store)
