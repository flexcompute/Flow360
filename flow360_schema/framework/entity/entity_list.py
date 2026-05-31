"""EntityList: typed, validating container for entity collections."""

import logging
from typing import Annotated, Any, Union, get_args, get_origin

import pydantic as pd

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_selector import EntitySelector
from flow360_schema.framework.validation.validators import contextual_model_validator

logger = logging.getLogger(__name__)


def _validate_selector_type(selector: EntitySelector, valid_type_names: list[str]) -> EntitySelector:
    """Ensure a selector targets one of the EntityList's allowed types."""
    if selector.target_class not in valid_type_names:
        raise ValueError(
            f"Selector target_class ({selector.target_class}) is incompatible "
            f"with EntityList types {valid_type_names}."
        )
    return selector


def _ensure_is_entity(entity: EntityBase | Any) -> EntityBase:
    """Ensure the input is an entity instance."""
    if isinstance(entity, EntityBase):
        return entity

    raise ValueError(f"Type({type(entity)}) of input to `entities` ({entity}) is not valid. Expected entity instance.")


def _get_valid_entity_type_info(entity_list_cls: type["EntityList"]) -> tuple[list[type], set[str]]:
    """Return valid EntityList types together with their discriminator names."""
    valid_types = entity_list_cls._get_valid_entity_types()
    valid_type_names: set[str] = set()
    for valid_type in valid_types:
        field = valid_type.model_fields["private_attribute_entity_type_name"]  # type: ignore[attr-defined]
        valid_type_name = field.default
        if not isinstance(valid_type_name, str):
            raise TypeError(
                f"EntityList valid type {valid_type.__name__} must define a string "
                "`private_attribute_entity_type_name` default."
            )
        valid_type_names.add(valid_type_name)
    return valid_types, valid_type_names


def _filter_entities_by_valid_types(
    entity_list_cls: type["EntityList"],
    entities: list[EntityBase],
) -> list[EntityBase]:
    """Filter entity objects using the allowed types declared on a specific EntityList class."""
    if not entities:
        return entities

    valid_types, valid_type_names = _get_valid_entity_type_info(entity_list_cls)
    filtered_entities = []
    for entity in entities:
        entity_type_name = entity.private_attribute_entity_type_name
        if entity_type_name in valid_type_names:
            filtered_entities.append(entity)
        else:
            logger.debug(
                "Entity '%s' (type=%s) filtered out: not in EntityList valid types %s",
                getattr(entity, "name", "<unknown>"),
                entity_type_name,
                valid_type_names,
            )

    return filtered_entities


class _CombinedMeta(type(Flow360BaseModel), type):  # type: ignore[misc]
    pass


class _EntityListMeta(_CombinedMeta):
    def __getitem__(cls, entity_types: type | tuple[type, ...]) -> type:
        """
        Creates a new class with the specified entity types as a list of stored entities.
        """
        if not isinstance(entity_types, tuple):
            entity_types = (entity_types,)
        union_type = Annotated[
            Union[entity_types], pd.Field(discriminator="private_attribute_entity_type_name")  # type: ignore[valid-type]  # noqa: UP007
        ]
        annotations = {
            "stored_entities": list[union_type]
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


class EntityList(Flow360BaseModel, metaclass=_EntityListMeta):  # type: ignore[metaclass]
    """
    The type accepting a list of entities or selectors.

    Attributes:
        stored_entities (List[Union[EntityBase, Tuple[str, registry]]]): List of stored entities, which can be
            instances of `Box`, `Cylinder`, or strings representing naming patterns.
    """

    stored_entities: list[Any] = pd.Field(
        description="List of manually picked entities in addition to the ones selected by selectors."
    )
    selectors: list[EntitySelector] | None = pd.Field(
        None, description="Selectors on persistent entities for rule-based selection."
    )

    @pd.field_validator("stored_entities", mode="before")
    @classmethod
    def _filter_stored_entities(cls, value: list[Any]) -> list[Any]:
        """
        Filter explicit ``stored_entities`` before discriminator validation runs.

        This ``before`` validator is intentionally kept to preserve EntityList's current
        behavior for explicit entity inputs. Without it, unsupported entity objects inside
        ``stored_entities`` would fail immediately during Pydantic discriminator validation
        instead of being ignored by EntityList's explicit-input filtering semantics.
        """
        if not value:
            return value

        stored_entities = [item for item in value if isinstance(item, EntityBase)]
        if not stored_entities:
            return value

        filtered_entities = _filter_entities_by_valid_types(cls, stored_entities)
        if stored_entities and not filtered_entities:
            valid_types, _ = _get_valid_entity_type_info(cls)
            valid_type_name_list = [valid_type.__name__ for valid_type in valid_types]
            raise ValueError(f"Can not find any valid entity of type {valid_type_name_list} from the input.")
        filtered_iter = iter(filtered_entities)
        next_valid_entity = next(filtered_iter, None)
        result = []
        for item in value:
            if not isinstance(item, EntityBase):
                result.append(item)
                continue
            if item is next_valid_entity:
                result.append(item)
                next_valid_entity = next(filtered_iter, None)
        return result

    @contextual_model_validator(mode="after")
    def _ensure_entities_after_expansion(self, param_info: Any) -> "EntityList":
        """
        Ensure the EntityList resolves to at least one valid entity in the current context.

        Explicit ``stored_entities`` are cleaned first via per-entity manual-assignment
        validation. If selectors are present, resolution then expands selectors against the
        cleaned explicit entities and validates the resolved result as a whole.
        """
        # Clean explicit manual assignments first; selector-resolved entities do not use this hook.
        manual_assignments: list[EntityBase] = self.stored_entities
        filtered_assignments = [
            item for item in manual_assignments if item._manual_assignment_validation(param_info) is not None
        ]
        # Bypass validate_on_assignment to avoid recursive validation while updating the cleaned list.
        object.__setattr__(
            self,
            "stored_entities",
            filtered_assignments,
        )

        # Selector resolution starts from the cleaned explicit assignments.
        resolved_entities: list[EntityBase] = filtered_assignments
        if self.selectors:
            resolved_entities = param_info.expand_entity_list(self)

        if not resolved_entities:
            raise ValueError("No entities were selected.")

        # Contextual per-entity checks run on the final resolved entity set.
        for item in resolved_entities:
            item._per_entity_type_validation(param_info)
        return self

    @classmethod
    def _get_valid_entity_types(cls) -> list[type]:
        entity_field_type = cls.__annotations__.get("stored_entities")

        if entity_field_type is None:
            raise TypeError("Internal error, the metaclass for EntityList is not properly set.")

        # Handle List[...]
        origin: object = get_origin(entity_field_type)
        if origin is not list:
            # Not a List, handle other cases or raise an error
            raise TypeError("Expected 'stored_entities' to be a List.")
        inner_type = get_args(entity_field_type)[0]  # Extract the inner type

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
    def _build_result(cls, entities_to_store: list[Any], entity_selectors_to_store: list[Any]) -> dict[str, Any]:
        """Build the final result dictionary."""
        return {
            "stored_entities": entities_to_store,
            "selectors": entity_selectors_to_store if entity_selectors_to_store else None,
        }

    @classmethod
    def _process_single_item(
        cls,
        item: EntityBase | EntitySelector,
        valid_type_names: list[str],
        entities_to_store: list[EntityBase],
        entity_selectors_to_store: list[Any],
    ) -> None:
        """Process a single item (entity or selector) and add to appropriate storage lists."""
        if isinstance(item, EntitySelector):
            entity_selectors_to_store.append(_validate_selector_type(item, valid_type_names))
        else:
            processed_entity = _ensure_is_entity(item)
            entities_to_store.append(processed_entity)

    @pd.model_validator(mode="before")
    @classmethod
    def deserializer(
        cls, input_data: dict[str, Any] | list[Any] | EntityBase | EntitySelector | None
    ) -> dict[str, Any]:
        """
        Flatten List[EntityBase] and put into stored_entities.

        Note: explicit ``stored_entities`` are pre-filtered by ``_filter_stored_entities``
        before discriminator validation. Selector results are filtered later during
        selector expansion.
        """
        entities_to_store: list[Any] = []
        entity_selectors_to_store: list[Any] = []
        valid_types = tuple(cls._get_valid_entity_types())
        valid_type_names = [t.__name__ for t in valid_types]

        if input_data is None:
            raise ValueError("None is not a valid input to `entities`.")
        if isinstance(input_data, list):
            # -- User input mode. --
            # List content might be entity Python objects or selector Python objects
            if input_data == []:
                raise ValueError("Invalid input type to `entities`, list is empty.")
            for item in input_data:
                if isinstance(item, list):  # Nested list comes from assets __getitem__
                    # Process all entities without filtering
                    processed_entities = [_ensure_is_entity(individual) for individual in item]
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
