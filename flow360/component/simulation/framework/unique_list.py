"""Unique list classes for Simulation framework."""

from collections import Counter, OrderedDict
from copy import deepcopy
from typing import Annotated, Any, List, Union

import pydantic as pd

from flow360.component.flow360_params.flow360_fields import get_aliases
from flow360.component.simulation.framework.base_model import Flow360BaseModel


class _CombinedMeta(type(Flow360BaseModel), type):
    pass


class _UniqueListMeta(_CombinedMeta):
    def __getitem__(cls, item_types):
        """
        Creates a new class with the specified item types as a list.
        """
        if not isinstance(item_types, tuple):
            item_types = (item_types,)
        union_type = Union[item_types]
        annotations = {"items": List[union_type]}
        new_cls = type(
            f"{cls.__name__}[{','.join([t.__name__ if hasattr(t, '__name__') else repr(t) for t in item_types])}]",
            (cls,),
            {"__annotations__": annotations},
        )
        return new_cls


def _validate_unique_list(v: List) -> List:
    if len(v) != len(set(v)):
        raise ValueError(
            "Input item to this list must be unique "
            f"but {[str(item) for item, count in Counter(v).items() if count > 1]} appears multiple times."
        )
    return v


class UniqueItemList(Flow360BaseModel, metaclass=_UniqueListMeta):
    """
    A list of general type items that must be unique
    (uniqueness is determined by the item's __eq__ and __hash__ method).

    We will **not** try to remove duplicate items as choice is user's preference.
    """

    items: Annotated[List, {"uniqueItems": True}, pd.Field()]

    @pd.field_validator("items", mode="after")
    @classmethod
    def check_unique(cls, v):
        """Check if the items are unique after type checking"""
        return _validate_unique_list(v)

    @pd.model_validator(mode="before")
    @classmethod
    def _format_input_to_list(cls, input_data: Union[dict, list, Any]):
        if isinstance(input_data, list):
            return {"items": input_data}
        if isinstance(input_data, dict):
            # Add alias "stored_entities" so front end can monitor change in entities.
            if "items" not in input_data and "stored_entities" not in input_data:
                raise KeyError(
                    f"Invalid input to `entities` [UniqueItemList], dict {input_data} is missing the key 'items'."
                )
            if "items" in input_data:
                return {"items": input_data["items"]}
            return {"items": input_data["stored_entities"]}
        # Single reference to an entity
        return {"items": [input_data]}

    def append(self, obj):
        """Append an item to `UniqueItemList`."""
        items_copy = deepcopy(self.items)
        items_copy.append(obj)
        self.items = items_copy  # To trigger validation


def _validate_unique_aliased_item(v: List[str]) -> List[str]:
    deduplicated_list = []
    for item in v:
        if get_aliases(item)[1] not in deduplicated_list and item not in deduplicated_list:
            deduplicated_list.append(item)
    return deduplicated_list


class UniqueAliasedStringList(Flow360BaseModel, metaclass=_UniqueListMeta):
    """
    A list of items that must be unique by original name or by aliased name.
    Expect string only and we will remove the duplicate ones.
    """

    items: Annotated[List[str], {"uniqueItems": True}]

    @pd.field_validator("items", mode="after")
    @classmethod
    def deduplicate(cls, v):
        """Deduplicate the list by original name or aliased name."""
        return _validate_unique_aliased_item(v)

    @pd.model_validator(mode="before")
    @classmethod
    def _format_input_to_list(cls, input_data: Union[dict, list, str]):
        if isinstance(input_data, list):
            return {"items": input_data}
        if isinstance(input_data, dict):
            return {"items": input_data["items"]}
        return {"items": [input_data]}


class UniqueStringList(Flow360BaseModel):
    """
    A list of string that must be unique by original name or by aliased name.
    Expect string only and we will remove the duplicate ones.
    """

    items: List[str] = pd.Field([])

    @pd.model_validator(mode="before")
    @classmethod
    def _format_input_to_list(cls, input_data: Union[dict, list, str]):
        if isinstance(input_data, list):
            return {"items": input_data}
        if isinstance(input_data, dict):
            if input_data == {}:
                return {"items": []}
            return {"items": input_data["items"]}
        return {"items": [input_data]}

    @pd.field_validator("items", mode="after")
    @classmethod
    def ensure_unique(cls, v):
        """Deduplicate the list"""
        return list(OrderedDict.fromkeys(v))

    def append(self, obj):
        """Append an item to `UniqueStringList`."""
        items_copy = deepcopy(self.items)
        items_copy.append(obj)
        self.items = items_copy  # To trigger validation
