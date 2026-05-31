"""Unique list classes for Simulation framework."""

from collections import OrderedDict
from copy import deepcopy
from typing import Annotated, Any, Union

import pydantic as pd

from flow360_schema.framework.base_model import Flow360BaseModel


class _CombinedMeta(type(Flow360BaseModel), type):  # type: ignore[misc]
    pass


class _UniqueListMeta(_CombinedMeta):
    def __getitem__(cls, item_types: type | tuple[type, ...]) -> type:
        """
        Creates a new class with the specified item types as a list.
        """
        if not isinstance(item_types, tuple):
            item_types = (item_types,)
        union_type = Union[item_types]  # type: ignore[valid-type]  # noqa: UP007 - runtime dynamic union construction
        annotations = {"items": list[union_type]}
        new_cls = type(
            f"{cls.__name__}[{','.join([t.__name__ if hasattr(t, '__name__') else repr(t) for t in item_types])}]",
            (cls,),
            {"__annotations__": annotations},
        )
        return new_cls


def _remove_duplicates(v: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    seen_add = seen.add
    return [x for x in v if not (x in seen or seen_add(x))]


class UniqueItemList(Flow360BaseModel, metaclass=_UniqueListMeta):  # type: ignore[metaclass]
    """
    A list of general type items that must be unique
    (uniqueness is determined by the item's __eq__ and __hash__ method).

    Duplicates present in the list will be removed.
    """

    items: Annotated[list[Any], {"uniqueItems": True}]

    @pd.field_validator("items", mode="after")
    @classmethod
    def check_unique(cls, v: list[Any]) -> list[Any]:
        """Check if the items are unique after type checking"""
        return _remove_duplicates(v)

    @pd.model_validator(mode="before")
    @classmethod
    def deserializer(cls, input_data: dict[str, Any] | list[Any] | Any) -> dict[str, list[Any]]:
        """Deserializer handling both JSON dict input as well as Python object input"""
        if isinstance(input_data, list):
            return {"items": input_data}
        if isinstance(input_data, dict):
            if "items" not in input_data:
                raise KeyError(
                    f"Invalid input to `entities` [UniqueItemList], dict {input_data} is missing the key 'items'."
                )
            return {"items": input_data["items"]}
        # Single reference to an entity
        return {"items": [input_data]}

    def append(self, obj: Any) -> None:
        """Append an item to `UniqueItemList`."""
        items_copy = deepcopy(self.items)
        items_copy.append(obj)
        self.items = items_copy  # To trigger validation


class UniqueStringList(Flow360BaseModel):
    """
    A list of string that must be unique by original name or by aliased name.
    Expect string only and we will remove the duplicate ones.
    """

    items: list[str] = pd.Field([])

    @pd.model_validator(mode="before")
    @classmethod
    def deserializer(cls, input_data: dict[str, Any] | list[Any] | str) -> dict[str, list[Any]]:
        """Deserializer handling both JSON dict input as well as Python object input"""
        if isinstance(input_data, list):
            return {"items": input_data}
        if isinstance(input_data, dict):
            if input_data == {}:
                return {"items": []}
            return {"items": input_data["items"]}
        return {"items": [input_data]}

    @pd.field_validator("items", mode="after")
    @classmethod
    def ensure_unique(cls, v: list[str]) -> list[str]:
        """Deduplicate the list"""
        return list(OrderedDict.fromkeys(v))

    def append(self, obj: str) -> None:
        """Append an item to `UniqueStringList`."""
        items_copy = deepcopy(self.items)
        items_copy.append(obj)
        self.items = items_copy  # To trigger validation
