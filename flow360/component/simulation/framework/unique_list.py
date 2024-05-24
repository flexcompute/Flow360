from collections import Counter
from typing import Annotated, List, Union

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
            f"Input item to this list must be unique but {[str(item) for item, count in Counter(v).items() if count > 1]} appears multiple times."
        )
    return v


class UniqueItemList(Flow360BaseModel, metaclass=_UniqueListMeta):
    """
    A list of general type items that must be unique (uniqueness is determined by the item's __eq__ and __hash__ method).

    We will **not** try to remove duplicate items as choice is user's preference.
    """

    items: Annotated[List, {"uniqueItems": True}]

    @pd.field_validator("items", mode="after")
    def check_unique(cls, v):
        """Check if the items are unique after type checking"""
        return _validate_unique_list(v)

    @pd.model_validator(mode="before")
    @classmethod
    def _format_input_to_list(cls, input: Union[dict, list]):
        if isinstance(input, list):
            return dict(items=input)
        elif isinstance(input, dict):
            return dict(items=input["items"])
        else:  # Single reference to an entity
            return dict(items=[input])


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
    def deduplicate(cls, v):
        # for item in v:
        #     if isinstance(item, str) == False:
        #         raise ValueError(f"Expected string for the list but got {item.__class__.__name__}.")
        return _validate_unique_aliased_item(v)

    @pd.model_validator(mode="before")
    @classmethod
    def _format_input_to_list(cls, input: Union[dict, list]):
        if isinstance(input, list):
            return dict(items=input)
        elif isinstance(input, dict):
            return dict(items=input["items"])
        else:  # Single reference to an entity
            return dict(items=[input])
