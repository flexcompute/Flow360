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
            f"{cls.__name__}[{','.join([t.__name__ for t in item_types])}]",
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


def _validate_unique_aliased_item(v: List[str]) -> List[str]:
    for item in v:
        if get_aliases(item)[1] in v:
            raise ValueError(
                f"Input item to this list must be unique but {item} and {get_aliases(item)[1]} are both present."
            )
    return v


class UniqueItemList(Flow360BaseModel, metaclass=_UniqueListMeta):
    items: Annotated[List, {"uniqueItems": True}]

    @pd.field_validator("items", mode="after")
    def check_unique(cls, v):
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


class UniqueAliasedItemList(UniqueItemList):
    @pd.field_validator("items", mode="after")
    def check_unique(cls, v):
        _validate_unique_aliased_item(v)
        return _validate_unique_list(v)
