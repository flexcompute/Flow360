""" Base class for cached models. """

import abc
import inspect
from functools import wraps
from typing import Any, Callable, Dict

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel


# pylint: disable=missing-function-docstring
class CachedModelBase(Flow360BaseModel, metaclass=abc.ABCMeta):
    """Base class for cached models."""

    @classmethod
    def model_constructor(cls, func: Callable) -> Callable:
        @classmethod
        @wraps(func)
        def wrapper(cls, *args, **kwargs):
            sig = inspect.signature(func)
            result = func(cls, *args, **kwargs)
            defaults = {
                k: v.default
                for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty
            }
            # pylint: disable=protected-access
            result._cached = result.__annotations__["_cached"](
                **{**result._cached.model_dump(), **defaults, **kwargs}
            )
            # pylint: disable=protected-access
            result._cached.constructor = func.__name__
            return result

        return wrapper

    def __init__(self, **data):
        cached = data.pop("_cached", None)
        super().__init__(**data)
        if cached:
            try:
                self._cached = self.__annotations__["_cached"].model_validate(cached)
            except pd.ValidationError:
                pass
        else:
            defaults = {name: field.default for name, field in self.model_fields.items()}
            self._cached = self.__annotations__["_cached"](
                **{**defaults, **data}, constructor="default"
            )

    @pd.model_serializer(mode="wrap")
    def serialize_model(self, handler) -> Dict[str, Any]:
        serialize_self = handler(self)
        serialize_self["_cached"] = (
            self._cached.model_dump(exclude_none=True) if self._cached else None
        )
        return serialize_self
