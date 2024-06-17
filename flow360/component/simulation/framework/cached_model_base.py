import abc
import inspect
from functools import wraps
from typing import Any, Callable, Dict, Literal
from contextlib import contextmanager


import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.types import TYPE_TAG_STR



@contextmanager
def _model_attribute_unlock(model, attr: str):
    try:
        model.model_fields[attr].frozen = False
        yield
    finally:
        model.model_fields[attr].frozen = True



class MultiConstructorModelBase(Flow360BaseModel, metaclass=abc.ABCMeta):
    _constructor: Literal['default'] = pd.Field('default', frozen=True)

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
            result._cached = result.__annotations__["_cached"](
                **{**result._cached.model_dump(), **defaults, **kwargs}
            )
            with _model_attribute_unlock(result, '_constructor'):
                result._constructor = func.__name__
            return result

        return wrapper

    def __init__(self, **data):
        cached = data.pop("_cached", None)
        super().__init__(**data)
        if cached:
            try:
                self._cached = self.__annotations__["_cached"].model_validate(cached)
            except:
                pass
        else:
            defaults = {name: field.default for name, field in self.model_fields.items()}
            defaults.pop(TYPE_TAG_STR)
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
