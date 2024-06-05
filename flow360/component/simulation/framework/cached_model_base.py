import abc
from typing import Any, Dict

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel


class CachedModelBase(Flow360BaseModel, metaclass=abc.ABCMeta):
    def __init__(self, **data):
        cached = data.pop("_cached", None)
        super().__init__(**data)
        if cached:
            try:
                self._cached = self.__annotations__["_cached"].model_validate(cached)
            except:
                pass

    @pd.model_serializer(mode="wrap")
    def serialize_model(self, handler) -> Dict[str, Any]:
        serialize_self = handler(self)
        serialize_self["_cached"] = self._cached.model_dump() if self._cached else None
        return serialize_self
