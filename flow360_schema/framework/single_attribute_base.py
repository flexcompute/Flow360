"""Single-attribute base model for simple wrapper models."""

from __future__ import annotations

import abc
from typing import Any

import pydantic as pd

from .base_model import Flow360BaseModel


class SingleAttributeModel(Flow360BaseModel, metaclass=abc.ABCMeta):
    """Base class for models that wrap a single ``value`` field."""

    value: Any = pd.Field()

    def __init__(self, value: Any = None, type_name: Any = None) -> None:
        if value is None:
            raise ValueError(f"Value must be provided for {self.__class__.__name__}.")
        model_data = {"value": value}
        if type_name is not None:
            model_data["type_name"] = type_name
        super().__init__(**model_data)
