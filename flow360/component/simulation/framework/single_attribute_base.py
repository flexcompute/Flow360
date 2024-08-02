"""Single attribute base model."""

import abc
from typing import Any

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel


class SingleAttributeModel(Flow360BaseModel, metaclass=abc.ABCMeta):
    """Base class for single attribute models."""

    value: Any = pd.Field()

    # pylint: disable=unused-argument
    def __init__(self, value: Any = None, type_name=None):
        if value is None:
            raise ValueError(f"Value must be provided for {self.__class__.__name__}.")
        super().__init__(value=value)
