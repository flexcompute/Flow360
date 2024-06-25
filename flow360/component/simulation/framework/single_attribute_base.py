"""Single attribute base model."""

import abc
from typing import Any

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel


class SingleAttributeModel(Flow360BaseModel, metaclass=abc.ABCMeta):
    """Base class for single attribute models."""

    value: Any = pd.Field()

    def __init__(self, value: Any):
        super().__init__(value=value)
