import abc
from typing import Any

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel


class SingleAttributeModel(Flow360BaseModel, metaclass=abc.ABCMeta):
    value: Any = pd.Field()

    def __init__(self, value: Any, **kwargs):
        super().__init__(value=value, **kwargs)