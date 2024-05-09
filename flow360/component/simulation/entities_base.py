from typing import List

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel


class EntitiesBase(Flow360BaseModel):
    """Abstraction of `entities` implementation"""

    entities: List[str] = pd.Field()
