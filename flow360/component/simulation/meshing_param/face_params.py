from typing import Literal

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Face


class FaceRefinement(Flow360BaseModel):
    entities: EntityList[Face] = pd.Field(alias="faces")
    type: Literal["aniso", "projectAnisoSpacing", "none"] = pd.Field()
    max_edge_length: pd.PositiveFloat = pd.Field()
    first_layer_thickness: pd.PositiveFloat = pd.Field()
