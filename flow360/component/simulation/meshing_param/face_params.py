from typing import Literal

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel
from flow360.component.simulation.entities_base import EntitiesBase


class FaceRefinement(EntitiesBase):
    max_edge_length: pd.PositiveFloat = pd.Field()
    curvature_resolution_angle: pd.PositiveFloat = pd.Field()
    growth_rate: pd.PositiveFloat = pd.Field()
    first_layer_thickness: pd.PositiveFloat = pd.Field()
    type: Literal["aniso", "projectAnisoSpacing", "none"] = pd.Field()
