from typing import Tuple

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel

##:: Geometrical Volume ::##


class Box(Flow360BaseModel):
    center: Tuple[float, float] = pd.Field()
    size: Tuple[float, float, float] = pd.Field()


class Cylinder(Flow360BaseModel):
    axis: Tuple[float, float, float] = pd.Field()
    center: Tuple[float, float, float] = pd.Field()
    height: float = pd.Field()
    inner_radius: pd.PositiveFloat = pd.Field()
    outer_radius: pd.PositiveFloat = pd.Field()
