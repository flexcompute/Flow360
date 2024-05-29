from typing import Tuple

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel

##:: Geometrical Volume ::##


class ZoneBase(Flow360BaseModel):
    name: str = pd.Field()


class BoxZone(ZoneBase):
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    pass


class CylindricalZone(ZoneBase):
    axis: Tuple[float, float, float]
    center: Tuple[float, float, float]
    height: float
