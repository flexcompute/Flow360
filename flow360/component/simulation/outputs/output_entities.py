from typing import List, Literal, Tuple, final

import pydantic as pd

from flow360.component.simulation.framework.entity_base import EntityBase, EntityList
from flow360.component.simulation.primitives import Surface


@final
class Slice(EntityBase):
    _entity_type: Literal["SliceType"] = "SliceType"
    slice_normal: Tuple[float, float, float] = pd.Field()
    slice_origin: Tuple[float, float, float] = pd.Field()


@final
class Isosurface(EntityBase):
    _entity_type: Literal["IsosurfaceType"] = "IsosurfaceType"
    surface_field: str = pd.Field()
    # TODO: Maybe we need some unit helper function to help user figure out what is the value to use here?
    surface_field_magnitude: float = pd.Field(description="Expect scaled value.")


@final
class SurfaceIntegralMonitor(EntityBase):
    _entity_type: Literal["SurfaceIntegralMonitorType"] = "SurfaceIntegralMonitorType"
    type: Literal["surfaceIntegral"] = pd.Field("surfaceIntegral", frozen=True)
    entities: EntityList[Surface] = pd.Field(alias="surfaces")


@final
class ProbeMonitor(EntityBase):
    _entity_type: Literal["ProbeMonitorType"] = "ProbeMonitorType"
    type: Literal["probe"] = pd.Field("probe", frozen=True)
    locations: List[Tuple[float, float, float]] = pd.Field()
