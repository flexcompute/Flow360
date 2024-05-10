from typing import List, Optional, Tuple, Union, Literal

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel
from flow360.component.simulation.entities_base import EntitiesBase

"""Mostly the same as Flow360Param counterparts.
Caveats:
1. Dicts like "Surfaces", "Slices" etc should be accepting entities instead of just string.

"""

SurfaceOutputFields = List[str]
VolumeOutputFields = List[str]
SliceOutputFields = List[str]
MonitorOutputFields = List[str]


class EntityWithOutput(EntitiesBase):
    output_fields = []


class Slice(EntityWithOutput):
    slice_normal: Tuple[float, float, float] = pd.Field()
    slice_origin: Tuple[float, float, float] = pd.Field()


class IsoSurface(EntityWithOutput):
    surface_field: str = pd.Field()
    surface_field_magnitude: float = pd.Field()


class SurfaceIntegralMonitor(EntitiesBase):
    type: Literal["surfaceIntegral"] = pd.Field("surfaceIntegral", frozen=True)
    surfaces: EntitiesBase = pd.Field()
    output_fields: Optional[MonitorOutputFields] = pd.Field(default=[])


class ProbeMonitor(EntitiesBase):
    type: Literal["probe"] = pd.Field("probe", frozen=True)
    monitor_locations: List[Tuple[float, float, float]] = pd.Field()
    output_fields: Optional[MonitorOutputFields] = pd.Field(default=[])


class SurfaceOutput(EntitiesBase):
    write_single_file: Optional[bool] = pd.Field(default=False)
    output_fields: Optional[SurfaceOutputFields] = pd.Field(default=[])


class VolumeOutput(EntitiesBase):
    output_fields: Optional[VolumeOutputFields] = pd.Field(default=[])
    volumes: Optional[List[EntityWithOutput]] = pd.Field()


class SliceOutput(EntitiesBase):
    output_fields: Optional[SliceOutputFields] = pd.Field(default=[])
    slices: List[Slice] = pd.Field()


class IsoSurfaceOutput(EntitiesBase):
    output_fields: Optional[SurfaceOutputFields] = pd.Field(default=[])


class MonitorOutput(EntitiesBase):
    output_fields: Optional[MonitorOutputFields] = pd.Field(default=[])


class AeroAcousticOutput(Flow360BaseModel):
    patch_type: Optional[str] = pd.Field("solid", frozen=True)
    observers: List[Tuple[float, float, float]] = pd.Field()
    write_per_surface_output: Optional[bool] = pd.Field()


class UserDefinedFields(Flow360BaseModel):
    pass


OutputTypes = Union[
    SurfaceOutput,
    VolumeOutput,
    SliceOutput,
    IsoSurfaceOutput,
    MonitorOutput,
    AeroAcousticOutput,
    UserDefinedFields,
]
