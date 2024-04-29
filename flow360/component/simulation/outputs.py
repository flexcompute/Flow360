from typing import Union

from flow360.component.simulation.base_model import Flow360BaseModel


class SurfaceOutput(Flow360BaseModel):
    pass


class VolumeOutput(Flow360BaseModel):
    pass


class SliceOutput(Flow360BaseModel):
    pass


class IsoSurfaceOutput(Flow360BaseModel):
    pass


class MonitorOutput(Flow360BaseModel):
    pass


class AeroAcousticOutput(Flow360BaseModel):
    pass


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
