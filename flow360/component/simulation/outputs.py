from typing import Union

import pydantic as pd

from flow360.component.flow360_params.params_base import Flow360BaseModel


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
