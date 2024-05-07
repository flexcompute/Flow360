"""
Contains basically only boundary conditons for now. In future we can add new models like 2D equations.
"""

from typing import List, Literal, Optional, Tuple, Union

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel

##:: Geometrical surfaces ::##


class Patch(Flow360BaseModel):
    custom_name: Optional[str] = pd.Field("NoName")
    mesh_patch_name: Optional[str] = pd.Field("NoName")
    pass


##:: Physical surfaces (bounday) ::##


class Boundary(Flow360BaseModel):
    """Basic Boundary class"""

    _type: str
    entities: list[Patch] = pd.Field(None)


class NoSlipWall(Boundary):
    pass


class SlipWall(Boundary):
    pass


class RiemannInvariant(Boundary):
    pass


class FreestreamBoundary(Boundary):
    pass


class WallFunction(Boundary):
    pass


...

SurfaceTypes = Union[NoSlipWall, WallFunction, SlipWall, FreestreamBoundary]
