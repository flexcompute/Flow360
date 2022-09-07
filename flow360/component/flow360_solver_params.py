"""
Flow360 solver parameters
"""
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Extra, Field


class MeshBoundary(BaseModel):
    """
    Mesh boundary
    """

    no_slip_walls: Optional[Union[List[str], List[int]]] = Field(alias="noSlipWalls")


class Flow360MeshParams(BaseModel):
    """
    Flow360 mesh parameters
    """

    boundaries: MeshBoundary


class Boundary(BaseModel):
    """
    Basic Boundary class
    """

    type: str


class NoSlipWall(Boundary):
    """
    No slip wall boundary
    """

    type = "NoSlipWall"
    velocity: Union[float, str] = Field(alias="Velocity")


class SlipWall(Boundary):
    """
    Slip wall boundary
    """

    type = "SlipWall"


BoundaryType = Union[NoSlipWall, SlipWall]


class ActuatorDisk(BaseModel):
    """
    Actuator disk component
    """

    center: Any
    axis_thrust: Any = Field(alias="axisThrust")
    thickness: Any
    force_per_area: Any = Field(alias="forcePerArea")


class Flow360Params(BaseModel, extra=Extra.allow):
    """
    Flow360 solver parameters
    """

    boundaries: Optional[Dict[str, BoundaryType]]
