"""
Contains basically only boundary conditons for now. In future we can add new models like 2D equations.
"""

from typing import List, Literal, Optional, Tuple, Union

from flow360.component.flow360_params.params_base import Flow360BaseModel
from flow360.component.flow360_params.boundaries import Boundary, BoundaryWithTurbulenceQuantities


class NoSlipWall(Boundary):
    pass


class SlipWall(Boundary):
    pass


class RiemannInvariant(Boundary):
    pass


class FreestreamBoundary(BoundaryWithTurbulenceQuantities):
    pass


class WallFunction(Boundary):
    pass


...

SurfaceTypes = Union[NoSlipWall, WallFunction]
