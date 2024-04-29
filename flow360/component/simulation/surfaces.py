"""
Contains basically only boundary conditons for now. In future we can add new models like 2D equations.
"""

from typing import List, Literal, Optional, Tuple, Union

from flow360.component.simulation.base_model import Flow360BaseModel


class Boundary:
    pass


class BoundaryWithTurbulenceQuantities:
    pass


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
