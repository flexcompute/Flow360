"""
Contains basically only boundary conditons for now. In future we can add new models like 2D equations.
"""

from abc import ABCMeta
from typing import List, Literal, Optional, Tuple, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel

BoundaryVelocityType = Tuple[pd.StrictStr, pd.StrictStr, pd.StrictStr]


class Surface(Flow360BaseModel):
    mesh_patch_name: str = pd.Field()


class BoundaryBase(Flow360BaseModel, metaclass=ABCMeta):
    """`name` attribute is contained within the entity"""

    type: str = pd.Field()


class BoundaryBaseWithTurbulenceQuantities(BoundaryBase, metaclass=ABCMeta):
    pass


class Wall(BoundaryBase):
    """Replace Flow360Param:
    - NoSlipWall
    - IsothermalWall
    - HeatFluxWall
    - WallFunction
    - SolidIsothermalWall
    - SolidAdiabaticWall
    """

    type: Literal["Wall"] = pd.Field("Wall", frozen=True)
    use_wall_function: bool = pd.Field()
    velocity: Optional[BoundaryVelocityType] = pd.Field()
    velocity_type: Optional[Literal["absolute", "relative"]] = pd.Field(default="relative")
    temperature: Union[pd.PositiveFloat, pd.StrictStr] = pd.Field()
    heat_flux: Union[float, pd.StrictStr] = pd.Field(
        alias="heatFlux", options=["Value", "Expression"]
    )


class SlipWall(BoundaryBase):
    type: Literal["SlipWall"] = pd.Field("SlipWall", frozen=True)


class RiemannInvariant(BoundaryBase):
    type: Literal["RiemannInvariant"] = pd.Field("RiemannInvariant", frozen=True)


class FreestreamBoundary(BoundaryBaseWithTurbulenceQuantities):
    type: Literal["Freestream"] = pd.Field("Freestream", frozen=True)
    velocity: Optional[BoundaryVelocityType] = pd.Field(alias="Velocity")
    velocity_type: Optional[Literal["absolute", "relative"]] = pd.Field(
        default="relative", alias="velocityType"
    )


class SubsonicOutflowPressure(BoundaryBaseWithTurbulenceQuantities):
    """Same as Flow360Param"""


class SubsonicOutflowMach(BoundaryBaseWithTurbulenceQuantities):
    """Same as Flow360Param"""


class SubsonicInflow(BoundaryBaseWithTurbulenceQuantities):
    """Same as Flow360Param"""


class SupersonicInflow(BoundaryBaseWithTurbulenceQuantities):
    """Same as Flow360Param"""


class MassInflow(BoundaryBaseWithTurbulenceQuantities):
    """Same as Flow360Param"""


class MassOutflow(BoundaryBaseWithTurbulenceQuantities):
    """Same as Flow360Param"""


class TranslationallyPeriodic(BoundaryBase):
    """Same as Flow360Param"""


class RotationallyPeriodic(BoundaryBase):
    """Same as Flow360Param"""


class SymmetryPlane(BoundaryBase):
    """Same as Flow360Param"""


class VelocityInflow(BoundaryBaseWithTurbulenceQuantities):
    """Same as Flow360Param"""


class PressureOutflow(BoundaryBaseWithTurbulenceQuantities):
    """Same as Flow360Param"""


...

SurfaceTypes = Union[
    Wall,
    SlipWall,
    FreestreamBoundary,
    SubsonicOutflowPressure,
    SubsonicOutflowMach,
    SubsonicInflow,
    SupersonicInflow,
    MassInflow,
    MassOutflow,
    TranslationallyPeriodic,
    RotationallyPeriodic,
    SymmetryPlane,
    RiemannInvariant,
    VelocityInflow,
    PressureOutflow,
]
