from typing import List, Literal, Optional, Tuple, Union

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel
from flow360.component.simulation.entities_base import EntitiesBase


class Farfield(Flow360BaseModel):
    """
    Farfield type for meshing
    """

    type: Literal["auto", "quasi-3d", "user-defined"] = pd.Field()


class Transformation(Flow360BaseModel):
    axis_of_rotation: Optional[Tuple[float, float, float]] = pd.Field()
    angle_of_rotation: Optional[float] = pd.Field()


class BoxRefinement(EntitiesBase):
    type = pd.Field("box", frozen=True)
    spacing: pd.PositiveFloat = pd.Field()
    transformation: Optional[Transformation] = pd.Field()


class CylinderRefinement(EntitiesBase):
    """Note: `type` can actually be infered from the entity passed in (BoxZone or CylindricalZone). So there is no point differentiating `CylinderRefinement` from `BoxRefinement`"""

    type = pd.Field("cylinder", frozen=True)
    spacing: pd.PositiveFloat = pd.Field()


class RotorDisk(EntitiesBase):
    """:class: RotorDisk"""

    spacing_axial: pd.PositiveFloat = pd.Field()
    spacing_radial: pd.PositiveFloat = pd.Field()
    spacing_circumferential: pd.PositiveFloat = pd.Field()


class SlidingInterface(EntitiesBase):
    """:class: SlidingInterface for meshing"""

    spacing_axial: pd.PositiveFloat = pd.Field()
    spacing_radial: pd.PositiveFloat = pd.Field()
    spacing_circumferential: pd.PositiveFloat = pd.Field()
    enclosed_objects: Optional[List[EntitiesBase]] = pd.Field()


ZoneRefinementTypes = Union[BoxRefinement, CylinderRefinement, RotorDisk, SlidingInterface]
