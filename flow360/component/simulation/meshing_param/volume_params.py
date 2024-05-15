from typing import List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel
from flow360.component.simulation.entities_base import EntitiesBase


class Farfield(Flow360BaseModel):
    """
    Farfield type for meshing
    """

    type: Literal["auto", "quasi-3d", "user-defined"] = pd.Field()


class UniformRefinement(EntitiesBase):
    """TODO: `type` can actually be infered from the type of entity passed in (Box or Cylinder)."""

    type: str = pd.Field("NotSet")  # Should be "box" or "cylinder"
    spacing: pd.PositiveFloat = pd.Field()


class CylindricalRefinement(EntitiesBase):
    """:class: CylindricalRefinement
    Note: This uniffies RotorDisk and SlidingInterface.
    For SlidingInterface, enclosed_objects: Optional[List[EntitiesBase]] = pd.Field() will be infered from mesh data.
    """

    spacing_axial: pd.PositiveFloat = pd.Field()
    spacing_radial: pd.PositiveFloat = pd.Field()
    spacing_circumferential: pd.PositiveFloat = pd.Field()
    enclosed_objects: Optional[List[EntitiesBase]] = pd.Field(None)


ZoneRefinementTypes = Union[UniformRefinement, CylindricalRefinement]
