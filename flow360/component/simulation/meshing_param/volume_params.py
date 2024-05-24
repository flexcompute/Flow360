from typing import List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Box, Cylinder


class Farfield(Flow360BaseModel):
    """
    Farfield type for meshing
    """

    type: Literal["auto", "quasi-3d", "user-defined"] = pd.Field()


class UniformRefinement(Flow360BaseModel):
    """TODO: `type` can actually be infered from the type of entity passed in (Box or Cylinder)."""

    spacing: pd.PositiveFloat = pd.Field()
    entities: EntityList[Box, Cylinder] = pd.Field()


class RotorDisks(Flow360BaseModel):
    entities: EntityList[Cylinder] = pd.Field()
    spacing_axial: pd.PositiveFloat = pd.Field()
    spacing_radial: pd.PositiveFloat = pd.Field()
    spacing_circumferential: pd.PositiveFloat = pd.Field()


class SlidingInterface(RotorDisks):
    """:class:  SlidingInterface
    TODO: For SlidingInterface, enclosed_objects: Optional[List[EntitiesBase]] = pd.Field() will be infered from mesh data.
    """

    enclosed_objects: EntityList[Box, Cylinder] = pd.Field(None)


ZoneRefinementTypes = Union[UniformRefinement, RotorDisks, SlidingInterface]
