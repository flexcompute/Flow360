from typing import List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Box, Cylinder, Face
from flow360.component.simulation.unit_system import LengthType

"""
Meshing settings that applies to volumes.
"""


class UniformRefinement(Flow360BaseModel):
    spacing: LengthType.Positive = pd.Field()
    entities: EntityList[Box, Cylinder] = pd.Field()


class AxisymmetricRefinement(Flow360BaseModel):
    """
    Note:
    - This basically creates the "rotorDisks" type of volume refinement that we used to have. If enclosed_objects is provided then it is implied that this is a sliding interface.

    - In the future to support arbitrary-axisymmetric shaped sliding interface, we will define classes parallel to AxisymmetricRefinement.

    - `enclosed_objects` is actually just a way of specifying the enclosing patches of a volume zone. Therefore in the future when supporting arbitrary-axisymmetric shaped sliding interface, we may not need this attribute at all. For example if the new class already has an entry to list all the enclosing patches.

    - We may provide a helper function to automatically determine what is inside the encloeud_objects list based on the mesh data. But this currently is out of scope due to the estimated efforts.
    """

    entities: EntityList[Cylinder] = pd.Field()
    spacing_axial: LengthType.PositiveFloat = pd.Field()
    spacing_radial: LengthType.PositiveFloat = pd.Field()
    spacing_circumferential: LengthType.PositiveFloat = pd.Field()
    enclosed_objects: Optional[EntityList[Box, Cylinder, Face]] = pd.Field(
        None,
        description="Entities enclosed by this sliding interface. Can be faces, boxes and/or other cylinders etc. This helps determining the volume zone boundary.",
    )


ZoneRefinementTypes = Union[UniformRefinement, AxisymmetricRefinement]
