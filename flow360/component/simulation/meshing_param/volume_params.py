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
    entities: EntityList[Box, Cylinder] = pd.Field()
    spacing: LengthType.Positive = pd.Field()


class AxisymmetricRefinement(Flow360BaseModel):
    """
    Note:
    - This basically creates the "rotorDisks" type of volume refinement that we used to have.

    - `enclosed_objects` is actually just a way of specifying the enclosing patches of a volume zone. Therefore in the future when supporting arbitrary-axisymmetric shaped sliding interface, we may not need this attribute at all. For example if the new class already has an entry to list all the enclosing patches.

    - We may provide a helper function to automatically determine what is inside the encloeud_objects list based on the mesh data. But this currently is out of scope due to the estimated efforts.
    """

    entities: EntityList[Cylinder] = pd.Field()
    spacing_axial: LengthType.Positive = pd.Field()
    spacing_radial: LengthType.Positive = pd.Field()
    spacing_circumferential: LengthType.Positive = pd.Field()


class RotationCylinder(AxisymmetricRefinement):
    """This is the original SlidingInterface. This will create new volume zones
    Will add RotationSphere class in the future.
    Please refer to
    https://www.notion.so/flexcompute/Python-model-design-document-78d442233fa944e6af8eed4de9541bb1?pvs=4#c2de0b822b844a12aa2c00349d1f68a3
    """

    enclosed_objects: Optional[EntityList[Cylinder, Face]] = pd.Field(
        None,
        description="Entities enclosed by this sliding interface. Can be faces, boxes and/or other cylinders etc. This helps determining the volume zone boundary.",
    )


VolumeRefinementTypes = Union[UniformRefinement, AxisymmetricRefinement]
