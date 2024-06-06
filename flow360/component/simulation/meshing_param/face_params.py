from typing import Literal, Optional

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Face
from flow360.component.simulation.unit_system import LengthType

"""
Meshing settings that applies to faces.
"""


class FaceRefinement(Flow360BaseModel):
    """
    These affects surface meshing.

    Note:
    - `None` entities will be expanded (or just ignored and convert to global default, depending on implementation)
    before submission. This is supposed to be applied to all the matching entities. We allow this so that we do not
    need to have dedicated field for global settings. This is also consistent with the `FluidDynamics` class' design.

    - For `FaceRefinement` we may need validation to detect if default has been set or not. This is because we need
    these defaults so that the when face name is not present, what config we ues. Depending on how we go down the road.
    """

    entities: Optional[EntityList[Face]] = pd.Field(None, alias="faces")
    max_edge_length: LengthType.Positive = pd.Field(
        description="Local maximum edge length for surface cells."
    )
    curvature_resolution_angle: pd.PositiveFloat = pd.Field(
        description="""
        Global maximum angular deviation in degrees. This value will restrict:
        (1) The angle between a cell’s normal and its underlying surface normal
        (2) The angle between a line segment’s normal and its underlying curve normal
        """
    )


class BoundaryLayerRefinement(Flow360BaseModel):
    """
    These affects volume meshing.
    Note:
    - We do not support per volume specification of these settings so the entities will be **obsolete** for now.
    Should we have it at all in the release?

    - `None` entities will be expanded (or just ignored and convert to global default, depending on implementation) before
    submission. This is supposed to be applied to all the matching entities. We allow this so that we do not need to
    have dedicated field for global settings. This is also consistent with the `FluidDynamics` class' design.
    """

    type: Literal["aniso", "projectAnisoSpacing", "none"] = pd.Field()
    entities: Optional[EntityList[Face]] = pd.Field(None, alias="faces")
    first_layer_thickness: LengthType.Positive = pd.Field(
        description="First layer thickness for volumetric anisotropic layers."
    )
    growth_rate: pd.PositiveFloat = pd.Field(
        description="Growth rate for volume prism layers.", ge=1
    )  # Note:  Per face specification is actually not supported. This is a global setting in mesher.
