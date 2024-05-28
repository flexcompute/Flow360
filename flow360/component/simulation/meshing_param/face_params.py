from typing import Literal

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Face

"""
Meshing settings that applies to faces.
"""


class FaceRefinement(Flow360BaseModel):
    """
    These affects surface meshing.
    """

    entities: EntityList[Face] = pd.Field(alias="faces")
    max_edge_length: LengthType.PositiveFloat = pd.Field(
        description="Local maximum edge length for surface cells."
    )


class BoundaryLayerRefinement(Flow360BaseModel):
    """
    These affects volume meshing.
    Note:
    We do not support per volume specification of these settings so the entities will be **obsolete** for now.
    Should we have it at all in the release?
    """

    type: Literal["aniso", "projectAnisoSpacing", "none"] = pd.Field()
    entities: EntityList[Face] = pd.Field()
    first_layer_thickness: LengthType.PositiveFloat = pd.Field(
        description="First layer thickness for volumetric anisotropic layers."
    )
    growth_rate: pd.PositiveFloat = pd.Field(
        description="Growth rate for volume prism layers."
    )  # Note:  Per face specification is actually not supported. This is a global setting in mesher.
