"""Edge based meshing parameters for meshing."""

from typing import Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Edge
from flow360.component.simulation.unit_system import AngleType, LengthType


class AngleBasedRefinement(Flow360BaseModel):
    """Surface edge refinement by specifying curvature resolution in degrees"""

    type: Literal["angle"] = pd.Field("angle", frozen=True)
    value: AngleType = pd.Field()


class HeightBasedRefinement(Flow360BaseModel):
    """Surface edge refinement by specifying first layer height of the anisotropic layers"""

    type: Literal["height"] = pd.Field("height", frozen=True)
    # pylint: disable=no-member
    value: LengthType.Positive = pd.Field()


class AspectRatioBasedRefinement(Flow360BaseModel):
    """Surface edge refinement by specifying maximum aspect ratio of the anisotropic cells"""

    type: Literal["aspectRatio"] = pd.Field("aspectRatio", frozen=True)
    value: pd.PositiveFloat = pd.Field()


class ProjectAnisoSpacing(Flow360BaseModel):
    """Project the anisotropic spacing from neighboring faces to the edge"""

    type: Literal["projectAnisoSpacing"] = pd.Field("projectAnisoSpacing", frozen=True)


class _BaseEdgeRefinement(Flow360BaseModel):
    entities: EntityList[Edge] = pd.Field(alias="edges")
    growth_rate: Optional[float] = pd.Field(
        None, description="Growth rate for volume prism layers.", ge=1
    )  # Note:  Per edge specification is actually not supported. This is a global setting in mesher.


class SurfaceEdgeRefinement(_BaseEdgeRefinement):
    """
    Grow anisotropic layers orthogonal to the edge.

    If `method` is None then it projects the anisotropic spacing from neighboring faces to the edge
    (equivalent to `ProjectAniso` in old params).
    """

    refinement_type: Literal["SurfaceEdgeRefinement"] = pd.Field(
        "SurfaceEdgeRefinement", frozen=True
    )
    method: Optional[
        Union[
            AngleBasedRefinement,
            HeightBasedRefinement,
            AspectRatioBasedRefinement,
            ProjectAnisoSpacing,
        ]
    ] = pd.Field(None, discriminator="type")
