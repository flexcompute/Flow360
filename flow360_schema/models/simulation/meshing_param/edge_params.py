"""Edge based meshing parameters for meshing."""

from typing import Literal

import pydantic as pd

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_list import EntityList
from flow360_schema.framework.physical_dimensions import Angle, Length
from flow360_schema.models.entities.geometry_entities import Edge
from flow360_schema.models.simulation.validation.validation_context import (
    ParamsValidationInfo,
    contextual_model_validator,
)


class AngleBasedRefinement(Flow360BaseModel):
    """
    Surface edge refinement by specifying curvature resolution angle.

    Example
    -------

      >>> fl.AngleBasedRefinement(value=8*fl.u.deg)

    ====
    """

    type: Literal["angle"] = pd.Field("angle", frozen=True)
    value: Angle.Float64 = pd.Field()


class HeightBasedRefinement(Flow360BaseModel):
    """
    Surface edge refinement by specifying first layer height of the anisotropic layers.

    Example
    -------

      >>> fl.HeightBasedRefinement(value=1e-4*fl.u.m)

    ====
    """

    type: Literal["height"] = pd.Field("height", frozen=True)
    value: Length.PositiveFloat64 = pd.Field()


class AspectRatioBasedRefinement(Flow360BaseModel):
    """
    Surface edge refinement by specifying maximum aspect ratio of the anisotropic cells.

    Example
    -------

      >>> fl.AspectRatioBasedRefinement(value=10)

    ====
    """

    type: Literal["aspectRatio"] = pd.Field("aspectRatio", frozen=True)
    value: pd.PositiveFloat = pd.Field()


class ProjectAnisoSpacing(Flow360BaseModel):
    """
    Project the anisotropic spacing from neighboring faces to the edge.

    Example
    -------

      >>> fl.ProjectAnisoSpacing()

    ====
    """

    type: Literal["projectAnisoSpacing"] = pd.Field("projectAnisoSpacing", frozen=True)


class SurfaceEdgeRefinement(Flow360BaseModel):
    """
    Setting for growing anisotropic layers orthogonal to the specified `Edge` (s).

    .. note::
        Surface edge refinement is only available with the Legacy mesher. It is
        not supported with the Beta mesher or the Geometry AI mesher.

    Example
    -------

      >>> fl.SurfaceEdgeRefinement(
      ...     edges=[geometry["edge1"], geometry["edge2"]],
      ...     method=fl.HeightBasedRefinement(value=1e-4)
      ... )

    ====
    """

    name: str | None = pd.Field("Surface edge refinement")
    refinement_type: Literal["SurfaceEdgeRefinement"] = pd.Field("SurfaceEdgeRefinement", frozen=True)
    entities: EntityList[Edge] = pd.Field(alias="edges")
    method: AngleBasedRefinement | HeightBasedRefinement | AspectRatioBasedRefinement | ProjectAnisoSpacing = pd.Field(
        discriminator="type",
        description="Method for determining the spacing. See :class:`AngleBasedRefinement`,"
        " :class:`HeightBasedRefinement`, :class:`AspectRatioBasedRefinement`, :class:`ProjectAnisoSpacing`",
    )

    @contextual_model_validator(mode="after")
    def ensure_legacy_mesher(self, param_info: ParamsValidationInfo):
        """Ensure that neither geometry AI nor the beta mesher is enabled when using this feature."""
        # Geometry AI always runs on the beta mesher, so check it first for the more specific message.
        if param_info.use_geometry_AI:
            raise ValueError("SurfaceEdgeRefinement is not currently supported with geometry AI.")
        if param_info.is_beta_mesher:
            raise ValueError("SurfaceEdgeRefinement is not currently supported with the beta mesher.")
        return self
