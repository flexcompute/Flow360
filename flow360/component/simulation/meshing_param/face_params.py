"""Face based meshing parameters for meshing."""

from typing import Literal, Optional

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.models.surface_models import EntityListAllowingGhost
from flow360.component.simulation.primitives import (
    GhostCircularPlane,
    GhostSurface,
    Surface,
    WindTunnelGhostSurface,
)
from flow360.component.simulation.unit_system import AngleType, LengthType
from flow360.component.simulation.validation.validation_context import (
    ParamsValidationInfo,
    contextual_field_validator,
    contextual_model_validator,
)
from flow360.component.simulation.validation.validation_utils import (
    check_deleted_surface_in_entity_list,
    check_geometry_ai_features,
    check_ghost_surface_usage_policy_for_face_refinements,
)


class SurfaceRefinement(Flow360BaseModel):
    """
    Setting for refining surface elements for given `Surface`.

    Example
    -------

      >>> fl.SurfaceRefinement(
      ...     faces=[geometry["face1"], geometry["face2"]],
      ...     max_edge_length=0.001*fl.u.m
      ... )

    ====
    """

    name: Optional[str] = pd.Field("Surface refinement")
    refinement_type: Literal["SurfaceRefinement"] = pd.Field("SurfaceRefinement", frozen=True)
    entities: EntityListAllowingGhost[
        Surface, WindTunnelGhostSurface, GhostSurface, GhostCircularPlane
    ] = pd.Field(alias="faces")
    # pylint: disable=no-member
    max_edge_length: Optional[LengthType.Positive] = pd.Field(
        None, description="Maximum edge length of surface cells."
    )

    curvature_resolution_angle: Optional[AngleType.Positive] = pd.Field(
        None,
        description=(
            "Default maximum angular deviation in degrees. "
            "This value will restrict the angle between a cell’s normal and its underlying surface normal."
        ),
    )

    resolve_face_boundaries: Optional[bool] = pd.Field(
        None,
        description="Flag to specify whether boundaries between adjacent faces should be resolved "
        + "accurately during the surface meshing process using anisotropic mesh refinement.",
    )

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        check_ghost_surface_usage_policy_for_face_refinements(
            value.stored_entities, feature_name="SurfaceRefinement", param_info=param_info
        )
        return check_deleted_surface_in_entity_list(value, param_info)

    @contextual_field_validator(
        "curvature_resolution_angle", "resolve_face_boundaries", mode="after"
    )
    @classmethod
    def ensure_geometry_ai_features(cls, value, info, param_info: ParamsValidationInfo):
        """Validate that the feature is only used when Geometry AI is enabled."""
        return check_geometry_ai_features(cls, value, info, param_info)

    @pd.model_validator(mode="after")
    def require_at_least_one_setting(self):
        """Ensure that at least one of max_edge_length, curvature_resolution_angle,
        or resolve_face_boundaries is specified for SurfaceRefinement.
        """
        if (
            self.max_edge_length is None
            and self.curvature_resolution_angle is None
            and self.resolve_face_boundaries is None
        ):
            raise ValueError(
                "SurfaceRefinement requires at least one of 'max_edge_length', "
                "'curvature_resolution_angle', or 'resolve_face_boundaries' to be specified."
            )
        return self


class GeometryRefinement(Flow360BaseModel):
    """
    Setting for refining surface elements for given `Surface`.

    Example
    -------

      >>> fl.GeometryRefinement(
      ...     faces=[geometry["face1"], geometry["face2"]],
      ...     geometry_accuracy=0.001*fl.u.m
      ... )

    ====
    """

    name: Optional[str] = pd.Field("Geometry refinement")
    refinement_type: Literal["GeometryRefinement"] = pd.Field("GeometryRefinement", frozen=True)
    entities: EntityList[Surface, WindTunnelGhostSurface] = pd.Field(alias="faces")
    # pylint: disable=no-member

    geometry_accuracy: Optional[LengthType.Positive] = pd.Field(
        None,
        description="The smallest length scale that will be resolved accurately by the surface meshing process. ",
    )

    preserve_thin_geometry: Optional[bool] = pd.Field(
        None,
        description="Flag to specify whether thin geometry features with thickness roughly equal "
        + "to geometry_accuracy should be resolved accurately during the surface meshing process.",
    )

    sealing_size: Optional[LengthType.NonNegative] = pd.Field(
        None,
        description="Threshold size below which all geometry gaps are automatically closed.",
    )

    # Note: No checking on deleted surfaces since geometry accuracy on deleted surface does impact the volume mesh.

    @contextual_model_validator(mode="after")
    def ensure_geometry_ai(self, param_info: ParamsValidationInfo):
        """Ensure feature is only activated with geometry AI enabled."""
        if not param_info.use_geometry_AI:
            raise ValueError("GeometryRefinement is only supported by geometry AI.")
        return self


class PassiveSpacing(Flow360BaseModel):
    """
    Passively control the mesh spacing either through adjacent `Surface`'s meshing
    setting or doing nothing to change existing surface mesh at all.

    Example
    -------

      >>> fl.PassiveSpacing(
      ...     faces=[geometry["face1"], geometry["face2"]],
      ...     type="projected"
      ... )

    ====
    """

    name: Optional[str] = pd.Field("Passive spacing")
    type: Literal["projected", "unchanged"] = pd.Field(
        description="""
        1. When set to *projected*, turn off anisotropic layers growing for this `Surface`. 
        Project the anisotropic spacing from the neighboring volumes to this face.

        2. When set to *unchanged*, turn off anisotropic layers growing for this `Surface`. 
        The surface mesh will remain unaltered when populating the volume mesh.
        """
    )
    refinement_type: Literal["PassiveSpacing"] = pd.Field("PassiveSpacing", frozen=True)
    entities: EntityListAllowingGhost[
        Surface, WindTunnelGhostSurface, GhostSurface, GhostCircularPlane
    ] = pd.Field(alias="faces")

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        check_ghost_surface_usage_policy_for_face_refinements(
            value.stored_entities, feature_name="PassiveSpacing", param_info=param_info
        )
        return check_deleted_surface_in_entity_list(value, param_info)


class BoundaryLayer(Flow360BaseModel):
    """
    Setting for growing anisotropic layers orthogonal to the specified `Surface` (s).

    Example
    -------

      >>> fl.BoundaryLayer(
      ...     faces=[geometry["face1"], geometry["face2"]],
      ...     first_layer_thickness=1e-5,
      ...     growth_rate=1.15
      ... )

    ====
    """

    name: Optional[str] = pd.Field("Boundary layer refinement")
    refinement_type: Literal["BoundaryLayer"] = pd.Field("BoundaryLayer", frozen=True)
    entities: EntityList[Surface, WindTunnelGhostSurface] = pd.Field(alias="faces")
    # pylint: disable=no-member
    first_layer_thickness: Optional[LengthType.Positive] = pd.Field(
        None,
        description="First layer thickness for volumetric anisotropic layers grown from given `Surface` (s).",
    )

    growth_rate: Optional[float] = pd.Field(
        None,
        ge=1,
        description="Growth rate for volume prism layers for given `Surface` (s)."
        " Supported only by the beta mesher.",
    )

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        return check_deleted_surface_in_entity_list(value, param_info)

    @contextual_field_validator("growth_rate", mode="after")
    @classmethod
    def invalid_growth_rate(cls, value, param_info: ParamsValidationInfo):
        """Ensure growth rate per face is not specified"""

        if value is not None and not param_info.is_beta_mesher:
            raise ValueError("Growth rate per face is only supported by the beta mesher.")
        return value

    @contextual_field_validator("first_layer_thickness", mode="after")
    @classmethod
    def require_first_layer_thickness(cls, value, param_info: ParamsValidationInfo):
        """Verify first layer thickness is specified"""
        if value is None and not param_info.is_beta_mesher:
            raise ValueError("First layer thickness is required.")
        return value
