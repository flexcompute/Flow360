"""Face based meshing parameters for meshing."""

from typing import Literal, Optional

import pydantic as pd

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.models.surface_models import EntityListAllowingGhost
from flow360.component.simulation.primitives import (
    GhostCircularPlane,
    GhostSurface,
    Surface,
)
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.validation.validation_context import (
    get_validation_info,
)
from flow360.component.simulation.validation.validation_utils import (
    check_deleted_surface_in_entity_list,
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
    entities: EntityListAllowingGhost[Surface, GhostSurface, GhostCircularPlane] = pd.Field(
        alias="faces"
    )
    # pylint: disable=no-member
    max_edge_length: LengthType.Positive = pd.Field(
        description="Maximum edge length of surface cells."
    )

    @pd.field_validator("entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value):
        """Ensure all boundaries will be present after mesher"""
        check_ghost_surface_usage_policy_for_face_refinements(
            value.stored_entities, feature_name="SurfaceRefinement"
        )
        return check_deleted_surface_in_entity_list(value)


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
    entities: EntityList[Surface] = pd.Field(alias="faces")
    # pylint: disable=no-member

    geometry_accuracy: Optional[LengthType.Positive] = pd.Field(
        None,
        description="The smallest length scale that will be resolved accurately by the surface meshing process. ",
    )

    preserve_thin_geometry: Optional[bool] = pd.Field(
        False,
        description="Flag to specify whether thin geometry features with thickness roughly equal "
        + "to geometry_accuracy should be resolved accurately during the surface meshing process.",
    )

    sealing_size: LengthType.NonNegative = pd.Field(
        0.0 * u.m,
        description="Threshold size below which all geometry gaps are automatically closed.",
    )

    # Note: No checking on deleted surfaces since geometry accuracy on deleted surface does impact the volume mesh.

    @pd.model_validator(mode="after")
    def ensure_geometry_ai(self):
        """Ensure feature is only activated with geometry AI enabled."""
        validation_info = get_validation_info()
        if validation_info is None:
            return self
        if not validation_info.use_geometry_AI:
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
    entities: EntityListAllowingGhost[Surface, GhostSurface, GhostCircularPlane] = pd.Field(
        alias="faces"
    )

    @pd.field_validator("entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value):
        """Ensure all boundaries will be present after mesher"""
        check_ghost_surface_usage_policy_for_face_refinements(
            value.stored_entities, feature_name="PassiveSpacing"
        )
        return check_deleted_surface_in_entity_list(value)


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
    entities: EntityList[Surface] = pd.Field(alias="faces")
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

    @pd.field_validator("entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value):
        """Ensure all boundaries will be present after mesher"""
        return check_deleted_surface_in_entity_list(value)

    @pd.field_validator("growth_rate", mode="after")
    @classmethod
    def invalid_growth_rate(cls, value):
        """Ensure growth rate per face is not specified"""
        validation_info = get_validation_info()

        if validation_info is None:
            return value

        if value is not None and not validation_info.is_beta_mesher:
            raise ValueError("Growth rate per face is only supported by the beta mesher.")
        return value

    @pd.field_validator("first_layer_thickness", mode="after")
    @classmethod
    def require_first_layer_thickness(cls, value):
        """Verify first layer thickness is specified"""
        validation_info = get_validation_info()

        if validation_info is None:
            return value

        if value is None and not validation_info.is_beta_mesher:
            raise ValueError("First layer thickness is required.")
        return value
