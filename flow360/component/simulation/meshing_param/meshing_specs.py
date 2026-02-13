"""Default settings for meshing using different meshing algorithms"""

from math import log2
from typing import Optional

import numpy as np
import pydantic as pd

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.updater import (
    DEFAULT_PLANAR_FACE_TOLERANCE,
    DEFAULT_SLIDING_INTERFACE_TOLERANCE,
)
from flow360.component.simulation.unit_system import AngleType, LengthType
from flow360.component.simulation.validation.validation_context import (
    SURFACE_MESH,
    VOLUME_MESH,
    ConditionalField,
    ContextField,
    ParamsValidationInfo,
    add_validation_warning,
    contextual_field_validator,
)
from flow360.component.simulation.validation.validation_utils import (
    check_geometry_ai_features,
)


class OctreeSpacing(Flow360BaseModel):
    """
    Helper class for octree-based meshers. Holds the base for the octree spacing and lows calculation of levels.
    """

    # pylint: disable=no-member
    base_spacing: LengthType.Positive

    @pd.model_validator(mode="before")
    @classmethod
    def _project_spacing_to_object(cls, input_data):
        if isinstance(input_data, u.unyt.unyt_quantity):
            return {"base_spacing": input_data}
        return input_data

    @pd.validate_call
    def __getitem__(self, idx: int):
        return self.base_spacing * (2 ** (-idx))

    # pylint: disable=no-member
    @pd.validate_call
    def to_level(self, spacing: LengthType.Positive):
        """
        Can be used to check in what refinement level would the given spacing result
        and if it is a direct match in the spacing series.
        """
        level = -log2(spacing / self.base_spacing)

        direct_spacing = np.isclose(level, np.round(level), atol=1e-8)
        returned_level = np.round(level) if direct_spacing else np.ceil(level)
        return returned_level, direct_spacing


class MeshingDefaults(Flow360BaseModel):
    """
    Default/global settings for meshing parameters.

    Example
    -------

      >>> fl.MeshingDefaults(
      ...     surface_max_edge_length=1*fl.u.m,
      ...     surface_edge_growth_rate=1.2,
      ...     curvature_resolution_angle=12*fl.u.deg,
      ...     boundary_layer_growth_rate=1.1,
      ...     boundary_layer_first_layer_thickness=1e-5*fl.u.m
      ... )

    ====
    """

    # pylint: disable=no-member
    geometry_accuracy: Optional[LengthType.Positive] = pd.Field(
        None,
        description="The smallest length scale that will be resolved accurately by the surface meshing process. "
        "This parameter is only valid when using geometry AI."
        "It can be overridden with class: ~flow360.GeometryRefinement.",
    )

    ##::   Default surface edge settings
    surface_edge_growth_rate: float = ContextField(
        1.2,
        ge=1,
        description="Growth rate of the anisotropic layers grown from the edges."
        "This can not be overridden per edge.",
        context=SURFACE_MESH,
    )

    ##::    Default boundary layer settings
    boundary_layer_growth_rate: float = ContextField(
        1.2,
        description="Default growth rate for volume prism layers.",
        ge=1,
        context=VOLUME_MESH,
    )
    # pylint: disable=no-member
    boundary_layer_first_layer_thickness: Optional[LengthType.Positive] = ConditionalField(
        None,
        description="Default first layer thickness for volumetric anisotropic layers."
        " This can be overridden with :class:`~flow360.BoundaryLayer`.",
        context=VOLUME_MESH,
    )  # Truly optional if all BL faces already have first_layer_thickness

    number_of_boundary_layers: Optional[pd.NonNegativeInt] = pd.Field(
        None,
        description="Default number of volumetric anisotropic layers."
        " The volume mesher will automatically calculate the required"
        " no. of layers to grow the boundary layer elements to isotropic size if not specified."
        " This is only supported by the beta mesher and can not be overridden per face.",
    )

    planar_face_tolerance: pd.NonNegativeFloat = pd.Field(
        DEFAULT_PLANAR_FACE_TOLERANCE,
        strict=True,
        description="Tolerance used for detecting planar faces in the input surface mesh / geometry"
        " that need to be remeshed, such as symmetry planes."
        " This tolerance is non-dimensional, and represents a distance"
        " relative to the largest dimension of the bounding box of the input surface mesh / geometry."
        " This can not be overridden per face.",
    )
    # pylint: disable=duplicate-code
    sliding_interface_tolerance: pd.NonNegativeFloat = ConditionalField(
        DEFAULT_SLIDING_INTERFACE_TOLERANCE,
        strict=True,
        description="Tolerance used for detecting / creating curves in the input surface mesh / geometry lying on"
        " sliding interfaces. This tolerance is non-dimensional, and represents a distance"
        " relative to the smallest radius of all sliding interfaces specified in meshing parameters."
        " This cannot be overridden per sliding interface.",
        context=VOLUME_MESH,
    )

    ##::    Default surface layer settings
    surface_max_edge_length: Optional[LengthType.Positive] = ConditionalField(
        None,
        description="Default maximum edge length for surface cells."
        " This can be overridden with :class:`~flow360.SurfaceRefinement`.",
        context=SURFACE_MESH,
    )

    surface_max_aspect_ratio: pd.PositiveFloat = ConditionalField(
        10.0,
        description="Maximum aspect ratio for surface cells for the GAI surface mesher."
        " This cannot be overridden per face",
        context=SURFACE_MESH,
    )

    surface_max_adaptation_iterations: pd.NonNegativeInt = ConditionalField(
        50,
        description="Maximum adaptation iterations for the GAI surface mesher.",
        context=SURFACE_MESH,
    )

    curvature_resolution_angle: AngleType.Positive = ContextField(
        12 * u.deg,
        description=(
            "Default maximum angular deviation in degrees. This value will restrict:"
            " 1. The angle between a cell’s normal and its underlying surface normal."
            " 2. The angle between a line segment’s normal and its underlying curve normal."
            " This can be overridden per face only when using geometry AI."
        ),
        context=SURFACE_MESH,
    )

    resolve_face_boundaries: bool = pd.Field(
        False,
        description="Flag to specify whether boundaries between adjacent faces should be resolved "
        + "accurately during the surface meshing process using anisotropic mesh refinement. "
        + "This option is only supported when using geometry AI, and can be overridden "
        + "per face with :class:`~flow360.SurfaceRefinement`.",
    )

    preserve_thin_geometry: bool = pd.Field(
        False,
        description="Flag to specify whether thin geometry features with thickness roughly equal "
        + "to geometry_accuracy should be resolved accurately during the surface meshing process. "
        + "This option is only supported when using geometry AI, and can be overridden "
        + "per face with :class:`~flow360.GeometryRefinement`.",
    )

    sealing_size: LengthType.NonNegative = pd.Field(
        0.0 * u.m,
        description="Threshold size below which all geometry gaps are automatically closed. "
        + "This option is only supported when using geometry AI, and can be overridden "
        + "per face with :class:`~flow360.GeometryRefinement`.",
    )

    remove_non_manifold_faces: bool = pd.Field(
        False,
        description="Flag to remove non-manifold and interior faces. "
        + "This option is only supported when using geometry AI.",
    )

    remove_hidden_geometry: bool = pd.Field(
        False,
        description="Flag to remove hidden geometry that is not visible to flow. "
        + "This option is only supported when using geometry AI.",
    )

    flooding_cell_size: Optional[LengthType.Positive] = pd.Field(
        None,
        description="Minimum cell size used for flood-fill exterior classification. "
        + "If not specified, the value is derived from geometry_accuracy and sealing_size. "
        + "This option is only supported when using geometry AI.",
    )

    edge_split_layers: int = pd.Field(
        1,
        ge=0,
        # Skip default-value validation so warnings are emitted only when users explicitly set this field.
        validate_default=False,
        description="The number of layers that are considered for edge splitting in the boundary layer mesh."
        + "This only affects beta mesher.",
    )

    octree_spacing: Optional[OctreeSpacing] = pd.Field(
        None,
        description="Octree spacing configuration for volume meshing. "
        "If specified, this will be used to control the base spacing for octree-based meshers.",
    )

    @contextual_field_validator("number_of_boundary_layers", mode="after")
    @classmethod
    def invalid_number_of_boundary_layers(cls, value, param_info: ParamsValidationInfo):
        """Ensure number of boundary layers is not specified"""
        if value is not None and not param_info.is_beta_mesher:
            raise ValueError("Number of boundary layers is only supported by the beta mesher.")
        return value

    @contextual_field_validator("edge_split_layers", mode="after")
    @classmethod
    def invalid_edge_split_layers(cls, value, param_info: ParamsValidationInfo):
        """Ensure edge split layers is only configured for beta mesher."""
        if value > 0 and not param_info.is_beta_mesher:
            add_validation_warning(
                "`edge_split_layers` is only supported by the beta mesher; "
                "this setting will be ignored."
            )
        return value

    @contextual_field_validator("geometry_accuracy", mode="after")
    @classmethod
    def invalid_geometry_accuracy(cls, value, param_info: ParamsValidationInfo):
        """Ensure geometry accuracy is not specified when GAI is not used"""
        if value is not None and not param_info.use_geometry_AI:
            raise ValueError("Geometry accuracy is only supported when geometry AI is used.")

        if value is None and param_info.use_geometry_AI:
            raise ValueError("Geometry accuracy is required when geometry AI is used.")
        return value

    @contextual_field_validator(
        "surface_max_aspect_ratio",
        "surface_max_adaptation_iterations",
        "resolve_face_boundaries",
        "preserve_thin_geometry",
        "sealing_size",
        "remove_non_manifold_faces",
        "remove_hidden_geometry",
        "flooding_cell_size",
        mode="after",
    )
    @classmethod
    def ensure_geometry_ai_features(cls, value, info, param_info: ParamsValidationInfo):
        """Validate that the feature is only used when Geometry AI is enabled."""
        return check_geometry_ai_features(cls, value, info, param_info)

    @contextual_field_validator("octree_spacing", mode="after")
    @classmethod
    def _set_default_octree_spacing(cls, octree_spacing, param_info: ParamsValidationInfo):
        """Set default octree_spacing to 1 * project_length_unit when not specified."""
        if octree_spacing is not None:
            return octree_spacing
        if param_info.project_length_unit is None:
            add_validation_warning(
                "No project length unit found; `octree_spacing` will not be set automatically. "
                "Octree spacing validation will be skipped."
            )
            return octree_spacing

        # pylint: disable=no-member
        project_length = 1 * LengthType.validate(param_info.project_length_unit)
        return OctreeSpacing(base_spacing=project_length)

    @pd.model_validator(mode="after")
    def validate_mutual_exclusion(self):
        """Ensure remove_non_manifold_faces and remove_hidden_geometry are not both True."""
        if self.remove_non_manifold_faces and self.remove_hidden_geometry:
            raise ValueError(
                "'remove_non_manifold_faces' and 'remove_hidden_geometry' cannot both be True. "
                "Please enable only one of these options."
            )
        return self

    @pd.model_validator(mode="after")
    def validate_flooding_cell_size_requires_remove_hidden_geometry(self):
        """Ensure flooding_cell_size is only specified when remove_hidden_geometry is True."""
        if self.flooding_cell_size is not None and not self.remove_hidden_geometry:
            raise ValueError(
                "'flooding_cell_size' can only be specified when 'remove_hidden_geometry' is True."
            )
        return self


class VolumeMeshingDefaults(Flow360BaseModel):
    """
    Default/global settings for volume meshing parameters. To be used with class:`ModularMeshingWorkflow`.
    """

    ##::    Default boundary layer settings
    boundary_layer_growth_rate: float = pd.Field(
        1.2,
        description="Default growth rate for volume prism layers.",
        ge=1,
    )
    # pylint: disable=no-member
    boundary_layer_first_layer_thickness: LengthType.Positive = pd.Field(
        description="Default first layer thickness for volumetric anisotropic layers."
        " This can be overridden with :class:`~flow360.BoundaryLayer`.",
    )

    number_of_boundary_layers: Optional[pd.NonNegativeInt] = pd.Field(
        None,
        description="Default number of volumetric anisotropic layers."
        " The volume mesher will automatically calculate the required"
        " no. of layers to grow the boundary layer elements to isotropic size if not specified."
        " This is only supported by the beta mesher and can not be overridden per face.",
    )

    octree_spacing: Optional[OctreeSpacing] = pd.Field(
        None,
        description="Octree spacing configuration for volume meshing. "
        "If specified, this will be used to control the base spacing for octree-based meshers.",
    )

    @contextual_field_validator("octree_spacing", mode="after")
    @classmethod
    def _set_default_octree_spacing(cls, octree_spacing, param_info: ParamsValidationInfo):
        """Set default octree_spacing to 1 * project_length_unit when not specified."""
        if octree_spacing is not None:
            return octree_spacing
        if param_info.project_length_unit is None:
            add_validation_warning(
                "No project length unit found; `octree_spacing` will not be set automatically. "
                "Octree spacing validation will be skipped."
            )
            return octree_spacing

        # pylint: disable=no-member
        project_length = 1 * LengthType.validate(param_info.project_length_unit)
        return OctreeSpacing(base_spacing=project_length)
