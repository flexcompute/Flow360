"""Default settings for meshing using different meshing algorithms"""

from math import log2
from typing import Optional

import numpy as np
import pydantic as pd

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.updater import DEFAULT_PLANAR_FACE_TOLERANCE
from flow360.component.simulation.unit_system import AngleType, LengthType
from flow360.component.simulation.validation.validation_context import (
    SURFACE_MESH,
    VOLUME_MESH,
    ConditionalField,
    ContextField,
    get_validation_info,
)


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

    planar_face_tolerance: Optional[pd.NonNegativeFloat] = pd.Field(
        DEFAULT_PLANAR_FACE_TOLERANCE,
        description="Tolerance used for detecting planar faces in the input surface mesh / geometry"
        " that need to be remeshed, such as symmetry planes."
        " This tolerance is non-dimensional, and represents a distance"
        " relative to the largest dimension of the bounding box of the input surface mesh / geometry."
        " This can not be overridden per face.",
    )

    ##::    Default surface layer settings
    surface_max_edge_length: Optional[LengthType.Positive] = ConditionalField(
        None,
        description="Default maximum edge length for surface cells."
        " This can be overridden with :class:`~flow360.SurfaceRefinement`.",
        context=SURFACE_MESH,
    )

    surface_max_aspect_ratio: Optional[pd.PositiveFloat] = ConditionalField(
        10.0,
        description="Maximum aspect ratio for surface cells for the GAI surface mesher."
        " This cannot be overridden per face",
        context=SURFACE_MESH,
    )

    surface_max_adaptation_iterations: Optional[pd.NonNegativeInt] = ConditionalField(
        50,
        description="Maximum adaptation iterations for the GAI surface mesher.",
        context=SURFACE_MESH,
    )

    curvature_resolution_angle: Optional[AngleType.Positive] = ContextField(
        12 * u.deg,
        description=(
            "Default maximum angular deviation in degrees. This value will restrict:"
            " 1. The angle between a cell’s normal and its underlying surface normal."
            " 2. The angle between a line segment’s normal and its underlying curve normal."
            " This can not be overridden per face."
        ),
        context=SURFACE_MESH,
    )

    preserve_thin_geometry: Optional[bool] = pd.Field(
        False,
        description="Flag to specify whether thin geometry features with thickness roughly equal "
        + "to geometry_accuracy should be resolved accurately during the surface meshing process."
        + "This can be overridden with class: ~flow360.GeometryRefinement",
    )

    sealing_size: LengthType.NonNegative = pd.Field(
        0.0 * u.m,
        description="Threshold size below which all geometry gaps are automatically closed. "
        + "This can be overridden with class: ~flow360.GeometryRefinement",
    )

    @pd.field_validator("number_of_boundary_layers", mode="after")
    @classmethod
    def invalid_number_of_boundary_layers(cls, value):
        """Ensure number of boundary layers is not specified"""
        validation_info = get_validation_info()

        if validation_info is None:
            return value

        if value is not None and not validation_info.is_beta_mesher:
            raise ValueError("Number of boundary layers is only supported by the beta mesher.")
        return value

    @pd.field_validator("geometry_accuracy", mode="after")
    @classmethod
    def invalid_geometry_accuracy(cls, value):
        """Ensure geometry accuracy is not specified when GAI is not used"""
        validation_info = get_validation_info()

        if validation_info is None:
            return value

        if value is not None and not validation_info.use_geometry_AI:
            raise ValueError("Geometry accuracy is only supported when geometry AI is used.")

        if value is None and validation_info.use_geometry_AI:
            raise ValueError("Geometry accuracy is required when geometry AI is used.")
        return value

    @pd.field_validator(
        "surface_max_aspect_ratio",
        "surface_max_adaptation_iterations",
        "preserve_thin_geometry",
        "sealing_size",
        mode="after",
    )
    @classmethod
    def invalid_geometry_ai_features(cls, value, info):
        """Ensure surface max aspect ratio is not specified when GAI is not used"""
        validation_info = get_validation_info()

        if validation_info is None:
            return value

        # pylint: disable=unsubscriptable-object
        default_value = cls.model_fields[info.field_name].default
        if value != default_value and not validation_info.use_geometry_AI:
            raise ValueError(f"{info.field_name} is only supported when geometry AI is used.")

        return value


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
