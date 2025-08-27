"""Default settings for meshing using different meshing algorithms"""

from typing import Optional

import pydantic as pd
from typing_extensions import Self

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.updater import DEFAULT_PLANAR_FACE_TOLERANCE
from flow360.component.simulation.unit_system import AngleType, AreaType, LengthType
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
        "surface_max_aspect_ratio", "surface_max_adaptation_iterations", mode="after"
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


class BetaVolumeMeshingDefaults(Flow360BaseModel):
    ##::    Default boundary layer settings
    boundary_layer_growth_rate: float = pd.Field(
        1.2,
        description="Default growth rate for volume prism layers.",
        ge=1,
    )
    # pylint: disable=no-member
    boundary_layer_first_layer_thickness: Optional[LengthType.Positive] = pd.Field(
        None,
        description="Default first layer thickness for volumetric anisotropic layers."
        " This can be overridden with :class:`~flow360.BoundaryLayer`.",
    )  # Truly optional if all BL faces already have first_layer_thickness

    gap_treatment_strength: Optional[float] = pd.Field(
        default=0,
        ge=0,
        le=1,
        description="Narrow gap treatment strength used when two surfaces are in close proximity."
        " Use a value between 0 and 1, where 0 is no treatment and 1 is the most conservative treatment."
        " This parameter has a global impact where the anisotropic transition into the isotropic mesh."
        " However the impact on regions without close proximity is negligible.",
    )

    number_of_boundary_layers: Optional[pd.NonNegativeInt] = pd.Field(
        None,
        description="Default number of volumetric anisotropic layers."
        " The volume mesher will automatically calculate the required"
        " no. of layers to grow the boundary layer elements to isotropic size if not specified."
        " This is only supported by the beta mesher and can not be overridden per face.",
    )


class SnappySurfaceMeshingDefaults(Flow360BaseModel):
    min_spacing: LengthType.Positive = pd.Field()
    max_spacing: LengthType.Positive = pd.Field()
    gap_resolution: LengthType.Positive = pd.Field()

    @pd.model_validator(mode="after")
    def _check_spacing_order(self) -> Self:
        if self.min_spacing and self.max_spacing:
            if self.min_spacing > self.max_spacing:
                raise ValueError("Minimum spacing must be lower than maximum spacing.")
        return self


class SnappyQualityMetrics(Flow360BaseModel):
    """
    Mesh quality control parameters for snappyHexMesh meshing process.

    Parameters
    ----------
    max_non_ortho : Optional[AngleType.Positive], default: 85°
        Maximum face non-orthogonality angle: the angle made by the vector between
        the two adjacent cell centres across the common face and the face normal.
        Set to None to disable this metric.

    max_boundary_skewness : Optional[AngleType], default: 20°
        Maximum boundary skewness. Set to None or -1° to disable this metric.

    max_internal_skewness : Optional[AngleType], default: 50°
        Maximum internal face skewness. Set to None or -1° to disable this metric.

    max_concave : Optional[AngleType.Positive], default: 50°
        Maximum cell concavity. Set to None to disable this metric.

    min_vol : Optional[float], default: None
        Minimum cell pyramid volume [m³].
        Set to None to disable this metric (uses -1e30 internally).

    min_tet_quality : Optional[float], default: None
        Minimum tetrahedron quality.
        Set to None to disable this metric (uses -1e30 internally).

    min_area : Optional[AreaType.Positive], default: None
        Minimum face area. Set to None to disable.

    min_twist : Optional[float], default: None
        Minimum twist. Controls the twist quality of faces.
        Set to None to disable this metric.

    min_determinant : Optional[float], default: None
        Minimum cell determinant. Set to None to disable this metric (uses -1e30 internally).

    min_vol_ratio : float, default: 0
        Minimum volume ratio between adjacent cells.

    min_face_weight : float, default: 0
        Minimum face interpolation weight. Controls the quality of face interpolation.

    min_triangle_twist : Optional[float], default: None
        Minimum triangle twist. Set to None to disable this metric.

    n_smooth_scale : Optional[pd.NonNegativeInt], default: 4
        Number of smoothing iterations. Used in combination with error_reduction.

    error_reduction : Optional[float], default: 0.75
        Error reduction factor. Used in combination with n_smooth_scale.
        Must be between 0 and 1.

    min_vol_collapse_ratio : float, default: 0
        Minimum volume collapse ratio. If > 0: preserves single cells with all points
        on the surface if the resulting volume after snapping is larger than
        min_vol_collapse_ratio times the old volume (i.e., not collapsed to flat cell).
        If < 0: always deletes such cells.
    """

    max_non_ortho: Optional[AngleType.Positive] = pd.Field(default=85 * u.deg)
    max_boundary_skewness: Optional[AngleType] = pd.Field(default=20 * u.deg)
    max_internal_skewness: Optional[AngleType] = pd.Field(default=50 * u.deg)
    max_concave: Optional[AngleType.Positive] = pd.Field(default=50 * u.deg)
    min_vol: Optional[float] = pd.Field(default=None)
    min_tet_quality: Optional[float] = pd.Field(default=None)
    min_area: Optional[AreaType.Positive] = pd.Field(default=None)
    min_twist: Optional[float] = pd.Field(default=None)
    min_determinant: Optional[float] = pd.Field(default=None)
    min_vol_ratio: Optional[float] = pd.Field(default=0)
    min_face_weight: Optional[float] = pd.Field(default=0)
    min_triangle_twist: Optional[float] = pd.Field(default=None)
    n_smooth_scale: Optional[pd.NonNegativeInt] = pd.Field(default=4, ge=0)
    error_reduction: Optional[float] = pd.Field(default=0.75, ge=0, le=1)
    min_vol_collapse_ratio: Optional[float] = pd.Field(0)

    @pd.field_validator("max_non_ortho", "max_concave", mode="after")
    @classmethod
    def disable_angle_metrics_w_defaults(cls, value):
        if value is None:
            return 180 * u.deg
        if value > 180 * u.deg:
            raise ValueError("Value must be less that 180 degrees.")
        else:
            return value

    @pd.field_validator("max_boundary_skewness", "max_internal_skewness", mode="after")
    @classmethod
    def disable_skewness_metric(cls, value):
        if value is None:
            return -1 * u.deg
        if value.to("degree") <= 0 * u.deg and value.to("degree") != -1 * u.deg:
            raise ValueError(
                f"Maximum skewness must be positive (your value: {value}). To disable enter None or -1*u.deg."
            )
        else:
            return value

    @pd.field_validator("min_vol", "min_tet_quality", "min_determinant", mode="after")
    @classmethod
    def disable_by_low_value(cls, value):
        if value is None:
            return -1e30
        else:
            return value

    @pd.field_validator("n_smooth_scale", "error_reduction", mode="after")
    @classmethod
    def disable_by_zero(cls, value):
        if value is None:
            return 0
        else:
            return value


class SnappyCastellatedMeshControls(Flow360BaseModel):
    resolve_feature_angle: Optional[AngleType.Positive] = pd.Field(default=25 * u.deg)
    n_cells_between_levels: Optional[pd.NonNegativeInt] = pd.Field(1)
    min_refinement_cells: Optional[pd.NonNegativeInt] = pd.Field(10)

    @pd.field_validator("resolve_feature_angle", mode="after")
    @classmethod
    def angle_limits(cls, value):
        if value is None:
            return value
        if value > 180 * u.deg:
            raise ValueError("resolve_feature_angle must be between 0 and 180 degrees.")
        return value


class SnappySnapControls(Flow360BaseModel):
    n_smooth_patch: pd.NonNegativeInt = pd.Field(3)
    tolerance: pd.PositiveFloat = pd.Field(2)
    n_solve_iter: pd.NonNegativeInt = pd.Field(30)
    n_relax_iter: pd.NonNegativeInt = pd.Field(5)
    n_feature_snap_iter: pd.NonNegativeInt = pd.Field(15)
    multi_region_feature_snap: bool = pd.Field(True)
    strict_region_snap: bool = pd.Field(False)


class SnappySmoothControls(Flow360BaseModel):
    lambda_factor: Optional[pd.NonNegativeFloat] = pd.Field(0.7)
    mu_factor: Optional[pd.NonNegativeFloat] = pd.Field(0.71)
    iterations: Optional[pd.NonNegativeInt] = pd.Field(5)
    min_elem: Optional[pd.NonNegativeInt] = pd.Field(None)
    min_len: Optional[LengthType.NonNegative] = pd.Field(None)
    included_angle: Optional[AngleType.NonNegative] = pd.Field(150 * u.deg)
