"""Setting groups for meshing using snappy"""

from typing import Optional

import pydantic as pd
from typing_extensions import Self

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import AngleType, AreaType, LengthType


class SurfaceMeshingDefaults(Flow360BaseModel):
    """
    Default/global settings for snappyHexMesh surface meshing parameters.
    To be used with class:`ModularMeshingWorkflow`.
    """

    # pylint: disable=no-member
    min_spacing: LengthType.Positive = pd.Field()
    max_spacing: LengthType.Positive = pd.Field()
    gap_resolution: LengthType.Positive = pd.Field()

    @pd.model_validator(mode="after")
    def _check_spacing_order(self) -> Self:
        if self.min_spacing and self.max_spacing:
            if self.min_spacing > self.max_spacing:
                raise ValueError("Minimum spacing must be lower than maximum spacing.")
        return self


class QualityMetrics(Flow360BaseModel):
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

    # pylint: disable=no-member
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
        """Disable a quality metric in OpenFOAM by setting a specific value."""
        if value is None:
            return 180 * u.deg
        if value > 180 * u.deg:
            raise ValueError("Value must be less that 180 degrees.")
        return value

    @pd.field_validator("max_boundary_skewness", "max_internal_skewness", mode="after")
    @classmethod
    def disable_skewness_metric(cls, value):
        """Disable a quality metric in OpenFOAM by setting a specific value."""
        if value is None:
            return -1 * u.deg
        if value.to("degree") <= 0 * u.deg and value.to("degree") != -1 * u.deg:
            raise ValueError(
                f"Maximum skewness must be positive (your value: {value}). To disable enter None or -1*u.deg."
            )
        return value

    @pd.field_validator("min_vol", "min_tet_quality", "min_determinant", mode="after")
    @classmethod
    def disable_by_low_value(cls, value):
        """Disable a quality metric in OpenFOAM by setting a specific value."""
        if value is None:
            return -1e30
        return value

    @pd.field_validator("n_smooth_scale", "error_reduction", mode="after")
    @classmethod
    def disable_by_zero(cls, value):
        """Disable a quality metric in OpenFOAM by setting a specific value."""
        if value is None:
            return 0
        return value


class CastellatedMeshControls(Flow360BaseModel):
    """
    snappyHexMesh castellation controls.

    Parameters
    ----------
    resolve_feature_angle : Optional[AngleType.Positive], default: 25°
        This parameter controls the local curvature refinement. The higher the value,
        the less features it captures. Applies maximum level of refinement to cells
        that can see intersections whose angle exceeds this value.

    n_cells_between_levels: Optional[pd.NonNegativeInt], default: 1
        This parameter controls the transition between cell refinement levels. Number
        of buffer layers of cells between different levels of refinement.

    min_refinement_cells: Optional[pd.NonNegativeInt], default: 10
        The refinement along the surfaces may spend many iterations on refinement of
        only few cells. Whenever the number of cells to be refined is less than or equal
        to this value, the refinement will stop. Unless the parameter is set to zero,
        at least one refining iteration will be performed.
    """

    # pylint: disable=no-member
    resolve_feature_angle: Optional[AngleType.Positive] = pd.Field(default=25 * u.deg)
    n_cells_between_levels: Optional[pd.NonNegativeInt] = pd.Field(1)
    min_refinement_cells: Optional[pd.NonNegativeInt] = pd.Field(10)

    @pd.field_validator("resolve_feature_angle", mode="after")
    @classmethod
    def angle_limits(cls, value):
        """Limit angular values."""
        if value is None:
            return value
        if value > 180 * u.deg:
            raise ValueError("resolve_feature_angle must be between 0 and 180 degrees.")
        return value


class SnapControls(Flow360BaseModel):
    """
    snappyHexMesh snap controls.

    Parameters
    ----------
    n_smooth_patch: pd.NonNegativeInt, default: 3
        Number of patch smoothing iterations before finding correspondence to surface.

    tolerance: pd.PositiveFloat, default: 2
        Ratio of distance for points to be attracted by surface feature point or edge,
        to local maximum edge length.

    n_solve_iter: pd.NonNegativeInt, default: 30
        Number of mesh displacement relaxation iterations

    n_relax_iter: pd.NonNegativeInt, default: 5
        Number of relaxation iterations during the snapping. If the mesh does not conform the geometry
        and all the iterations are spend, user may try to increase the number of iterations.

    n_feature_snap_iter: pd.NonNegativeInt, default: 15
        Number of relaxation iterations used for snapping onto the features.
        If not specified, feature snapping will be disabled.

    multi_region_feature_snap: bool, default: True
        When using explicitFeatureSnap and this switch is on, features between multiple
        surfaces will be captured. This is useful for multi-region meshing where the internal
        mesh must conform the region geometrical boundaries.

    strict_region_snap: bool, default: False
        Attract points only to the surface they originate from. This can improve snapping of
        intersecting surfaces.
    """

    # pylint: disable=no-member
    n_smooth_patch: pd.NonNegativeInt = pd.Field(3)
    tolerance: pd.PositiveFloat = pd.Field(2)
    n_solve_iter: pd.NonNegativeInt = pd.Field(30)
    n_relax_iter: pd.NonNegativeInt = pd.Field(5)
    n_feature_snap_iter: pd.NonNegativeInt = pd.Field(15)
    multi_region_feature_snap: bool = pd.Field(True)
    strict_region_snap: bool = pd.Field(False)


class SmoothControls(Flow360BaseModel):
    """
    snappyHexMesh smoothing controls.

    Parameters
    ----------
    lambda_factor: Optional[pd.NonNegativeFloat], default: 0.7
        Lambda value within [0,1]

    mu_factor: Optional[pd.NonNegativeFloat], default: 0.71
        Mu value within [0,1]

    iterations: Optional[pd.NonNegativeInt], default: 5
        Number of smoothing iterations
    """

    # pylint: disable=no-member
    lambda_factor: Optional[pd.NonNegativeFloat] = pd.Field(0.7)
    mu_factor: Optional[pd.NonNegativeFloat] = pd.Field(0.71)
    iterations: Optional[pd.NonNegativeInt] = pd.Field(5)

    @pd.field_validator("iterations", mode="after")
    @classmethod
    def disable_by_zero(cls, value):
        """Disable a quality metric in OpenFOAM by setting a specific valuesmoothing when None is set."""
        if value is None:
            return 0
        return value
