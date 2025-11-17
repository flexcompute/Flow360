"""Setting groups for meshing using snappy"""

from typing import Literal, Union

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
                raise ValueError("Minimum spacing must be lower than or equal to maximum spacing.")
        return self


class QualityMetrics(Flow360BaseModel):
    """
    Mesh quality control parameters for snappyHexMesh meshing process.
    """

    # pylint: disable=no-member
    max_non_orthogonality: Union[AngleType.Positive, Literal[False]] = pd.Field(
        default=85 * u.deg,
        alias="max_non_ortho",
        description="Maximum face non-orthogonality angle: the angle made by the vector between the two adjacent "
        "cell centres across the common face and the face normal. Set to False to disable this metric.",
    )
    max_boundary_skewness: Union[AngleType.Positive, Literal[False]] = pd.Field(
        default=20 * u.deg,
        description="Maximum boundary skewness. Set to False to disable this metric.",
    )
    max_internal_skewness: Union[AngleType.Positive, Literal[False]] = pd.Field(
        default=50 * u.deg,
        description="Maximum internal face skewness. Set to False to disable this metric.",
    )
    max_concavity: Union[AngleType.Positive, Literal[False]] = pd.Field(
        default=50 * u.deg,
        alias="max_concave",
        description="Maximum cell concavity. Set to False to disable this metric.",
    )
    min_pyramid_cell_volume: Union[float, Literal[False]] = pd.Field(
        default=False,
        alias="min_vol",
        description="Minimum cell pyramid volume [m³]. Set to False to disable this metric (uses -1e30 internally).",
    )
    min_tetrahedron_quality: Union[float, Literal[False]] = pd.Field(
        default=False,
        alias="min_tet_quality",
        description="Minimum tetrahedron quality. Set to False to disable this metric (uses -1e30 internally).",
    )
    min_face_area: Union[AreaType.Positive, Literal[False]] = pd.Field(
        default=False, alias="min_area", description="Minimum face area. Set to False to disable."
    )
    min_twist: Union[float, Literal[False]] = pd.Field(
        default=False,
        description="Minimum twist. Controls the twist quality of faces. Set to False to disable this metric.",
    )
    min_cell_determinant: Union[float, Literal[False]] = pd.Field(
        default=False,
        alias="min_determinant",
        description="Minimum cell determinant. Set to False to disable this metric (uses -1e5 internally).",
    )
    min_volume_ratio: Union[pd.NonNegativeFloat, Literal[False]] = pd.Field(
        default=0, alias="min_vol_ratio", description="Minimum volume ratio between adjacent cells."
    )
    min_face_weight: Union[pd.NonNegativeFloat, Literal[False]] = pd.Field(
        default=0,
        description="Minimum face interpolation weight. Controls the quality of face interpolation.",
    )
    min_triangle_twist: Union[float, Literal[False]] = pd.Field(
        default=False, description="Minimum triangle twist. Set to False to disable this metric."
    )
    n_smooth_scale: pd.NonNegativeInt = pd.Field(
        default=4,
        description="Number of smoothing iterations. Used in combination with error_reduction.",
    )
    error_reduction: float = pd.Field(
        default=0.75,
        ge=0,
        le=1,
        description="Error reduction factor. Used in combination with n_smooth_scale. Must be between 0 and 1.",
    )
    min_volume_collapse_ratio: Union[float, Literal[False]] = pd.Field(
        default=0,
        alias="min_vol_collapse_ratio",
        description="Minimum volume collapse ratio. If > 0: preserves single cells with all pointson the surface "
        "if the resulting volume after snapping is larger than min_vol_collapse_ratio "
        "times the old volume (i.e., not collapsed to flat cell). If < 0: always deletes such cells.",
    )

    @pd.field_validator(
        "max_non_orthogonality",
        "max_concavity",
        "max_boundary_skewness",
        "max_internal_skewness",
        mode="after",
    )
    @classmethod
    def disable_angle_metrics_w_defaults(cls, value):
        """Maximum value with units for angle metrics."""
        if value and value > 180 * u.deg:
            raise ValueError("Value must be less than or equal to 180 degrees.")
        return value


class CastellatedMeshControls(Flow360BaseModel):
    """
    snappyHexMesh castellation controls.
    """

    # pylint: disable=no-member
    resolve_feature_angle: AngleType.Positive = pd.Field(
        default=25 * u.deg,
        description="This parameter controls the local curvature refinement. "
        "The higher the value, the less features it captures. "
        "Applies maximum level of refinement to cells that can see intersections whose angle exceeds this value.",
    )
    n_cells_between_levels: pd.NonNegativeInt = pd.Field(
        1,
        description="This parameter controls the transition between cell refinement levels. "
        "Number of buffer layers of cells between different levels of refinement.",
    )
    min_refinement_cells: pd.NonNegativeInt = pd.Field(
        10,
        description="The refinement along the surfaces may spend many iterations on refinement of only few cells. "
        "Whenever the number of cells to be refined is less than or equal to this value, the refinement will stop. "
        "Unless the parameter is set to zero, at least one refining iteration will be performed.",
    )

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
    """

    # pylint: disable=no-member
    n_smooth_patch: pd.NonNegativeInt = pd.Field(
        3,
        description="Number of patch smoothing iterations before finding correspondence to surface.",
    )
    tolerance: pd.PositiveFloat = pd.Field(
        2,
        description="Ratio of distance for points to be attracted by surface feature point or edge, "
        "to local maximum edge length.",
    )
    n_solve_iterations: pd.NonNegativeInt = pd.Field(
        30, alias="n_solve_iter", description="Number of mesh displacement relaxation iterations."
    )
    n_relax_iterations: pd.NonNegativeInt = pd.Field(
        5,
        alias="n_relax_iter",
        description="Number of relaxation iterations during the snapping. "
        "If the mesh does not conform the geometry and all the iterations are spend, "
        "user may try to increase the number of iterations.",
    )
    n_feature_snap_iterations: pd.NonNegativeInt = pd.Field(
        15,
        alias="n_feature_snap_iter",
        description="Number of relaxation iterations used for snapping onto the features."
        " If not specified, feature snapping will be disabled.",
    )
    multi_region_feature_snap: bool = pd.Field(
        True,
        description="When using explicitFeatureSnap and this switch is on, "
        "features between multiple surfaces will be captured. "
        "This is useful for multi-region meshing where the internal mesh "
        "must conform the region geometrical boundaries.",
    )
    strict_region_snap: bool = pd.Field(
        False,
        description="Attract points only to the surface they originate from. "
        "This can improve snapping of intersecting surfaces.",
    )


class SmoothControls(Flow360BaseModel):
    """
    Mesh smoothing controls.
    """

    # pylint: disable=no-member
    lambda_factor: pd.NonNegativeFloat = pd.Field(
        0.7, le=1, description="Controls the strength of smoothing in a single iteration."
    )
    mu_factor: pd.NonNegativeFloat = pd.Field(
        0.71,
        le=1,
        description="Controls the strength of geometry inflation during a single iteration. "
        "It is reccomended to set mu to be a little higher than lambda.",
    )
    iterations: pd.NonNegativeInt = pd.Field(5, description="Number of smoothing iterations.")
