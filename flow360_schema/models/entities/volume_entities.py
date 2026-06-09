"""Volume entity definitions.

Box, VoxelGrid, Cylinder, Sphere, AxisymmetricBody, GenericVolume,
CustomVolume, SeedpointVolume.
"""

# mypy: disable-error-code="import-not-found"

from __future__ import annotations

import logging
from typing import Any, Literal, final

import pydantic as pd
import unyt
from typing_extensions import Self

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_list import EntityList
from flow360_schema.framework.entity.entity_operation import (
    _extract_rotation_matrix,
    _rotation_matrix_to_axis_angle,
    _transform_direction,
    _transform_point,
    _validate_uniform_scale_and_transform_center,
    rotation_matrix_from_axis_and_angle,
)
from flow360_schema.framework.entity.entity_utils import generate_uuid
from flow360_schema.framework.entity.geometric_types import Axis
from flow360_schema.framework.multi_constructor_model_base import MultiConstructorBaseModel
from flow360_schema.framework.physical_dimensions import Angle, Length
from flow360_schema.framework.validation.context import add_validation_warning
from flow360_schema.framework.validation.validators import (
    contextual_field_validator,
    contextual_model_validator,
)
from flow360_schema.models.simulation.framework.updater_utils import deprecation_reminder

from .base import OrthogonalAxes, _VolumeEntityBase
from .surface_entities import Surface

logger = logging.getLogger(__name__)

NDArray = Any


class BoxCache(Flow360BaseModel):
    """Cached computation results for Box orientation."""

    # `axes` always exists as it needs to be used. More like storage than input cache.
    axes: OrthogonalAxes | None = pd.Field(None)
    center: Length.Vector3 | None = pd.Field(None)  # type: ignore[valid-type]
    size: Length.PositiveVector3 | None = pd.Field(None)  # type: ignore[valid-type]
    name: str | None = pd.Field(None)


@final
class Box(MultiConstructorBaseModel, _VolumeEntityBase):
    """
    Box in three-dimensional space, defined by center, size, and rotation.

    Example
    -------
    >>> Box(
    ...     name="box",
    ...     axis_of_rotation=(1, 0, 0),
    ...     angle_of_rotation=45 * unyt.deg,
    ...     center=(1, 1, 1) * unyt.m,
    ...     size=(0.2, 0.3, 2) * unyt.m,
    ... )

    Define a box using principal axes:

    >>> Box.from_principal_axes(
    ...     name="box",
    ...     axes=[(0, 1, 0), (0, 0, 1)],
    ...     center=(0, 0, 0) * unyt.m,
    ...     size=(0.2, 0.3, 2) * unyt.m,
    ... )
    """

    type_name: Literal["Box"] = pd.Field("Box", frozen=True)  # type: ignore[assignment]
    center: Length.Vector3 = pd.Field(description="The coordinates of the center of the box.")  # type: ignore[valid-type]
    size: Length.PositiveVector3 = pd.Field(description="The dimensions of the box (length, width, height).")  # type: ignore[valid-type]
    axis_of_rotation: Axis = pd.Field(  # type: ignore[assignment]
        default=(0, 0, 1),
        description="The rotation axis. Cannot change once specified.",
        frozen=True,
    )
    angle_of_rotation: Angle.Float64 = pd.Field(  # type: ignore[valid-type]
        default=0 * unyt.degree,
        description="The rotation angle. Cannot change once specified.",
        frozen=True,
    )
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)
    private_attribute_input_cache: BoxCache = pd.Field(BoxCache(), frozen=True)  # type: ignore[call-arg]
    private_attribute_entity_type_name: Literal["Box"] = pd.Field("Box", frozen=True)

    @MultiConstructorBaseModel.model_constructor
    @pd.validate_call
    def from_principal_axes(
        cls,
        name: str,
        center: Length.Vector3,  # type: ignore[valid-type]
        size: Length.PositiveVector3,  # type: ignore[valid-type]
        axes: OrthogonalAxes,
    ) -> Self:
        """Construct box from principal axes."""
        import numpy as np

        x_axis, y_axis = np.array(axes[0]), np.array(axes[1])
        z_axis = np.cross(x_axis, y_axis)

        rotation_matrix = np.transpose(np.asarray([x_axis, y_axis, z_axis], dtype=np.float64))

        # Calculate the rotation axis using eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eig(rotation_matrix)
        axis = np.real(eigvecs[:, np.where(np.isreal(eigvals))])
        if axis.shape[2] > 1:  # in case of 0 rotation angle
            axis = axis[:, :, 0]
        axis = np.ndarray.flatten(axis)

        angle = np.sum(abs(np.angle(eigvals))) / 2

        # Find correct angle sign
        matrix_test = rotation_matrix_from_axis_and_angle(axis, angle)
        angle *= -1 if np.isclose(rotation_matrix[0, :] @ matrix_test[:, 0], 1) else 1

        return cls(  # type: ignore[operator, no-any-return]
            name=name,
            center=center,
            size=size,
            axis_of_rotation=tuple(axis),
            angle_of_rotation=angle * unyt.rad,
        )

    @pd.model_validator(mode="after")
    def _convert_axis_and_angle_to_coordinate_axes(self) -> Self:
        """Convert axis-angle orientation to coordinate axes representation."""
        import numpy as np

        if not self.private_attribute_input_cache.axes:
            axis = np.asarray(self.axis_of_rotation, dtype=np.float64)
            angle = self.angle_of_rotation.to("rad").v.item()  # type: ignore[attr-defined]

            axis = axis / np.linalg.norm(axis)

            rotation_matrix = rotation_matrix_from_axis_and_angle(axis, angle)

            self.private_attribute_input_cache.axes = np.transpose(rotation_matrix[:, :2]).tolist()

        return self

    @property
    def axes(self) -> OrthogonalAxes | None:
        """Return the axes that the box is aligned with."""
        return self.private_attribute_input_cache.axes

    @pd.field_validator("center", "size", mode="after")
    @classmethod
    def _update_input_cache(cls, value: Any, info: pd.ValidationInfo) -> Any:
        setattr(info.data["private_attribute_input_cache"], info.field_name, value)  # type: ignore[arg-type]
        return value

    def _apply_transformation(self, matrix: NDArray) -> Self:
        """Apply 3x4 transformation matrix with uniform scale validation and rotation composition."""
        import numpy as np

        new_center, uniform_scale = _validate_uniform_scale_and_transform_center(matrix, self.center, "Box")

        # Combine rotations: existing rotation + transformation rotation
        existing_axis = np.asarray(self.axis_of_rotation, dtype=np.float64)
        existing_axis = existing_axis / np.linalg.norm(existing_axis)
        existing_angle = self.angle_of_rotation.to("rad").v.item()  # type: ignore[attr-defined]
        rot_existing = rotation_matrix_from_axis_and_angle(existing_axis, existing_angle)

        rot_transform = _extract_rotation_matrix(matrix)

        rot_combined = rot_transform @ rot_existing

        new_axis, new_angle = _rotation_matrix_to_axis_angle(rot_combined)

        new_size = self.size * uniform_scale

        return self.model_copy(
            update={
                "center": new_center,
                "axis_of_rotation": tuple(new_axis),
                "angle_of_rotation": new_angle * unyt.rad,
                "size": new_size,
            }
        )


@final
class VoxelGrid(_VolumeEntityBase):
    """
    Axis-aligned voxelized region used as a render target for direct volume rendering.

    Resolution is intrinsic to the voxelized 3D-texture entity (it shapes the
    underlying data, not just the rendered image), so it lives on the entity
    rather than in a render-side settings class. The grid is strictly axis-aligned
    — rotation and scene transforms are not supported on the schema side; the
    renderer applies any scene-level transform at draw time.

    Example
    -------
    >>> VoxelGrid(
    ...     name="grid",
    ...     center=(0, 0, 0) * unyt.m,
    ...     size=(1, 1, 1) * unyt.m,
    ...     resolution=(256, 256, 256),
    ... )
    """

    center: Length.Vector3 = pd.Field(description="The coordinates of the center of the region.")  # type: ignore[valid-type]
    size: Length.PositiveVector3 = pd.Field(description="The dimensions of the region (length, width, height).")  # type: ignore[valid-type]
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)
    private_attribute_entity_type_name: Literal["VoxelGrid"] = pd.Field("VoxelGrid", frozen=True)
    resolution: tuple[pd.PositiveInt, pd.PositiveInt, pd.PositiveInt] = pd.Field(
        description=(
            "Voxel grid dimensions (X, Y, Z). Memory cost scales as X*Y*Z*4 bytes; "
            "512^3 is a reasonable preview cap, 1024^3 a final-render cap."
        ),
    )


@final
class Sphere(_VolumeEntityBase):
    """
    Sphere in three-dimensional space.

    Example
    -------
    >>> Sphere(
    ...     name="sphere_zone",
    ...     center=(0, 0, 0) * unyt.m,
    ...     radius=1.5 * unyt.m,
    ...     axis=(0, 0, 1),
    ... )
    """

    private_attribute_entity_type_name: Literal["Sphere"] = pd.Field("Sphere", frozen=True)
    center: Length.Vector3 = pd.Field(description="The center point of the sphere.")  # type: ignore[valid-type]
    radius: Length.PositiveFloat64 = pd.Field(description="The radius of the sphere.")  # type: ignore[valid-type]
    axis: Axis = pd.Field(  # type: ignore[assignment]
        default=(0, 0, 1),
        description="The axis of rotation for the sphere (used in sliding interfaces).",
    )
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    def _apply_transformation(self, matrix: NDArray) -> Self:
        """Apply 3x4 transformation matrix with uniform scale validation."""
        import numpy as np

        new_center, uniform_scale = _validate_uniform_scale_and_transform_center(matrix, self.center, "Sphere")

        axis_array = np.asarray(self.axis)
        transformed_axis = _transform_direction(axis_array, matrix)
        new_axis = tuple(transformed_axis / np.linalg.norm(transformed_axis))

        new_radius = self.radius * uniform_scale

        return self.model_copy(
            update={
                "center": new_center,
                "axis": new_axis,
                "radius": new_radius,
            }
        )


@final
class Cylinder(_VolumeEntityBase):
    """
    Cylinder in three-dimensional space.

    Example
    -------
    >>> Cylinder(
    ...     name="bet_disk_volume",
    ...     center=(0, 0, 0) * unyt.inch,
    ...     axis=(0, 0, 1),
    ...     outer_radius=150 * unyt.inch,
    ...     height=15 * unyt.inch,
    ... )
    """

    private_attribute_entity_type_name: Literal["Cylinder"] = pd.Field("Cylinder", frozen=True)
    axis: Axis = pd.Field(description="The axis of the cylinder.")
    center: Length.Vector3 = pd.Field(description="The center point of the cylinder.")  # type: ignore[valid-type]
    height: Length.PositiveFloat64 = pd.Field(description="The height of the cylinder.")  # type: ignore[valid-type]
    inner_radius: Length.NonNegativeFloat64 | None = pd.Field(  # type: ignore[valid-type]
        0 * unyt.m, description="The inner radius of the cylinder."
    )
    outer_radius: Length.PositiveFloat64 = pd.Field(description="The outer radius of the cylinder.")  # type: ignore[valid-type]
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    @pd.model_validator(mode="after")
    def _check_inner_radius_is_less_than_outer_radius(self) -> Self:
        if self.inner_radius is not None and self.inner_radius >= self.outer_radius:
            raise ValueError(
                f"Cylinder inner radius ({self.inner_radius}) must be less than outer radius ({self.outer_radius})."
            )
        return self

    def _apply_transformation(self, matrix: NDArray) -> Self:
        """Apply 3x4 transformation matrix with uniform scale validation."""
        import numpy as np

        new_center, uniform_scale = _validate_uniform_scale_and_transform_center(matrix, self.center, "Cylinder")

        axis_array = np.asarray(self.axis)
        transformed_axis = _transform_direction(axis_array, matrix)
        new_axis = tuple(transformed_axis / np.linalg.norm(transformed_axis))

        new_height = self.height * uniform_scale
        new_outer_radius = self.outer_radius * uniform_scale
        new_inner_radius = self.inner_radius * uniform_scale if self.inner_radius is not None else None

        return self.model_copy(
            update={
                "center": new_center,
                "axis": new_axis,
                "height": new_height,
                "outer_radius": new_outer_radius,
                "inner_radius": new_inner_radius,
            }
        )


@final
class AxisymmetricSegment(Flow360BaseModel):
    """Reference to the region bounded by a segment of an AxisymmetricBody profile curve."""

    model_config = pd.ConfigDict(frozen=True)
    type_name: Literal["AxisymmetricSegment"] = pd.Field("AxisymmetricSegment", frozen=True)
    owning_entity_id: str = pd.Field(description="The private_attribute_id of the owning AxisymmetricBody.")
    index: int = pd.Field(ge=0, description="Index along the profile curve (0-based).")


@final
class AxisymmetricBody(_VolumeEntityBase):
    """
    Generic body of revolution, represented as (axial, radial) profile polyline
    with arbitrary center and axial direction.

    First and last profile samples must connect to axis (radius = 0).

    Example
    -------
    >>> AxisymmetricBody(
    ...     name="cone_frustum_body",
    ...     center=(0, 0, 0) * unyt.inch,
    ...     axis=(0, 0, 1),
    ...     profile_curve=[
    ...         (-1, 0) * unyt.inch,
    ...         (-1, 1) * unyt.inch,
    ...         (1, 2) * unyt.inch,
    ...         (1, 0) * unyt.inch,
    ...     ],
    ... )
    """

    private_attribute_entity_type_name: Literal["AxisymmetricBody"] = pd.Field("AxisymmetricBody", frozen=True)
    axis: Axis = pd.Field(description="The axis of the body of revolution.")
    center: Length.Vector3 = pd.Field(description="The center point of the body of revolution.")  # type: ignore[valid-type]
    profile_curve: list[Length.Vector2] = pd.Field(  # type: ignore[valid-type]
        description="The (Axial, Radial) profile of the body of revolution.",
        min_length=2,
        frozen=True,  # ensure AxisymmetricSegment references are immutable
    )
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    @pd.field_validator("profile_curve", mode="after")
    @classmethod
    def _check_radial_profile_is_positive(cls, curve: list[Length.Vector2]) -> list[Length.Vector2]:  # type: ignore[valid-type]
        first_point = curve[0]
        if first_point[1] != 0:  # type: ignore[index]
            raise ValueError(
                f"Expect first profile sample to be (Axial, 0.0). Found invalid point: {str(first_point)}."
            )

        last_point = curve[-1]
        if last_point[1] != 0:  # type: ignore[index]
            raise ValueError(f"Expect last profile sample to be (Axial, 0.0). Found invalid point: {str(last_point)}.")

        for profile_point in curve[1:-1]:
            if profile_point[1] < 0:  # type: ignore[index]
                raise ValueError(
                    f"Expect profile samples to be (Axial, Radial) samples with positive Radial."
                    f" Found invalid point: {str(profile_point)}."
                )

        return curve

    @pd.field_validator("profile_curve", mode="after")
    @classmethod
    def _check_profile_curve_has_no_duplicates(cls, curve: list[Length.Vector2]) -> list[Length.Vector2]:  # type: ignore[valid-type]
        for i in range(len(curve) - 1):
            p1, p2 = curve[i], curve[i + 1]
            if p1[0] == p2[0] and p1[1] == p2[1]:  # type: ignore[index]
                raise ValueError(
                    f"Profile curve has duplicate consecutive points at indices {i} and {i + 1}: {str(p1)}."
                )

        return curve

    def segment(self, index: int) -> AxisymmetricSegment:
        """Return an AxisymmetricSegment reference for the given profile curve segment index."""
        num_segments = len(self.profile_curve) - 1
        if index < 0 or index >= num_segments:
            raise IndexError(f"Segment index {index} out of range [0, {num_segments - 1}]")
        return AxisymmetricSegment(owning_entity_id=self.private_attribute_id, index=index)  # type: ignore[call-arg]

    def _apply_transformation(self, matrix: NDArray) -> Self:
        """Apply 3x4 transformation matrix with uniform scale validation."""
        import numpy as np

        new_center, uniform_scale = _validate_uniform_scale_and_transform_center(
            matrix, self.center, "AxisymmetricBody"
        )

        axis_array = np.asarray(self.axis)
        transformed_axis = _transform_direction(axis_array, matrix)
        new_axis = tuple(transformed_axis / np.linalg.norm(transformed_axis))

        new_profile_curve = []
        for point in self.profile_curve:
            point_array = np.asarray(point.value)  # type: ignore[attr-defined]
            scaled_point_array = point_array * uniform_scale
            new_profile_curve.append(type(point)(scaled_point_array, point.units))  # type: ignore[attr-defined, misc]

        return self.model_copy(
            update={
                "center": new_center,
                "axis": new_axis,
                "profile_curve": new_profile_curve,
            }
        )


@final
class GenericVolume(_VolumeEntityBase):
    """
    Auto-constructed volume entity from uploaded volume mesh metadata.

    Not exposed to end users. Contains only basic connectivity/mesh information.
    """

    private_attribute_entity_type_name: Literal["GenericVolume"] = pd.Field("GenericVolume", frozen=True)
    axes: OrthogonalAxes | None = pd.Field(None, description="")  # Porous media support
    axis: Axis | None = pd.Field(None)  # Rotation support
    center: Length.Vector3 | None = pd.Field(None, description="")  # type: ignore[valid-type]  # Rotation support


@final
class CustomVolume(_VolumeEntityBase):
    """Volume zone defined by its bounding entities, generated by the volume mesher."""

    private_attribute_entity_type_name: Literal["CustomVolume"] = pd.Field("CustomVolume", frozen=True)
    bounding_entities: EntityList[Surface, Cylinder, AxisymmetricBody, Sphere] = pd.Field(  # type: ignore[type-arg]
        description="The entities that define the boundaries of the custom volume."
    )
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    axes: OrthogonalAxes | None = pd.Field(None, description="")  # Porous media support
    axis: Axis | None = pd.Field(None)  # Rotation support
    center: Length.Vector3 | None = pd.Field(None, description="")  # type: ignore[valid-type]  # Rotation support

    @pd.model_validator(mode="before")
    @classmethod
    def _rename_boundaries_to_bounding_entities(cls, value: Any) -> Any:
        """Accept the legacy ``boundaries`` key and migrate to ``bounding_entities``."""
        if not isinstance(value, dict):
            return value

        if "boundaries" in value and "bounding_entities" not in value:
            value["bounding_entities"] = value.pop("boundaries")
            add_validation_warning(
                "`CustomVolume.boundaries` has been renamed to `bounding_entities`. "
                "Please update your code to use `bounding_entities`."
            )

        return value

    @contextual_field_validator("bounding_entities", mode="after")
    @classmethod
    def ensure_unique_boundary_names(
        cls,
        v: Any,
        param_info: Any,
    ) -> Any:
        """Check if the bounding entities have different names within a CustomVolume."""
        expanded_surfaces = param_info.expand_entity_list(v)
        if len(expanded_surfaces) != len({entity.name for entity in expanded_surfaces}):
            raise ValueError("The bounding entities of a CustomVolume must have different names.")
        return v

    @contextual_field_validator("bounding_entities", mode="after")
    @classmethod
    def _validate_bounding_entity_existence(cls, value: Any, param_info: Any) -> Any:
        """Ensure all boundaries will be present after mesher."""
        from flow360_schema.models.simulation.validation.validation_utils import (
            validate_entity_list_surface_existence,
        )

        return validate_entity_list_surface_existence(value, param_info)  # type: ignore[no-untyped-call]

    @contextual_model_validator(mode="after")
    def ensure_beta_mesher_and_compatible_farfield(self, param_info: Any) -> Self:
        """Check if the beta mesher is enabled and that the user is using a compatible farfield."""
        if param_info.is_beta_mesher and param_info.farfield_method in (
            "user-defined",
            "wind-tunnel",
            "auto",
        ):
            return self
        raise ValueError(
            "CustomVolume is supported only when the beta mesher is enabled "
            "and an automated, user-defined, or wind tunnel farfield is enabled."
        )

    def _apply_transformation(self, matrix: NDArray) -> Self:
        """Apply rotation from transformation matrix to axes only (no translation or scaling)."""
        import numpy as np

        if self.axes is None:
            return self

        rotation_matrix = _extract_rotation_matrix(matrix)

        x_axis_array = np.asarray(self.axes[0])
        y_axis_array = np.asarray(self.axes[1])

        new_x_axis = rotation_matrix @ x_axis_array
        new_y_axis = rotation_matrix @ y_axis_array

        new_axes = (tuple(new_x_axis), tuple(new_y_axis))

        return self.model_copy(update={"axes": new_axes})

    def _per_entity_type_validation(self, param_info: Any) -> Self:
        """Validate that CustomVolume is listed in meshing->volume_zones."""
        if self.name not in param_info.to_be_generated_custom_volumes:
            raise ValueError(
                f"CustomVolume {self.name} is not listed under meshing->volume_zones(or zones)->CustomZones."
            )
        return self


@final
class SeedpointVolume(_VolumeEntityBase):
    """
    Separate zone in the mesh, defined by one or more interior seed points.
    To be used only with snappyHexMesh.
    """

    private_attribute_entity_type_name: Literal["SeedpointVolume"] = pd.Field("SeedpointVolume", frozen=True)
    type: Literal["SeedpointVolume"] = pd.Field("SeedpointVolume", frozen=True)
    point_in_mesh: list[Length.Vector3] = pd.Field(  # type: ignore[valid-type]
        min_length=1,
        description=(
            "Seed point(s) for this custom volume zone. Accepts either one [x, y, z] point or a "
            "list of points [[x, y, z], ...]. Use with Snappy requires exactly one point per zone."
        ),
    )
    axes: OrthogonalAxes | None = pd.Field(
        None, description="Principal axes definition when using with PorousMedium"
    )  # Porous media support
    axis: Axis | None = pd.Field(None)  # Rotation support
    center: Length.Vector3 | None = pd.Field(None, description="")  # type: ignore[valid-type]  # Rotation support
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    @pd.field_validator("point_in_mesh", mode="before")
    @classmethod
    @deprecation_reminder("25.99.99")  # type: ignore[untyped-decorator]
    def _normalize_point_in_mesh(cls, value: Any) -> Any:
        """Accept a legacy single `[x, y, z]` point and wrap it as `[[x, y, z]]`.

        Slated for removal in 26 once the updater (added in 25.10.13) has
        migrated all stored configurations to the list-of-points form.
        """
        try:
            single_point = pd.TypeAdapter(Length.Vector3).validate_python(value)
        except Exception:  # pylint: disable=broad-exception-caught
            return value
        logger.warning(
            "SeedpointVolume.point_in_mesh as a single `[x, y, z]` is deprecated and "
            "will be removed in 26. Use `[[x, y, z], ...]` instead."
        )
        return [single_point]

    def _per_entity_type_validation(self, param_info: Any) -> Self:
        """Validate that SeedpointVolume is listed in meshing->volume_zones."""
        if self.name not in param_info.to_be_generated_custom_volumes:
            raise ValueError(
                f"SeedpointVolume {self.name} is not listed under meshing->volume_zones(or zones)->CustomZones."
            )
        return self

    def _apply_transformation(self, matrix: NDArray) -> Self:
        """Apply 3x4 transformation matrix to each seed point."""
        import numpy as np

        new_points = []
        for point in self.point_in_mesh:
            point_array = np.asarray(point.value)  # type: ignore[attr-defined]
            new_point_array = _transform_point(point_array, matrix)
            new_points.append(type(point)(new_point_array, point.units))  # type: ignore[attr-defined, misc]
        return self.model_copy(update={"point_in_mesh": new_points})
