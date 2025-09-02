"""
Primitive type definitions for simulation entities.
"""

import re
from abc import ABCMeta
from typing import Annotated, List, Literal, Optional, Tuple, Union, final

import numpy as np
import pydantic as pd
from pydantic import PositiveFloat
from scipy.linalg import eig
from typing_extensions import Self

import flow360.component.simulation.units as u
from flow360.component.simulation.entity_operation import (
    Transformation,
    rotation_matrix_from_axis_and_angle,
)
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import (
    EntityBase,
    EntityList,
    generate_uuid,
)
from flow360.component.simulation.framework.multi_constructor_model_base import (
    MultiConstructorBaseModel,
)
from flow360.component.simulation.framework.unique_list import UniqueStringList
from flow360.component.simulation.unit_system import AngleType, AreaType, LengthType
from flow360.component.simulation.user_code.core.types import ValueOrExpression
from flow360.component.simulation.utils import BoundingBoxType, model_attribute_unlock
from flow360.component.simulation.validation.validation_context import (
    get_validation_info,
)
from flow360.component.types import Axis

BOUNDARY_FULL_NAME_WHEN_NOT_FOUND = "This boundary does not exist!!!"


def _get_boundary_full_name(surface_name: str, volume_mesh_meta: dict[str, dict]) -> str:
    """Ideally volume_mesh_meta should be a pydantic model.

    TODO:  Note that the same surface_name may appear in different blocks. E.g.
    `farFieldBlock/slipWall`, and `plateBlock/slipWall`. Currently the mesher does not support splitting boundary into
    blocks but we will need to support this someday.
    """
    for zone_name, zone_meta in volume_mesh_meta["zones"].items():
        for existing_boundary_name in zone_meta["boundaryNames"]:
            pattern = re.escape(zone_name) + r"/(.*)"
            match = re.search(pattern, existing_boundary_name)
            if (
                match is not None and match.group(1) == surface_name
            ) or existing_boundary_name == surface_name:
                return existing_boundary_name

    # Not found
    return BOUNDARY_FULL_NAME_WHEN_NOT_FOUND


def _check_axis_is_orthogonal(axis_pair: Tuple[Axis, Axis]) -> Tuple[Axis, Axis]:
    axis_1, axis_2 = np.array(axis_pair[0]), np.array(axis_pair[1])
    dot_product = np.dot(axis_1, axis_2)
    if not np.isclose(dot_product, 0):
        raise ValueError(f"The two axes are not orthogonal, dot product is {dot_product}.")
    return axis_pair


OrthogonalAxes = Annotated[Tuple[Axis, Axis], pd.AfterValidator(_check_axis_is_orthogonal)]


class ReferenceGeometry(Flow360BaseModel):
    """
    :class:`ReferenceGeometry` class contains all geometrical related reference values.

    Example
    -------
    >>> ReferenceGeometry(
    ...     moment_center=(1, 2, 1) * u.m,
    ...     moment_length=(1, 1, 1) * u.m,
    ...     area=1.5 * u.m**2
    ... )
    >>> ReferenceGeometry(
    ...     moment_center=(1, 2, 1) * u.m,
    ...     moment_length=1 * u.m,
    ...     area=1.5 * u.m**2
    ... )  # Equivalent to above

    ====
    """

    # pylint: disable=no-member
    moment_center: Optional[LengthType.Point] = pd.Field(
        None, description="The x, y, z coordinate of moment center."
    )
    moment_length: Optional[Union[LengthType.Positive, LengthType.PositiveVector]] = pd.Field(
        None, description="The x, y, z component-wise moment reference lengths."
    )
    area: Optional[ValueOrExpression[AreaType.Positive]] = pd.Field(
        None, description="The reference area of the geometry."
    )
    private_attribute_area_settings: Optional[dict] = pd.Field(None)


class GeometryBodyGroup(EntityBase):
    """
    :class:`GeometryBodyGroup` represents a collection of bodies that are grouped for transformation.
    """

    private_attribute_registry_bucket_name: Literal["GeometryBodyGroupEntityType"] = (
        "GeometryBodyGroupEntityType"
    )
    private_attribute_tag_key: str = pd.Field(
        description="The tag/attribute string used to group bodies.",
    )
    private_attribute_entity_type_name: Literal["GeometryBodyGroup"] = pd.Field(
        "GeometryBodyGroup", frozen=True
    )
    private_attribute_sub_components: List[str] = pd.Field(
        description="A list of body IDs which constitutes the current body group"
    )
    private_attribute_color: Optional[str] = pd.Field(
        None, description="Color used for visualization"
    )
    transformation: Transformation = pd.Field(
        Transformation(), description="The transformation performed on the body group"
    )
    mesh_exterior: bool = pd.Field(
        True,
        description="Option to define whether to mesh exterior or interior of body group in geometry AI."
        "Note that this is a beta feature and the interface might change in future releases.",
    )


class _VolumeEntityBase(EntityBase, metaclass=ABCMeta):
    """All volumetric entities should inherit from this class."""

    ### Warning: Please do not change this as it affects registry bucketing.
    private_attribute_registry_bucket_name: Literal["VolumetricEntityType"] = "VolumetricEntityType"
    private_attribute_zone_boundary_names: UniqueStringList = pd.Field(
        UniqueStringList(),
        frozen=True,
        description="Boundary names of the zone WITH the prepending zone name.",
    )
    private_attribute_full_name: Optional[str] = pd.Field(None, frozen=True)

    def _is_volume_zone(self) -> bool:
        """This is not a zone if zone boundaries are not defined. For validation usage."""
        return self.private_attribute_zone_boundary_names is not None

    def _update_entity_info_with_metadata(self, volume_mesh_meta_data: dict[str, dict]) -> None:
        """
        Update the full name of zones once the volume mesh is done.
        e.g. rotating_cylinder --> rotatingBlock-rotating_cylinder
        """
        entity_name = self.name
        for zone_full_name, zone_meta in volume_mesh_meta_data["zones"].items():
            pattern = r"rotatingBlock-" + re.escape(entity_name)
            if entity_name == "__farfield_zone_name_not_properly_set_yet":
                # We have hardcoded name for farfield zone.
                pattern = r"stationaryBlock|fluid"
            match = re.search(pattern, zone_full_name)
            if match is not None or entity_name == zone_full_name:
                with model_attribute_unlock(self, "private_attribute_full_name"):
                    self.private_attribute_full_name = zone_full_name
                with model_attribute_unlock(self, "private_attribute_zone_boundary_names"):
                    self.private_attribute_zone_boundary_names = UniqueStringList(
                        items=zone_meta["boundaryNames"]
                    )
                break

    @property
    def full_name(self):
        """Gets the full name which includes the zone name"""
        if self.private_attribute_full_name is None:
            return self.name
        return self.private_attribute_full_name


class _SurfaceEntityBase(EntityBase, metaclass=ABCMeta):
    ### Warning: Please do not change this as it affects registry bucketing.
    private_attribute_registry_bucket_name: Literal["SurfaceEntityType"] = "SurfaceEntityType"
    private_attribute_full_name: Optional[str] = pd.Field(None, frozen=True)

    def _update_entity_info_with_metadata(self, volume_mesh_meta_data: dict) -> None:
        """
        Update parent zone name once the volume mesh is done.
        """
        with model_attribute_unlock(self, "private_attribute_full_name"):
            self.private_attribute_full_name = _get_boundary_full_name(
                self.name, volume_mesh_meta_data
            )

    @property
    def full_name(self):
        """Gets the full name which includes the zone name"""
        if self.private_attribute_full_name is None:
            return self.name
        return self.private_attribute_full_name


class _EdgeEntityBase(EntityBase, metaclass=ABCMeta):
    ### Warning: Please do not change this as it affects registry bucketing.
    private_attribute_registry_bucket_name: Literal["EdgeEntityType"] = "EdgeEntityType"


@final
class Edge(_EdgeEntityBase):
    """
    Edge which contains a set of grouped edges from geometry.
    """

    ### Warning: Please do not change this as it affects registry bucketing.
    private_attribute_registry_bucket_name: Literal["EdgeEntityType"] = pd.Field(
        "EdgeEntityType", frozen=True
    )
    private_attribute_entity_type_name: Literal["Edge"] = pd.Field("Edge", frozen=True)
    private_attribute_tag_key: Optional[str] = pd.Field(
        None,
        description="The tag/attribute string used to group geometry edges to form this `Edge`.",
    )
    private_attribute_sub_components: Optional[List[str]] = pd.Field(
        [], description="The edge ids in geometry that composed into this `Edge`."
    )


@final
class GenericVolume(_VolumeEntityBase):
    """
    Do not expose.
    This type of entity will get auto-constructed by assets when loading metadata.
    By design these GenericVolume entities should only contain basic connectivity/mesh information.
    """

    private_attribute_entity_type_name: Literal["GenericVolume"] = pd.Field(
        "GenericVolume", frozen=True
    )
    axes: Optional[OrthogonalAxes] = pd.Field(None, description="")  # Porous media support
    axis: Optional[Axis] = pd.Field(None)  # Rotation support
    # pylint: disable=no-member
    center: Optional[LengthType.Point] = pd.Field(None, description="")  # Rotation support


class BoxCache(Flow360BaseModel):
    """BoxCache"""

    # `axes` will always exist as it needs to be used. So `axes` is more like a storage than input cache.
    axes: Optional[OrthogonalAxes] = pd.Field(None)
    # pylint: disable=no-member
    center: Optional[LengthType.Point] = pd.Field(None)
    size: Optional[LengthType.PositiveVector] = pd.Field(None)
    name: Optional[str] = pd.Field(None)


@final
class Box(MultiConstructorBaseModel, _VolumeEntityBase):
    """
    :class:`Box` class represents a box in three-dimensional space.

    Example
    -------
    >>> fl.Box(
    ...     name="box",
    ...     axis_of_rotation = (1, 0, 0),
    ...     angle_of_rotation = 45 * fl.u.deg,
    ...     center = (1, 1, 1) * fl.u.m,
    ...     size=(0.2, 0.3, 2) * fl.u.m,
    ... )

    Define a box using principal axes:

    >>> fl.Box.from_principal_axes(
    ...     name="box",
    ...     axes=[(0, 1, 0), (0, 0, 1)],
    ...     center=(0, 0, 0) * fl.u.m,
    ...     size=(0.2, 0.3, 2) * fl.u.m,
    ... )

    ====
    """

    type_name: Literal["Box"] = pd.Field("Box", frozen=True)
    # pylint: disable=no-member
    center: LengthType.Point = pd.Field(description="The coordinates of the center of the box.")
    size: LengthType.PositiveVector = pd.Field(
        description="The dimensions of the box (length, width, height)."
    )
    axis_of_rotation: Axis = pd.Field(
        default=(0, 0, 1),
        description="The rotation axis. Cannot change once specified.",
        frozen=True,
    )
    angle_of_rotation: AngleType = pd.Field(
        default=0 * u.degree,
        description="The rotation angle. Cannot change once specified.",
        frozen=True,
    )
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)
    private_attribute_input_cache: BoxCache = pd.Field(BoxCache(), frozen=True)
    private_attribute_entity_type_name: Literal["Box"] = pd.Field("Box", frozen=True)

    # pylint: disable=no-self-argument
    @MultiConstructorBaseModel.model_constructor
    @pd.validate_call
    def from_principal_axes(
        cls,
        name: str,
        center: LengthType.Point,
        size: LengthType.PositiveVector,
        axes: OrthogonalAxes,
    ):
        """
        Construct box from principal axes
        """
        # validate
        x_axis, y_axis = np.array(axes[0]), np.array(axes[1])
        z_axis = np.cross(x_axis, y_axis)

        rotation_matrix = np.transpose(np.asarray([x_axis, y_axis, z_axis], dtype=np.float64))

        # Calculate the rotation axis n
        eig_rotation = eig(rotation_matrix)
        axis = np.real(eig_rotation[1][:, np.where(np.isclose(eig_rotation[0], 1))])
        if axis.shape[2] > 1:  # in case of 0 rotation angle
            axis = axis[:, :, 0]
        axis = np.ndarray.flatten(axis)

        angle = np.sum(abs(np.angle(eig_rotation[0]))) / 2

        # Find correct angle
        matrix_test = rotation_matrix_from_axis_and_angle(axis, angle)
        angle *= -1 if np.isclose(rotation_matrix[0, :] @ matrix_test[:, 0], 1) else 1

        # pylint: disable=not-callable
        return cls(
            name=name,
            center=center,
            size=size,
            axis_of_rotation=tuple(axis),
            angle_of_rotation=angle * u.rad,
        )

    @pd.model_validator(mode="after")
    def _convert_axis_and_angle_to_coordinate_axes(self) -> Self:
        """
        Converts the Box object's axis and angle orientation information to a
        coordinate axes representation.
        """
        # Ensure the axis is a numpy array
        if not self.private_attribute_input_cache.axes:
            axis = np.asarray(self.axis_of_rotation, dtype=np.float64)
            angle = self.angle_of_rotation.to("rad").v.item()

            # Normalize the axis vector
            axis = axis / np.linalg.norm(axis)

            rotation_matrix = rotation_matrix_from_axis_and_angle(axis, angle)

            # pylint: disable=assigning-non-slot
            self.private_attribute_input_cache.axes = np.transpose(rotation_matrix[:, :2]).tolist()

        return self

    @property
    def axes(self):
        """Return the axes that the box is aligned with."""
        return self.private_attribute_input_cache.axes

    @pd.field_validator("center", "size", mode="after")
    @classmethod
    def _update_input_cache(cls, value, info: pd.ValidationInfo):
        setattr(info.data["private_attribute_input_cache"], info.field_name, value)
        return value


@final
class Cylinder(_VolumeEntityBase):
    """
    :class:`Cylinder` class represents a cylinder in three-dimensional space.

    Example
    -------
    >>> fl.Cylinder(
    ...     name="bet_disk_volume",
    ...     center=(0, 0, 0) * fl.u.inch,
    ...     axis=(0, 0, 1),
    ...     outer_radius=150 * fl.u.inch,
    ...     height=15 * fl.u.inch,
    ... )

    ====
    """

    private_attribute_entity_type_name: Literal["Cylinder"] = pd.Field("Cylinder", frozen=True)
    axis: Axis = pd.Field(description="The axis of the cylinder.")
    # pylint: disable=no-member
    center: LengthType.Point = pd.Field(description="The center point of the cylinder.")
    height: LengthType.Positive = pd.Field(description="The height of the cylinder.")
    inner_radius: Optional[LengthType.NonNegative] = pd.Field(
        0 * u.m, description="The inner radius of the cylinder."
    )
    outer_radius: LengthType.Positive = pd.Field(description="The outer radius of the cylinder.")
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    @pd.model_validator(mode="after")
    def _check_inner_radius_is_less_than_outer_radius(self) -> Self:
        if self.inner_radius is not None and self.inner_radius >= self.outer_radius:
            raise ValueError(
                f"Cylinder inner radius ({self.inner_radius}) must be less than outer radius ({self.outer_radius})."
            )
        return self


class SurfacePrivateAttributes(Flow360BaseModel):
    """
     Private attributes for Surface.
    TODO: With the amount of private_attribute prefixes we have
    TODO: Maybe it makes more sense to lump them together to save storage space?
    """

    type_name: Literal["SurfacePrivateAttributes"] = pd.Field(
        "SurfacePrivateAttributes", frozen=True
    )
    bounding_box: BoundingBoxType = pd.Field(description="Bounding box of the surface.")


@final
class Surface(_SurfaceEntityBase):
    """
    :class:`Surface` represents a boundary surface in three-dimensional space.
    """

    private_attribute_entity_type_name: Literal["Surface"] = pd.Field("Surface", frozen=True)
    private_attribute_is_interface: Optional[bool] = pd.Field(
        None,
        frozen=True,
        description="This is required when generated from volume mesh "
        + "but not required when from surface mesh meta.",
    )
    private_attribute_tag_key: Optional[str] = pd.Field(
        None,
        description="The tag/attribute string used to group geometry faces to form this `Surface`.",
    )
    private_attribute_sub_components: Optional[List[str]] = pd.Field(
        [], description="The face ids in geometry that composed into this `Surface`."
    )
    private_attribute_color: Optional[str] = pd.Field(
        None, description="Color used for visualization"
    )
    private_attributes: Optional[SurfacePrivateAttributes] = pd.Field(None)

    # Note: private_attribute_id should not be `Optional` anymore.
    # B.C. Updater and geometry pipeline will populate it.

    def _overlaps(self, ghost_surface_center_y: Optional[float], length_tolerance: float) -> bool:
        if self.private_attributes is None:
            # Legacy cloud asset.
            return False
        # pylint: disable=no-member
        my_bounding_box = self.private_attributes.bounding_box
        if abs(my_bounding_box.ymax - ghost_surface_center_y) > length_tolerance:
            return False
        if abs(my_bounding_box.ymin - ghost_surface_center_y) > length_tolerance:
            return False
        return True

    def _will_be_deleted_by_mesher(
        # pylint: disable=too-many-arguments, too-many-return-statements
        self,
        at_least_one_body_transformed: bool,
        farfield_method: Optional[Literal["auto", "quasi-3d", "user-defined"]],
        global_bounding_box: Optional[BoundingBoxType],
        planar_face_tolerance: Optional[float],
        half_model_symmetry_plane_center_y: Optional[float],
        quasi_3d_symmetry_planes_center_y: Optional[tuple[float]],
    ) -> bool:
        """
        Check against the automated farfield method and
        determine if the current `Surface` will be deleted by the mesher.
        """
        if at_least_one_body_transformed:
            # If transformed then the following check will no longer be accurate
            # since we do not know the final bounding box for each surface and global model.
            return False

        if global_bounding_box is None or planar_face_tolerance is None or farfield_method is None:
            # VolumeMesh or Geometry/SurfaceMesh with legacy schema.
            return False

        if farfield_method == "user-defined":
            # Not applicable to user defined farfield
            return False

        length_tolerance = global_bounding_box.largest_dimension * planar_face_tolerance

        if farfield_method == "auto":
            if half_model_symmetry_plane_center_y is None:
                # Legacy schema.
                return False
            return self._overlaps(half_model_symmetry_plane_center_y, length_tolerance)

        if farfield_method == "quasi-3d":
            if quasi_3d_symmetry_planes_center_y is None:
                # Legacy schema.
                return False
            for plane_center_y in quasi_3d_symmetry_planes_center_y:
                if self._overlaps(plane_center_y, length_tolerance):
                    return True
            return False

        raise ValueError(f"Unknown auto farfield generation method: {farfield_method}.")


@final
class ImportedSurface(EntityBase):
    """ImportedSurface for post-processing"""

    private_attribute_registry_bucket_name: Literal["SurfaceEntityType"] = "SurfaceEntityType"
    private_attribute_entity_type_name: Literal["ImportedSurface"] = pd.Field(
        "ImportedSurface", frozen=True
    )
    file_name: str


class GhostSurface(_SurfaceEntityBase):
    """
    Represents a boundary surface that may or may not be generated therefore may or may not exist.
    It depends on the submitted geometry/Surface mesh. E.g. the symmetry plane in `AutomatedFarfield`.

    This is a token/place-holder used only on the Python API side.
    All `GhostSurface` entities will be replaced with exact entity instances before simulation.json submission.
    """

    private_attribute_entity_type_name: Literal["GhostSurface"] = pd.Field(
        "GhostSurface", frozen=True
    )


# pylint: disable=missing-class-docstring
@final
class GhostSphere(_SurfaceEntityBase):
    private_attribute_entity_type_name: Literal["GhostSphere"] = pd.Field(
        "GhostSphere", frozen=True
    )
    # Note: Making following optional since front end will not carry these over to assigned entities.
    center: Optional[List] = pd.Field(None, alias="center")
    max_radius: Optional[PositiveFloat] = pd.Field(None, alias="maxRadius")

    def exists(self, _) -> bool:
        """Ghost farfield always exists."""
        return True


# pylint: disable=missing-class-docstring
@final
class GhostCircularPlane(_SurfaceEntityBase):
    private_attribute_entity_type_name: Literal["GhostCircularPlane"] = pd.Field(
        "GhostCircularPlane", frozen=True
    )
    # Note: Making following optional since front end will not carry these over to assigned entities.
    center: Optional[List] = pd.Field(None, alias="center")
    max_radius: Optional[PositiveFloat] = pd.Field(None, alias="maxRadius")
    normal_axis: Optional[List] = pd.Field(None, alias="normalAxis")

    def _get_existence_dependency(self, validation_info):
        y_max = validation_info.global_bounding_box[1][1]
        y_min = validation_info.global_bounding_box[0][1]

        largest_dimension = -np.inf
        for dim in range(3):
            dimension = (
                validation_info.global_bounding_box[1][dim]
                - validation_info.global_bounding_box[0][dim]
            )
            largest_dimension = max(largest_dimension, dimension)

        tolerance = largest_dimension * validation_info.planar_face_tolerance
        return y_min, y_max, tolerance, largest_dimension

    def exists(self, validation_info) -> bool:
        """Mesher logic for symmetric plane existence."""

        if self.name != "symmetric":
            # Quasi-3D mode, no need to check existence.
            return True

        if validation_info is None:
            raise ValueError("Validation info is required for GhostCircularPlane existence check.")

        if validation_info.global_bounding_box is None:
            # This likely means the user try to use mesher on old cloud resources.
            # We cannot validate if symmetric exists so will let it pass. Pipeline will error out anyway.
            return True
        y_min, y_max, tolerance, _ = self._get_existence_dependency(validation_info)

        positive_half = abs(y_min) < tolerance < y_max
        negative_half = abs(y_max) < tolerance and y_min < -tolerance

        return positive_half or negative_half


class SurfacePair(Flow360BaseModel):
    """
    Represents a pair of surfaces.

    Attributes:
        pair (Tuple[Surface, Surface]): A tuple containing two Surface objects representing the pair.
    """

    pair: Tuple[Surface, Surface]

    @pd.field_validator("pair", mode="after")
    @classmethod
    def check_unique(cls, v):
        """Check if pairing with self."""
        if v[0].name == v[1].name:
            raise ValueError("A surface cannot be paired with itself.")
        return v

    @pd.model_validator(mode="before")
    @classmethod
    def _format_input(cls, input_data: Union[dict, list, tuple]):
        if isinstance(input_data, (list, tuple)):
            return {"pair": input_data}
        if isinstance(input_data, dict):
            return {"pair": input_data["pair"]}
        raise ValueError("Invalid input data.")

    def __hash__(self):
        return hash(tuple(sorted([self.pair[0].name, self.pair[1].name])))

    def __eq__(self, other):
        if isinstance(other, SurfacePair):
            return tuple(sorted([self.pair[0].name, self.pair[1].name])) == tuple(
                sorted([other.pair[0].name, other.pair[1].name])
            )
        return False

    def __str__(self):
        return ",".join(sorted([self.pair[0].name, self.pair[1].name]))


class SnappyBody(Flow360BaseModel):
    """
    Represents a group of faces forming a body for snappyHexMesh.
    Bodies and their regions are defined in the ASCII STL file by using the solid -> endsolid keywords with a body::region naming scheme.
    """

    body_name: str = pd.Field()


class MeshZone(Flow360BaseModel):
    """
    Represents a separate zone in the mesh, defined by a point inside it.
    """

    point_in_mesh: LengthType.Point
    name: str


VolumeEntityTypes = Union[GenericVolume, Cylinder, Box, str]


@final
class CustomVolume(_VolumeEntityBase):
    """
    CustomVolume is a volume zone defined by its surrounding surfaces.
    It will be generated by the volume mesher.
    """

    private_attribute_entity_type_name: Literal["CustomVolume"] = pd.Field(
        "CustomVolume", frozen=True
    )
    type: Literal["CustomVolume"] = pd.Field("CustomVolume", frozen=True)
    boundaries: EntityList[Surface] = pd.Field(
        description="The surfaces that define the boundaries of the custom volume."
    )
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    @pd.field_validator("boundaries", mode="after")
    @classmethod
    def ensure_unique_boundary_names(cls, v):
        """Check if the boundaries have different names within a CustomVolume."""
        if len(v.stored_entities) != len({boundary.name for boundary in v.stored_entities}):
            raise ValueError("The boundaries of a CustomVolume must have different names.")
        return v

    @pd.model_validator(mode="after")
    def ensure_usable(self):
        """Check if the beta mesher is enabled and that the user is using user defined farfield."""
        validation_info = get_validation_info()
        if validation_info is None:
            return self
        if validation_info.is_beta_mesher and validation_info.farfield_method == "user-defined":
            return self
        raise ValueError(
            "CustomVolume is only supported when beta mesher and user defined farfield are enabled."
        )
