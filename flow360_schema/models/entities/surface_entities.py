"""Surface entity definitions.

Surface, GhostSurface, GhostSphere, GhostCircularPlane, SurfacePair, Mirrored entities.
"""

from abc import ABCMeta, abstractmethod
from typing import Annotated, Any, ClassVar, Literal, final

import pydantic as pd
from pydantic import PositiveFloat
from typing_extensions import Self

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.bounding_box import BoundingBoxType
from flow360_schema.framework.entity.entity_base import EntityBase
from flow360_schema.framework.entity.entity_utils import generate_uuid
from flow360_schema.framework.validation.context import add_validation_warning

from .base import SurfacePrivateAttributes, _auto_symmetric_plane_exists_from_bbox, _SurfaceEntityBase


@final
class Surface(_SurfaceEntityBase):
    """Represents a boundary surface in three-dimensional space."""

    private_attribute_entity_type_name: Literal["Surface"] = pd.Field("Surface", frozen=True)
    private_attribute_is_interface: bool | None = pd.Field(
        None,
        frozen=True,
        description="This is required when generated from volume mesh "
        + "but not required when from surface mesh meta.",
    )
    private_attribute_tag_key: str | None = pd.Field(
        None,
        description="The tag/attribute string used to group geometry faces to form this `Surface`.",
    )
    private_attribute_sub_components: list[str] | None = pd.Field(
        [], description="The face ids in geometry that composed into this `Surface`."
    )
    private_attribute_color: str | None = pd.Field(None, description="Color used for visualization")
    private_attributes: SurfacePrivateAttributes | None = pd.Field(None)

    def _lies_on(self, ghost_surface_center_y: float | None, length_tolerance: float) -> bool:
        # Check if the surface lies entirely within tolerance of the center y
        if self.private_attributes is None:
            # Legacy cloud asset.
            return False
        my_bounding_box = self.private_attributes.bounding_box
        if abs(my_bounding_box.ymax - ghost_surface_center_y) > length_tolerance:  # type: ignore[operator]
            return False
        if abs(my_bounding_box.ymin - ghost_surface_center_y) > length_tolerance:  # type: ignore[operator]
            return False
        return True

    def _will_be_deleted_by_mesher(
        self,
        entity_transformation_detected: bool,
        farfield_method: Literal["auto", "quasi-3d", "quasi-3d-periodic", "user-defined", "wind-tunnel"] | None,
        global_bounding_box: BoundingBoxType | None,
        planar_face_tolerance: float | None,
        half_model_symmetry_plane_center_y: float | None,
        quasi_3d_symmetry_planes_center_y: tuple[float] | None,
        farfield_domain_type: str | None = None,
    ) -> bool:
        """
        Check against the automated farfield method and
        determine if the current `Surface` will be deleted by the mesher.
        """
        if entity_transformation_detected:
            # If transformed then the following check will no longer be accurate
            # since we do not know the final bounding box for each surface and global model.
            return False

        if global_bounding_box is None or planar_face_tolerance is None or farfield_method is None:
            # VolumeMesh or Geometry/SurfaceMesh with legacy schema.
            return False

        length_tolerance = global_bounding_box.largest_dimension * planar_face_tolerance

        if (
            farfield_domain_type in ("half_body_positive_y", "half_body_negative_y")
            and self.private_attributes is not None
        ):
            # Wrong half
            y_min = self.private_attributes.bounding_box.ymin
            y_max = self.private_attributes.bounding_box.ymax

            if farfield_domain_type == "half_body_positive_y" and y_max < -length_tolerance:
                return True

            if farfield_domain_type == "half_body_negative_y" and y_min > length_tolerance:
                return True

        if farfield_method in ("user-defined", "wind-tunnel"):
            # User-defined: user surfaces are not deleted
            # Wind-tunnel: not applicable
            return False

        if farfield_method == "auto":
            if half_model_symmetry_plane_center_y is None:
                # Legacy schema.
                return False
            if farfield_domain_type not in ("half_body_positive_y", "half_body_negative_y") and (
                not _auto_symmetric_plane_exists_from_bbox(
                    global_bounding_box=global_bounding_box,
                    planar_face_tolerance=planar_face_tolerance,
                )
            ):
                return False
            return self._lies_on(half_model_symmetry_plane_center_y, length_tolerance)

        if farfield_method in ("quasi-3d", "quasi-3d-periodic"):
            if quasi_3d_symmetry_planes_center_y is None:
                # Legacy schema.
                return False
            for plane_center_y in quasi_3d_symmetry_planes_center_y:
                if self._lies_on(plane_center_y, length_tolerance):
                    return True
            return False

        raise ValueError(f"Unknown auto farfield generation method: {farfield_method}.")


class ImportedSurface(EntityBase):
    """ImportedSurface for post-processing."""

    private_attribute_entity_type_name: Literal["ImportedSurface"] = pd.Field("ImportedSurface", frozen=True)

    private_attribute_sub_components: list[str] | None = pd.Field(None, description="A list of sub components")
    file_name: str | None = None
    surface_mesh_id: str | None = None

    @pd.model_validator(mode="after")
    def _populate_id_from_name(self) -> Self:
        """Ensure a deterministic private_attribute_id exists.

        CoordinateSystemManager and MirrorManager use private_attribute_id as
        a dict key for entity tracking.  A deterministic id derived from name
        guarantees the same ImportedSurface always resolves to the same id,
        even when reconstructed from cloud metadata across sessions.
        """
        if self.private_attribute_id is None:
            object.__setattr__(self, "private_attribute_id", f"{self.name}_defaultBody")
        return self


class GhostSurface(_SurfaceEntityBase):
    """
    Represents a boundary surface that may or may not be generated therefore may or may not exist.
    It depends on the submitted geometry/Surface mesh. E.g. the symmetry plane in `AutomatedFarfield`.

    This is a token/place-holder used only on the Python API side.
    All `GhostSurface` entities will be replaced with exact entity instances before simulation.json submission.
    """

    name: str = pd.Field(frozen=True)

    private_attribute_entity_type_name: Literal["GhostSurface"] = pd.Field("GhostSurface", frozen=True)


class WindTunnelGhostSurface(GhostSurface):
    """Wind tunnel boundary patches."""

    private_attribute_entity_type_name: Literal["WindTunnelGhostSurface"] = pd.Field(  # type: ignore[assignment]
        "WindTunnelGhostSurface", frozen=True
    )
    # For frontend: list of floor types that use this boundary patch, or ["all"]
    used_by: list[Literal["StaticFloor", "FullyMovingFloor", "CentralBelt", "WheelBelts", "all"]] = pd.Field(
        default_factory=lambda: ["all"],  # type: ignore[arg-type]
        frozen=True,
    )

    def exists(self, _: Any) -> bool:
        """Currently, .exists() is only called on automated farfield."""
        raise ValueError(".exists should not be called on wind tunnel farfield")


@final
class GhostSphere(_SurfaceEntityBase):
    """Ghost farfield sphere — always exists."""

    private_attribute_entity_type_name: Literal["GhostSphere"] = pd.Field("GhostSphere", frozen=True)
    # Note: Making following optional since front end will not carry these over to assigned entities.
    center: list[Any] | None = pd.Field(None, alias="center")
    max_radius: PositiveFloat | None = pd.Field(None, alias="maxRadius")

    def exists(self, _: Any) -> bool:
        """Ghost farfield always exists."""
        return True


def compute_bbox_tolerance(global_bounding_box: Any, planar_face_tolerance: float) -> tuple[float, float]:
    """Compute the largest bounding-box dimension and the derived planar-face tolerance."""
    largest_dimension = 0.0
    for dim in range(3):
        largest_dimension = max(largest_dimension, global_bounding_box[1][dim] - global_bounding_box[0][dim])
    return largest_dimension, largest_dimension * planar_face_tolerance


@final
class GhostCircularPlane(_SurfaceEntityBase):
    """Ghost circular plane — symmetric plane existence depends on bounding box geometry."""

    private_attribute_entity_type_name: Literal["GhostCircularPlane"] = pd.Field("GhostCircularPlane", frozen=True)
    # Note: Making following optional since front end will not carry these over to assigned entities.
    center: list[Any] | None = pd.Field(None, alias="center")
    max_radius: PositiveFloat | None = pd.Field(None, alias="maxRadius")
    normal_axis: list[Any] | None = pd.Field(None, alias="normalAxis")

    def _get_existence_dependency(self, validation_info: Any) -> tuple[float, float, float, float]:
        y_max = validation_info.global_bounding_box[1][1]
        y_min = validation_info.global_bounding_box[0][1]
        largest_dimension, tolerance = compute_bbox_tolerance(
            validation_info.global_bounding_box, validation_info.planar_face_tolerance
        )
        return y_min, y_max, tolerance, largest_dimension

    def exists(self, validation_info: Any) -> bool:
        """For automated farfield, check mesher logic for symmetric plane existence."""
        if self.name != "symmetric":
            # Quasi-3D mode or user-named symmetry patch (exists by definition)
            return True

        if validation_info is None:
            raise ValueError("Validation info is required for GhostCircularPlane existence check.")

        if validation_info.global_bounding_box is None:
            # This likely means the user tries to use mesher on old cloud resources.
            # We cannot validate if symmetric exists so will let it pass. Pipeline will error out anyway.
            return True

        if validation_info.will_generate_forced_symmetry_plane():
            return True

        return _auto_symmetric_plane_exists_from_bbox(
            global_bounding_box=validation_info.global_bounding_box,
            planar_face_tolerance=validation_info.planar_face_tolerance,
        )

    def _per_entity_type_validation(self, param_info: Any) -> Self:
        """Validate ghost surface existence and configuration."""
        from flow360_schema.models.simulation.validation.validation_utils import (
            check_symmetric_boundary_existence,
            check_user_defined_farfield_symmetry_existence,
        )

        # These functions expect a list, so wrap self
        check_user_defined_farfield_symmetry_existence([self], param_info)  # type: ignore[no-untyped-call]
        check_symmetric_boundary_existence([self], param_info)  # type: ignore[no-untyped-call]
        return self


class SurfacePairBase(Flow360BaseModel):
    """
    Base class for surface pair objects.
    Subclasses must define a `pair` attribute with the appropriate surface type.
    """

    pair: tuple[_SurfaceEntityBase, _SurfaceEntityBase]

    @pd.field_validator("pair", mode="after")
    @classmethod
    def check_unique(
        cls,
        v: tuple[_SurfaceEntityBase, _SurfaceEntityBase],
    ) -> tuple[_SurfaceEntityBase, _SurfaceEntityBase]:
        """Check if pairing with self."""
        if v[0].name == v[1].name:
            raise ValueError("A surface cannot be paired with itself.")
        return v

    @pd.model_validator(mode="before")
    @classmethod
    def _format_input(cls, input_data: dict[str, Any] | list[Any] | tuple[Any, ...]) -> dict[str, Any]:
        if isinstance(input_data, (list, tuple)):
            return {"pair": input_data}
        if isinstance(input_data, dict):
            return {"pair": input_data["pair"]}
        raise ValueError("Invalid input data.")

    def __hash__(self) -> int:
        return hash(tuple(sorted([self.pair[0].name, self.pair[1].name])))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return tuple(sorted([self.pair[0].name, self.pair[1].name])) == tuple(
                sorted([other.pair[0].name, other.pair[1].name])
            )
        return False

    def __str__(self) -> str:
        return ",".join(sorted([self.pair[0].name, self.pair[1].name]))


class SurfacePair(SurfacePairBase):
    """
    Represents a pair of surfaces.

    Attributes:
        pair (Tuple[Surface, Surface]): A tuple containing two Surface objects representing the pair.
    """

    pair: tuple[Surface, Surface]


class GhostSurfacePair(SurfacePairBase):
    """
    Represents a pair of ghost surfaces.

    Attributes:
        pair (Tuple[GhostSurfaceType, GhostSurfaceType]):
            A tuple containing two GhostSurfaceType objects representing the pair.
            GhostSurface is for Python API, GhostCircularPlane is for Web UI.
    """

    GhostSurfaceType: ClassVar[type] = Annotated[  # type: ignore[assignment]
        GhostSurface | GhostCircularPlane,
        pd.Field(discriminator="private_attribute_entity_type_name"),
    ]

    pair: tuple[GhostSurfaceType, GhostSurfaceType]  # type: ignore[valid-type]


class _MirroredEntityBase(EntityBase, metaclass=ABCMeta):
    """
    Base class for mirrored entities (MirroredSurface, MirroredGeometryBodyGroup).
    Provides common validation logic for checking source entity and mirror plane existence.
    """

    mirror_plane_id: str = pd.Field(description="ID of the mirror plane.")

    @property
    @abstractmethod
    def source_entity_id_field_name(self) -> str:
        """Return the name of the field containing the source entity ID."""

    @property
    @abstractmethod
    def source_entity_type_name(self) -> str:
        """Return the entity type name of the source entity."""

    def _manual_assignment_validation(self, param_info: Any) -> Self | None:  # type: ignore[override]
        """Validate that source entity and mirror plane still exist."""
        registry = param_info.get_entity_registry()
        if registry is None:
            return self

        # Get source entity ID using the field name from subclass
        source_entity_id = getattr(self, self.source_entity_id_field_name)

        # Check if source entity exists
        source_entity = registry.find_by_type_name_and_id(
            entity_type=self.source_entity_type_name, entity_id=source_entity_id
        )
        if source_entity is None:
            add_validation_warning(
                f"{self.__class__.__name__} '{self.name}' references non-existent source "
                f"{self.source_entity_type_name.lower()} (id={source_entity_id}). "
                "This entity will be removed."
            )
            return None

        # Check if mirror plane exists
        mirror_plane = registry.find_by_type_name_and_id(entity_type="MirrorPlane", entity_id=self.mirror_plane_id)
        if mirror_plane is None:
            add_validation_warning(
                f"{self.__class__.__name__} '{self.name}' references non-existent mirror plane "
                f"(id={self.mirror_plane_id}). This entity will be removed."
            )
            return None

        return self


class MirroredSurface(_SurfaceEntityBase, _MirroredEntityBase):
    """Represents a mirrored surface."""

    name: str = pd.Field()
    surface_id: str = pd.Field(description="ID of the original surface being mirrored.", frozen=True)
    mirror_plane_id: str = pd.Field(description="ID of the mirror plane to mirror the surface.")

    private_attribute_entity_type_name: Literal["MirroredSurface"] = pd.Field("MirroredSurface", frozen=True)
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    # Private attribute used for draft-only bookkeeping. This must NOT affect schema or serialization.
    _geometry_body_group_id: str | None = pd.PrivateAttr(default=None)

    @property
    def source_entity_id_field_name(self) -> str:
        """Return the name of the field containing the source entity ID."""
        return "surface_id"

    @property
    def source_entity_type_name(self) -> str:
        """Return the entity type name of the source entity."""
        return "Surface"


class MirroredGeometryBodyGroup(_MirroredEntityBase):
    """Represents a mirrored geometry body group."""

    name: str = pd.Field()
    geometry_body_group_id: str = pd.Field(description="ID of the geometry body group to mirror.")
    mirror_plane_id: str = pd.Field(description="ID of the mirror plane to mirror the geometry body group.")

    private_attribute_entity_type_name: Literal["MirroredGeometryBodyGroup"] = pd.Field(
        "MirroredGeometryBodyGroup", frozen=True
    )
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    @property
    def source_entity_id_field_name(self) -> str:
        """Return the name of the field containing the source entity ID."""
        return "geometry_body_group_id"

    @property
    def source_entity_type_name(self) -> str:
        """Return the entity type name of the source entity."""
        return "GeometryBodyGroup"
