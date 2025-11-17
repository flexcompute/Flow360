"""
Meshing settings that applies to volumes.
"""

from abc import ABCMeta
from typing import Literal, Optional, Union

import pydantic as pd
from typing_extensions import deprecated

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.outputs.output_entities import Slice
from flow360.component.simulation.primitives import (
    AxisymmetricBody,
    Box,
    CustomVolume,
    Cylinder,
    GenericVolume,
    GhostSurface,
    Surface,
)
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.validation.validation_context import (
    get_validation_info,
)
from flow360.component.simulation.validation.validation_utils import (
    check_deleted_surface_in_entity_list,
)


class UniformRefinement(Flow360BaseModel):
    """
    Uniform spacing refinement inside specified region of mesh.

    Example
    -------

      >>> fl.UniformRefinement(
      ...     entities=[cylinder, box],
      ...     spacing=1*fl.u.cm
      ... )

    ====
    """

    name: Optional[str] = pd.Field("Uniform refinement")
    refinement_type: Literal["UniformRefinement"] = pd.Field("UniformRefinement", frozen=True)
    entities: EntityList[Box, Cylinder] = pd.Field(
        description=":class:`UniformRefinement` can be applied to :class:`~flow360.Box` "
        + "and :class:`~flow360.Cylinder` regions."
    )
    # pylint: disable=no-member
    spacing: LengthType.Positive = pd.Field(description="The required refinement spacing.")


class StructuredBoxRefinement(Flow360BaseModel):
    """
    - The mesh inside the :class:`StructuredBoxRefinement` is semi-structured.
    - The :class:`StructuredBoxRefinement` cannot enclose/intersect with other objects.
    - The spacings along the three box axes can be adjusted independently.

    Example
    -------

    >>> StructuredBoxRefinement(
    ...     entities=[
    ...        Box.from_principal_axes(
    ...           name="boxRefinement",
    ...           center=(0, 1, 1) * fl.u.cm,
    ...           size=(1, 2, 1) * fl.u.cm,
    ...           axes=((2, 2, 0), (-2, 2, 0)),
    ...       )
    ...     ],
    ...     spacing_axis1=7.5*u.cm,
    ...     spacing_axis2=10*u.cm,
    ...     spacing_normal=15*u.cm,
    ...   )
    ====
    """

    # pylint: disable=no-member
    # pylint: disable=too-few-public-methods
    name: Optional[str] = pd.Field("StructuredBoxRefinement")
    refinement_type: Literal["StructuredBoxRefinement"] = pd.Field(
        "StructuredBoxRefinement", frozen=True
    )
    entities: EntityList[Box] = pd.Field()

    spacing_axis1: LengthType.Positive = pd.Field(
        description="Spacing along the first axial direction."
    )
    spacing_axis2: LengthType.Positive = pd.Field(
        description="Spacing along the second axial direction."
    )
    spacing_normal: LengthType.Positive = pd.Field(
        description="Spacing along the normal axial direction."
    )

    @pd.model_validator(mode="after")
    def _validate_only_in_beta_mesher(self):
        """
        Ensure that StructuredBoxRefinement objects are only processed with the beta mesher.
        """
        validation_info = get_validation_info()
        if validation_info is None:
            return self
        if validation_info.is_beta_mesher:
            return self

        raise ValueError("`StructuredBoxRefinement` is only supported with the beta mesher.")


class AxisymmetricRefinementBase(Flow360BaseModel, metaclass=ABCMeta):
    """Base class for all refinements that requires spacing in axial, radial and circumferential directions."""

    # pylint: disable=no-member
    spacing_axial: LengthType.Positive = pd.Field(description="Spacing along the axial direction.")
    spacing_radial: LengthType.Positive = pd.Field(
        description="Spacing along the radial direction."
    )
    spacing_circumferential: LengthType.Positive = pd.Field(
        description="Spacing along the circumferential direction."
    )


class AxisymmetricRefinement(AxisymmetricRefinementBase):
    """
    - The mesh inside the :class:`AxisymmetricRefinement` is semi-structured.
    - The :class:`AxisymmetricRefinement` cannot enclose/intersect with other objects.
    - Users could create a donut-shape :class:`AxisymmetricRefinement` and place their hub/centerbody in the middle.
    - :class:`AxisymmetricRefinement` can be used for resolving the strong flow gradient
       along the axial direction for the actuator or BET disks.
    - The spacings along the axial, radial and circumferential directions can be adjusted independently.

    Example
    -------

      >>> fl.AxisymmetricRefinement(
      ...     entities=[cylinder],
      ...     spacing_axial=1e-4,
      ...     spacing_radial=0.3*fl.u.cm,
      ...     spacing_circumferential=5*fl.u.mm
      ... )

    ====
    """

    name: Optional[str] = pd.Field("Axisymmetric refinement")
    refinement_type: Literal["AxisymmetricRefinement"] = pd.Field(
        "AxisymmetricRefinement", frozen=True
    )
    entities: EntityList[Cylinder] = pd.Field()


class RotationVolume(AxisymmetricRefinementBase):
    """
    Creates a rotation volume mesh using cylindrical or axisymmetric body entities.

    - The mesh on :class:`RotationVolume` is guaranteed to be concentric.
    - The :class:`RotationVolume` is designed to enclose other objects, but it can't intersect with other objects.
    - Users can create a donut-shaped :class:`RotationVolume` and put their stationary centerbody in the middle.
    - This type of volume zone can be used to generate volume zones compatible with :class:`~flow360.Rotation` model.
    - Supports both :class:`Cylinder` and :class:`AxisymmetricBody` entities for defining the rotation volume geometry.

    .. note::
        The deprecated :class:`RotationCylinder` class is maintained for backward compatibility
        but only accepts :class:`Cylinder` entities. New code should use :class:`RotationVolume`.

    Example
    -------
    Using a Cylinder entity:

      >>> fl.RotationVolume(
      ...     name="RotationCylinder",
      ...     spacing_axial=0.5*fl.u.m,
      ...     spacing_circumferential=0.3*fl.u.m,
      ...     spacing_radial=1.5*fl.u.m,
      ...     entities=cylinder
      ... )

    Using an AxisymmetricBody entity:

      >>> fl.RotationVolume(
      ...     name="RotationConeFrustum",
      ...     spacing_axial=0.5*fl.u.m,
      ...     spacing_circumferential=0.3*fl.u.m,
      ...     spacing_radial=1.5*fl.u.m,
      ...     entities=axisymmetric_body
      ... )

    With enclosed entities:

      >>> fl.RotationVolume(
      ...     name="RotationVolume",
      ...     spacing_axial=0.5*fl.u.m,
      ...     spacing_circumferential=0.3*fl.u.m,
      ...     spacing_radial=1.5*fl.u.m,
      ...     entities=outer_cylinder,
      ...     enclosed_entities=[inner_cylinder, surface]
      ... )
    """

    # Note: Please refer to
    # Note: https://www.notion.so/flexcompute/Python-model-design-document-
    # Note: 78d442233fa944e6af8eed4de9541bb1?pvs=4#c2de0b822b844a12aa2c00349d1f68a3

    type: Literal["RotationVolume"] = pd.Field("RotationVolume", frozen=True)
    name: Optional[str] = pd.Field("Rotation Volume", description="Name to display in the GUI.")
    entities: EntityList[Cylinder, AxisymmetricBody] = pd.Field()
    enclosed_entities: Optional[EntityList[Cylinder, Surface, AxisymmetricBody, Box]] = pd.Field(
        None,
        description="Entities enclosed by :class:`RotationVolume`. "
        "Can be `Surface` and/or other :class:`~flow360.Cylinder`(s)"
        "and/or other :class:`~flow360.AxisymmetricBody`(s)"
        "and/or other :class:`~flow360.Box`(s)",
    )

    @pd.field_validator("entities", mode="after")
    @classmethod
    def _validate_single_instance_in_entity_list(cls, values):
        """
        [CAPABILITY-LIMITATION]
        Multiple instances in the entities is not allowed.
        Because enclosed_entities will almost certain be different.
        `enclosed_entities` is planned to be auto_populated in the future.
        """
        # pylint: disable=protected-access
        if len(values._get_expanded_entities(create_hard_copy=False)) > 1:
            raise ValueError(
                "Only single instance is allowed in entities for each `RotationVolume`."
            )
        return values

    @pd.field_validator("entities", mode="after")
    @classmethod
    def _validate_cylinder_name_length(cls, values):
        """
        Check the name length for the cylinder entities due to the 32-character
        limitation of all data structure names and labels in CGNS format.
        The current prefix is 'rotatingBlock-' with 14 characters.
        """
        validation_info = get_validation_info()
        if validation_info is None:
            return values
        if validation_info.is_beta_mesher:
            return values

        cgns_max_zone_name_length = 32
        max_cylinder_name_length = cgns_max_zone_name_length - len("rotatingBlock-")
        for entity in values.stored_entities:
            if isinstance(entity, Cylinder) and len(entity.name) > max_cylinder_name_length:
                raise ValueError(
                    f"The name ({entity.name}) of `Cylinder` entity in `RotationVolume` "
                    + f"exceeds {max_cylinder_name_length} characters limit."
                )
        return values

    @pd.field_validator("enclosed_entities", mode="after")
    @classmethod
    def _validate_enclosed_box_only_in_beta_mesher(cls, values):
        """
        Check the name length for the cylinder entities due to the 32-character
        limitation of all data structure names and labels in CGNS format.
        The current prefix is 'rotatingBlock-' with 14 characters.
        """
        validation_info = get_validation_info()
        if validation_info is None or values is None:
            return values
        if validation_info.is_beta_mesher:
            return values

        for entity in values.stored_entities:
            if isinstance(entity, Box):
                raise ValueError(
                    "`Box` entity in `RotationVolume.enclosed_entities` is only supported with the beta mesher."
                )

        return values

    @pd.field_validator("entities", mode="after")
    @classmethod
    def _validate_axisymmetric_only_in_beta_mesher(cls, values):
        """
        Ensure that axisymmetric RotationVolumes are only processed with the beta mesher.
        """
        validation_info = get_validation_info()
        if validation_info is None:
            return values
        if validation_info.is_beta_mesher:
            return values

        for entity in values.stored_entities:
            if isinstance(entity, AxisymmetricBody):
                raise ValueError(
                    "`AxisymmetricBody` entity for `RotationVolume` is only supported with the beta mesher."
                )
        return values

    @pd.field_validator("enclosed_entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value):
        """Ensure all boundaries will be present after mesher"""
        if value is None:
            return value
        return check_deleted_surface_in_entity_list(value)


@deprecated(
    "The `RotationCylinder` class is deprecated! Use `RotationVolume`,"
    "which supports both `Cylinder` and `AxisymmetricBody` entities instead."
)
class RotationCylinder(RotationVolume):
    """
    .. deprecated::
        Use :class:`RotationVolume` instead. This class is maintained for backward
        compatibility but will be removed in a future version.

    RotationCylinder creates a rotation volume mesh using cylindrical entities.

    - The mesh on :class:`RotationCylinder` is guaranteed to be concentric.
    - The :class:`RotationCylinder` is designed to enclose other objects, but it can't intersect with other objects.
    - Users could create a donut-shape :class:`RotationCylinder` and put their stationary centerbody in the middle.
    - This type of volume zone can be used to generate volume zone compatible with :class:`~flow360.Rotation` model.

    .. note::
        :class:`RotationVolume` now supports both :class:`Cylinder` and :class:`AxisymmetricBody` entities.
        Please migrate to using :class:`RotationVolume` directly.

    Example
    -------
      >>> fl.RotationCylinder(
      ...     name="RotationCylinder",
      ...     spacing_axial=0.5*fl.u.m,
      ...     spacing_circumferential=0.3*fl.u.m,
      ...     spacing_radial=1.5*fl.u.m,
      ...     entities=cylinder
      ... )
    """

    type: Literal["RotationCylinder"] = pd.Field("RotationCylinder", frozen=True)
    entities: EntityList[Cylinder] = pd.Field()


class _FarfieldBase(Flow360BaseModel):
    """Base class for farfield parameters."""

    domain_type: Optional[Literal["half_body_positive_y", "half_body_negative_y", "full_body"]] = (
        pd.Field(  # In the future, we will support more flexible half model types and full model via Union.
            None,
            description="""
            - half_body_positive_y: Trim to a half-model by slicing with the global Y=0 plane; keep the '+y' side for meshing and simulation.
            - half_body_negative_y: Trim to a half-model by slicing with the global Y=0 plane; keep the '-y' side for meshing and simulation.
            - full_body: Keep the full body for meshing and simulation without attempting to add symmetry planes.

            Warning: When using AutomatedFarfield, setting `domain_type` overrides the 'auto' symmetry plane behavior.
            """,
        )
    )

    @pd.field_validator("domain_type", mode="after")
    @classmethod
    def _validate_only_in_beta_mesher(cls, value):
        """
        Ensure that domain_type is only used with the beta mesher and GAI.
        """
        validation_info = get_validation_info()
        if validation_info is None:
            return value
        if not value or (
            validation_info.use_geometry_AI is True and validation_info.is_beta_mesher is True
        ):
            return value
        raise ValueError(
            "`domain_type` is only supported when using both GAI surface mesher and beta volume mesher."
        )


class AutomatedFarfield(_FarfieldBase):
    """
    Settings for automatic farfield volume zone generation.

    Example
    -------

      >>> fl.AutomatedFarfield(name="Farfield", method="auto")

    ====
    """

    type: Literal["AutomatedFarfield"] = pd.Field("AutomatedFarfield", frozen=True)
    name: Optional[str] = pd.Field("Automated Farfield")  # Kept optional for backward compatibility
    method: Literal["auto", "quasi-3d", "quasi-3d-periodic"] = pd.Field(
        default="auto",
        frozen=True,
        description="""
        - auto: The mesher will Sphere or semi-sphere will be generated based on the bounding box of the geometry.
            - Full sphere if min{Y} < 0 and max{Y} > 0.
            - +Y semi sphere if min{Y} = 0 and max{Y} > 0.
            - -Y semi sphere if min{Y} < 0 and max{Y} = 0.
        - quasi-3d: Thin disk will be generated for quasi 3D cases.
                    Both sides of the farfield disk will be treated as "symmetric plane"
        - quasi-3d-periodic: The two sides of the quasi-3d disk will be conformal
        Note: For quasi-3d, please do not group patches from both sides of the farfield disk into a single surface.
        """,
    )
    private_attribute_entity: GenericVolume = pd.Field(
        GenericVolume(name="__farfield_zone_name_not_properly_set_yet"),
        frozen=True,
        exclude=True,
    )
    relative_size: pd.PositiveFloat = pd.Field(
        default=50.0,
        description="Radius of the far-field (semi)sphere/cylinder relative to "
        "the max dimension of the geometry bounding box.",
    )

    @property
    def farfield(self):
        """Returns the farfield boundary surface."""
        # Make sure the naming is the same here and what the geometry/surface mesh pipeline generates.
        return GhostSurface(name="farfield")

    @property
    def symmetry_plane(self) -> GhostSurface:
        """
        Returns the symmetry plane boundary surface.
        """
        if self.method == "auto":
            return GhostSurface(name="symmetric")
        raise ValueError(
            "Unavailable for quasi-3d farfield methods. Please use `symmetry_planes` property instead."
        )

    @property
    def symmetry_planes(self):
        """Returns the symmetry plane boundary surface(s)."""
        # Make sure the naming is the same here and what the geometry/surface mesh pipeline generates.
        if self.method == "auto":
            return GhostSurface(name="symmetric")
        if self.method in ("quasi-3d", "quasi-3d-periodic"):
            return [
                GhostSurface(name="symmetric-1"),
                GhostSurface(name="symmetric-2"),
            ]
        raise ValueError(f"Unsupported method: {self.method}")

    @pd.field_validator("method", mode="after")
    @classmethod
    def _validate_quasi_3d_periodic_only_in_legacy_mesher(cls, values):
        """
        Check mesher and AutomatedFarfield method compatibility
        """
        validation_info = get_validation_info()
        if validation_info is None:
            return values
        if validation_info.is_beta_mesher and values == "quasi-3d-periodic":
            raise ValueError("Only legacy mesher can support quasi-3d-periodic")
        return values


class UserDefinedFarfield(_FarfieldBase):
    """
    Setting for user defined farfield zone generation.
    This means the "farfield" boundaries are coming from the supplied geometry file
    and meshing will take place inside this "geometry".

    Example
    -------

      >>> fl.UserDefinedFarfield(name="InnerChannel")

    ====
    """

    type: Literal["UserDefinedFarfield"] = pd.Field("UserDefinedFarfield", frozen=True)
    name: Optional[str] = pd.Field(None)

    @property
    def symmetry_plane(self) -> GhostSurface:
        """
        Returns the symmetry plane boundary surface.

        Warning: This should only be used when using GAI and beta mesher.
        """
        if self.domain_type not in ("half_body_positive_y", "half_body_negative_y"):
            raise ValueError(
                "Symmetry plane of user defined farfield is only supported when domain_type "
                "is `half_body_positive_y` or `half_body_negative_y`."
            )
        return GhostSurface(name="symmetric")


# pylint: disable=no-member
class StaticFloor(Flow360BaseModel):
    """Class for static wind tunnel floor with friction patch."""

    type_name: Literal["StaticFloor"] = pd.Field(
        "StaticFloor", description="Static floor with friction patch.", frozen=True
    )
    friction_patch_x_min: LengthType = pd.Field(
        default=-3 * u.m, description="Minimum x of friction patch."
    )
    friction_patch_x_max: LengthType = pd.Field(
        default=6 * u.m, description="Maximum x of friction patch."
    )
    friction_patch_width: LengthType.Positive = pd.Field(
        default=2 * u.m, description="Width of friction patch."
    )

    @pd.model_validator(mode="after")
    def _validate_friction_patch(self):
        if self.friction_patch_x_min >= self.friction_patch_x_max:
            raise ValueError(
                f"Friction patch minimum x ({self.friction_patch_x_min}) "
                f"must be less than maximum x ({self.friction_patch_x_max})."
            )
        return self


class FullyMovingFloor(Flow360BaseModel):
    """Class for fully moving wind tunnel floor with friction patch."""

    type_name: Literal["FullyMovingFloor"] = pd.Field(
        "FullyMovingFloor", description="Fully moving floor.", frozen=True
    )


# pylint: disable=no-member
class CentralBelt(Flow360BaseModel):
    """Class for wind tunnel floor with one central belt."""

    type_name: Literal["CentralBelt"] = pd.Field(
        "CentralBelt", description="Floor with central belt.", frozen=True
    )
    central_belt_x_min: LengthType = pd.Field(
        default=-2 * u.m, description="Minimum x of central belt."
    )
    central_belt_x_max: LengthType = pd.Field(
        default=2 * u.m, description="Maximum x of central belt."
    )
    central_belt_width: LengthType.Positive = pd.Field(
        default=1.2 * u.m, description="Width of central belt."
    )

    @pd.model_validator(mode="after")
    def _validate_central_belt(self):
        if self.central_belt_x_min >= self.central_belt_x_max:
            raise ValueError(
                f"Central belt minimum x ({self.central_belt_x_min}) "
                f"must be less than maximum x ({self.central_belt_x_max})."
            )
        return self


# pylint: disable=no-member
class WheelBelts(CentralBelt):
    """Class for wind tunnel floor with one central belt and four wheel belts."""

    type_name: Literal["WheelBelts"] = pd.Field(
        "WheelBelts", description="Floor with central belt and four wheel belts.", frozen=True
    )
    # No defaults for the below; user must specify
    front_wheel_belt_x_min: LengthType = pd.Field(description="Minimum x of front wheel belt.")
    front_wheel_belt_x_max: LengthType = pd.Field(description="Maximum x of front wheel belt.")
    front_wheel_belt_y_inner: LengthType.Positive = pd.Field(
        description="Inner y of front wheel belt."
    )
    front_wheel_belt_y_outer: LengthType.Positive = pd.Field(
        description="Outer y of front wheel belt."
    )
    rear_wheel_belt_x_min: LengthType = pd.Field(description="Minimum x of rear wheel belt.")
    rear_wheel_belt_x_max: LengthType = pd.Field(description="Maximum x of rear wheel belt.")
    rear_wheel_belt_y_inner: LengthType.Positive = pd.Field(
        description="Inner y of rear wheel belt."
    )
    rear_wheel_belt_y_outer: LengthType.Positive = pd.Field(
        description="Outer y of rear wheel belt."
    )

    @pd.model_validator(mode="after")
    def _validate_wheel_belt_params(self):
        if self.front_wheel_belt_x_min >= self.front_wheel_belt_x_max:
            raise ValueError(
                f"Front wheel belt minimum x ({self.front_wheel_belt_x_min}) "
                f"must be less than maximum x ({self.front_wheel_belt_x_max})."
            )
        if self.front_wheel_belt_x_max >= self.rear_wheel_belt_x_min:
            raise ValueError(
                f"Front wheel belt maximum x ({self.front_wheel_belt_x_max}) "
                f"must be less than rear wheel belt minimum x ({self.rear_wheel_belt_x_min})."
            )
        if self.rear_wheel_belt_x_min >= self.rear_wheel_belt_x_max:
            raise ValueError(
                f"Rear wheel belt minimum x ({self.rear_wheel_belt_x_min}) "
                f"must be less than maximum x ({self.rear_wheel_belt_x_max})."
            )
        if self.front_wheel_belt_y_inner >= self.front_wheel_belt_y_outer:
            raise ValueError(
                f"Front wheel belt inner y ({self.front_wheel_belt_y_inner}) "
                f"must be less than outer y ({self.front_wheel_belt_y_outer})."
            )
        if self.rear_wheel_belt_y_inner >= self.rear_wheel_belt_y_outer:
            raise ValueError(
                f"Rear wheel belt inner y ({self.rear_wheel_belt_y_inner}) "
                f"must be less than outer y ({self.rear_wheel_belt_y_outer})."
            )
        return self


# pylint: disable=no-member
class WindTunnelFarfield(_FarfieldBase):
    """
    Settings for analytic wind tunnel farfield generation.
    The user only needs to provide tunnel dimensions and floor type and dimensions, rather than a geometry.

    Example
    -------
        >>> fl.WindTunnelFarfield(
            width = 10 * fl.u.m,
            height = 10 * fl.u.m,
            inlet_x_position = -5 * fl.u.m,
            outlet_x_position = 15 * fl.u.m,
            floor_z_position = 0 * fl.u.m,
            floor_type = fl.CentralBelt(
                central_belt_x_min = -1 * fl.u.m,
                central_belt_x_max = 6 * fl.u.m,
                central_belt_width = 1.2 * fl.u.m
            )
        )
    """

    type: Literal["WindTunnelFarfield"] = pd.Field("WindTunnelFarfield", frozen=True)
    name: str = pd.Field("Wind Tunnel Farfield", description="Name of the wind tunnel farfield.")

    # Tunnel parameters
    width: LengthType.Positive = pd.Field(default=10 * u.m, description="Width of the wind tunnel.")
    height: LengthType.Positive = pd.Field(
        default=6 * u.m, description="Height of the wind tunnel."
    )
    inlet_x_position: LengthType = pd.Field(
        default=-20 * u.m, description="X-position of the inlet."
    )
    outlet_x_position: LengthType = pd.Field(
        default=40 * u.m, description="X-position of the outlet."
    )
    floor_z_position: LengthType = pd.Field(default=0 * u.m, description="Z-position of the floor.")

    floor_type: Union[
        StaticFloor,
        FullyMovingFloor,
        CentralBelt,
        WheelBelts,
    ] = pd.Field(default=StaticFloor, description="Floor type of the wind tunnel.")

    # up direction not yet supported; assume +Z

    def inlet(self) -> GhostSurface:
        """Returns the inlet boundary surface."""
        return GhostSurface(name="windTunnelInlet")

    def outlet(self) -> GhostSurface:
        """Returns the outlet boundary surface."""
        return GhostSurface(name="windTunnelOutlet")

    def symmetry_plane(self) -> GhostSurface:
        """
        Returns the symmetry plane boundary surface for half body domains.
        """
        if self.domain_type not in ("half_body_positive_y", "half_body_negative_y"):
            raise ValueError(
                "Symmetry plane for wind tunnel farfield is only supported when domain_type "
                "is `half_body_positive_y` or `half_body_negative_y`."
            )
        return GhostSurface(name="symmetric")

    def floor(self) -> GhostSurface:
        """Returns the floor boundary surface, excluding friction, central, and wheel belts if applicable."""
        return GhostSurface(name="windTunnelFloor")

    def ceiling(self) -> GhostSurface:
        """Returns the ceiling boundary surface."""
        return GhostSurface(name="windTunnelCeiling")

    def friction_patch(self) -> GhostSurface:
        """Returns the friction patch for StaticFloor floor type."""
        if self.floor_type is not StaticFloor:
            raise ValueError(
                "Friction patch for wind tunnel farfield "
                "is only supported if floor type is `StaticFloor`."
            )
        return GhostSurface(name="windTunnelFrictionPatch")

    def central_belt(self) -> GhostSurface:
        """Returns the central belt for CentralBelt or WheelBelts floor types."""
        if self.floor_type not in (CentralBelt, WheelBelts):
            raise ValueError(
                "Central belt for wind tunnel farfield "
                "is only supported if floor type is `CentralBelt` or `WheelBelts`."
            )
        return GhostSurface(name="windTunnelCentralBelt")

    def front_wheel_belts(self) -> GhostSurface:
        """Returns the front wheel belts for WheelBelts floor type."""
        if self.floor_type is not WheelBelts:
            raise ValueError(
                "Front wheel belts for wind tunnel farfield "
                "is only supported if floor type is `WheelBelts`."
            )
        return GhostSurface(name="windTunnelFrontWheelBelt")

    def rear_wheel_belts(self) -> GhostSurface:
        """Returns the rear wheel belts for WheelBelts floor type."""
        if self.floor_type is not WheelBelts:
            raise ValueError(
                "Rear wheel belts for wind tunnel farfield "
                "is only supported if floor type is `WheelBelts`."
            )
        return GhostSurface(name="windTunnelRearWheelBelt")

    @pd.model_validator(mode="after")
    def _validate_inlet_is_less_than_outlet(self):
        if self.inlet_x_position >= self.outlet_x_position:
            raise ValueError(
                f"Inlet x position ({self.inlet_x_position}) "
                f"must be less than outlet x position ({self.outlet_x_position})."
            )
        return self


class MeshSliceOutput(Flow360BaseModel):
    """
    :class:`MeshSliceOutput` class for mesh slice output settings.

    Example
    -------

    >>> fl.MeshSliceOutput(
    ...     slices=[
    ...         fl.Slice(
    ...             name="Slice_1",
    ...             normal=(0, 1, 0),
    ...             origin=(0, 0.56, 0)*fl.u.m
    ...         ),
    ...     ],
    ... )

    ====
    """

    name: str = pd.Field("Mesh slice output", description="Name of the `MeshSliceOutput`.")
    entities: EntityList[Slice] = pd.Field(
        alias="slices",
        description="List of output :class:`~flow360.Slice` entities.",
    )
    output_type: Literal["MeshSliceOutput"] = pd.Field("MeshSliceOutput", frozen=True)


class CustomZones(Flow360BaseModel):
    """
    :class:`CustomZones` class for creating volume zones from custom volumes.
    Names of the generated volume zones will be the names of the custom volumes.

    Example
    -------

      >>> fl.CustomZones(name="Custom zones", entities=[custom_volume1, custom_volume2], )

    ====
    """

    type: Literal["CustomZones"] = pd.Field("CustomZones", frozen=True)
    name: str = pd.Field("Custom zones", description="Name of the `CustomZones` meshing setting.")
    entities: EntityList[CustomVolume] = pd.Field(
        description="The custom volume zones to be generated."
    )
    element_type: Literal["mixed", "tetrahedra"] = pd.Field(
        default="mixed",
        description="The element type to be used for the generated volume zones."
        + " - mixed: Mesher will automatically choose the element types used."
        + " - tetrahedra: Only tetrahedra element type will be used for the generated volume zones.",
    )
