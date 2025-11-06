"""
Meshing settings that applies to volumes.
"""

from abc import ABCMeta
from typing import Literal, Optional

import pydantic as pd
from typing_extensions import deprecated

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
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
    project_to_surface: Optional[bool] = pd.Field(True)


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

    # pylint: disable=no-member
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
