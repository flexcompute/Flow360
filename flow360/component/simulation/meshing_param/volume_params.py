"""
Meshing settings that applies to volumes.
"""

# pylint: disable=too-many-lines

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
    MirroredSurface,
    SeedpointVolume,
    Sphere,
    Surface,
    WindTunnelGhostSurface,
)
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.validation.validation_context import (
    ParamsValidationInfo,
    add_validation_warning,
    contextual_field_validator,
    contextual_model_validator,
    get_validation_info,
)
from flow360.component.simulation.validation.validation_utils import (
    validate_entity_list_surface_existence,
)
from flow360.exceptions import Flow360ValueError


class classproperty:  # pylint: disable=invalid-name,too-few-public-methods
    """Descriptor to create class-level properties that can be accessed from the class itself."""

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, owner):
        return self.func(owner)


class UniformRefinement(Flow360BaseModel):
    """
    Uniform spacing refinement inside specified region of mesh.

    Example
    -------

      >>> fl.UniformRefinement(
      ...     entities=[cylinder, box, axisymmetric_body, sphere],
      ...     spacing=1*fl.u.cm
      ... )

    ====
    """

    name: Optional[str] = pd.Field("Uniform refinement")
    refinement_type: Literal["UniformRefinement"] = pd.Field("UniformRefinement", frozen=True)
    entities: EntityList[Box, Cylinder, AxisymmetricBody, Sphere] = pd.Field(
        description=":class:`UniformRefinement` can be applied to :class:`~flow360.Box`, "
        + ":class:`~flow360.Cylinder`, :class:`~flow360.AxisymmetricBody`, "
        + "and :class:`~flow360.Sphere` regions."
    )
    # pylint: disable=no-member
    spacing: LengthType.Positive = pd.Field(description="The required refinement spacing.")
    project_to_surface: Optional[bool] = pd.Field(
        None,
        description="Whether to include the refinement in the surface mesh. Defaults to True when using snappy.",
    )

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def check_entities_used_with_beta_mesher(cls, values, param_info: ParamsValidationInfo):
        """Check that AxisymmetricBody and Sphere are used with beta mesher."""

        if values is None:
            return values
        if param_info.is_beta_mesher:
            return values

        expanded = param_info.expand_entity_list(values)
        for entity in expanded:
            if isinstance(entity, AxisymmetricBody):
                raise ValueError(
                    "`AxisymmetricBody` entity for `UniformRefinement` is supported only with beta mesher."
                )
            if isinstance(entity, Sphere):
                raise ValueError(
                    "`Sphere` entity for `UniformRefinement` is supported only with beta mesher."
                )

        return values

    @contextual_model_validator(mode="after")
    def check_project_to_surface_with_snappy(self, param_info: ParamsValidationInfo):
        """Check that project_to_surface is used only with snappy."""
        if not param_info.use_snappy and self.project_to_surface is not None:
            raise ValueError("project_to_surface is supported only for snappyHexMesh.")

        return self


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

    @contextual_model_validator(mode="after")
    def _validate_only_in_beta_mesher(self, param_info: ParamsValidationInfo):
        """
        Ensure that StructuredBoxRefinement objects are only processed with the beta mesher.
        """
        if param_info.is_beta_mesher:
            return self

        raise ValueError("`StructuredBoxRefinement` is only supported with the beta mesher.")


class AxisymmetricRefinement(Flow360BaseModel):
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
    # pylint: disable=no-member
    spacing_axial: LengthType.Positive = pd.Field(description="Spacing along the axial direction.")
    spacing_radial: LengthType.Positive = pd.Field(
        description="Spacing along the radial direction."
    )
    spacing_circumferential: LengthType.Positive = pd.Field(
        description="Spacing along the circumferential direction."
    )


class RotationVolume(Flow360BaseModel):
    """
    Creates a rotation volume mesh using cylindrical, axisymmetric body, or sphere entities.

    - The mesh on :class:`RotationVolume` is guaranteed to be concentric.
    - The :class:`RotationVolume` is designed to enclose other objects, but it can't intersect with other objects.
    - Users can create a donut-shaped :class:`RotationVolume` and put their stationary centerbody in the middle.
    - This type of volume zone can be used to generate volume zones compatible with :class:`~flow360.Rotation` model.
    - Supports :class:`Cylinder`, :class:`AxisymmetricBody`, and :class:`Sphere` entities
      for defining the rotation volume geometry.

    .. note::
        The deprecated :class:`RotationCylinder` class is maintained for backward compatibility
        but only accepts :class:`Cylinder` entities. New code should use :class:`RotationVolume`.

    .. note::
        For :class:`Sphere` entities, only `spacing_circumferential` is required (uniform spacing on the surface).
        For :class:`Cylinder` and :class:`AxisymmetricBody` entities, `spacing_axial`, `spacing_radial`,
        and `spacing_circumferential` are all required.

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

    Using a Sphere entity (spherical sliding interface):

      >>> fl.RotationVolume(
      ...     name="RotationSphere",
      ...     spacing_circumferential=0.3*fl.u.m,
      ...     entities=sphere
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
    entities: EntityList[Cylinder, AxisymmetricBody, Sphere] = pd.Field()
    enclosed_entities: Optional[
        EntityList[Cylinder, Surface, MirroredSurface, AxisymmetricBody, Box, Sphere]
    ] = pd.Field(
        None,
        description=(
            "Entities enclosed by :class:`RotationVolume`. "
            "Can be :class:`~flow360.Surface` and/or other :class:`~flow360.Cylinder`"
            "and/or other :class:`~flow360.AxisymmetricBody`"
            "and/or other :class:`~flow360.Box`"
            "and/or other :class:`~flow360.Sphere`"
        ),
    )
    stationary_enclosed_entities: Optional[EntityList[Surface, MirroredSurface]] = pd.Field(
        None,
        description=(
            "Surface entities included in `enclosed_entities` which should remain stationary "
            "(excluded from rotation)."
        ),
    )
    # pylint: disable=no-member
    spacing_axial: Optional[LengthType.Positive] = pd.Field(
        None, description="Spacing along the axial direction."
    )
    spacing_radial: Optional[LengthType.Positive] = pd.Field(
        None, description="Spacing along the radial direction."
    )
    # This is actually a required field for all of Sphere, Cylinder, AxisymmetricBody entity
    # RotationVolumes, but making this not Optional causes validation to be triggered in pydantic
    # vs in validator below, giving different error messages than what we want.
    # Use of validation_default=False messes up schemas.
    spacing_circumferential: Optional[LengthType.Positive] = pd.Field(
        None, description="Spacing along the circumferential direction."
    )

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def _validate_single_instance_in_entity_list(cls, values, param_info: ParamsValidationInfo):
        """
        [CAPABILITY-LIMITATION]
        Only single instance is allowed in entities for each `RotationVolume`.
        """
        # Note: Should be fine without expansion since we only allow Draft entities here.
        # But using expand_entity_list for consistency and future-proofing.
        expanded_entities = param_info.expand_entity_list(values)
        if len(expanded_entities) > 1:
            raise ValueError(
                "Only single instance is allowed in entities for each `RotationVolume`."
            )
        return values

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def _validate_cylinder_name_length(cls, values, param_info: ParamsValidationInfo):
        """
        Check the name length for the cylinder entities due to the 32-character
        limitation of all data structure names and labels in CGNS format.
        The current prefix is 'rotatingBlock-' with 14 characters.
        """
        if param_info.is_beta_mesher:
            return values

        expanded_entities = param_info.expand_entity_list(values)
        cgns_max_zone_name_length = 32
        max_cylinder_name_length = cgns_max_zone_name_length - len("rotatingBlock-")
        for entity in expanded_entities:
            if isinstance(entity, Cylinder) and len(entity.name) > max_cylinder_name_length:
                raise ValueError(
                    f"The name ({entity.name}) of `Cylinder` entity in `RotationVolume` "
                    + f"exceeds {max_cylinder_name_length} characters limit."
                )
        return values

    @contextual_field_validator("enclosed_entities", mode="after")
    @classmethod
    def _validate_enclosed_entities_beta_mesher_only(cls, values, param_info: ParamsValidationInfo):
        """
        Ensure that Box and Sphere entities in enclosed_entities are only used with the beta mesher.
        """
        if values is None:
            return values
        if param_info.is_beta_mesher:
            return values

        expanded = param_info.expand_entity_list(values)  # Can Have `Surface`
        for entity in expanded:
            if isinstance(entity, Box):
                raise ValueError(
                    "`Box` entity in `RotationVolume.enclosed_entities` is only supported with the beta mesher."
                )
            if isinstance(entity, Sphere):
                raise ValueError(
                    "`Sphere` entity in `RotationVolume.enclosed_entities` is only supported with the beta mesher."
                )

        return values

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def _validate_entities_beta_mesher_only(cls, values, param_info: ParamsValidationInfo):
        """
        Ensure that AxisymmetricBody and Sphere entities are only used with the beta mesher.
        """
        if param_info.is_beta_mesher:
            return values

        expanded_entities = param_info.expand_entity_list(values)
        for entity in expanded_entities:
            if isinstance(entity, AxisymmetricBody):
                raise ValueError(
                    "`AxisymmetricBody` entity for `RotationVolume` is only supported with the beta mesher."
                )
            if isinstance(entity, Sphere):
                raise ValueError(
                    "`Sphere` entity for `RotationVolume` is only supported with the beta mesher."
                )
        return values

    @contextual_field_validator("enclosed_entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        return validate_entity_list_surface_existence(value, param_info)

    @contextual_field_validator("stationary_enclosed_entities", mode="after")
    @classmethod
    def _validate_stationary_enclosed_entities_only_in_beta_mesher(
        cls, values, param_info: ParamsValidationInfo
    ):
        """
        Ensure that stationary_enclosed_entities is only used with the beta mesher.
        """
        if values is None:
            return values
        if not param_info.is_beta_mesher:
            raise ValueError(
                "`stationary_enclosed_entities` in `RotationVolume` is only supported with the beta mesher."
            )
        return values

    @contextual_model_validator(mode="after")
    def _validate_stationary_enclosed_entities_subset(self, param_info: ParamsValidationInfo):
        """
        Ensure that stationary_enclosed_entities is a subset of enclosed_entities.
        """
        if self.stationary_enclosed_entities is None:
            return self

        if self.enclosed_entities is None:
            raise ValueError(
                "`stationary_enclosed_entities` cannot be specified when `enclosed_entities` is None."
            )

        # Get sets of entity names for comparison
        # pylint: disable=no-member
        expanded_enclosed_entities = param_info.expand_entity_list(self.enclosed_entities)
        enclosed_names = {entity.name for entity in expanded_enclosed_entities}
        expanded_stationary_enclosed_entities = param_info.expand_entity_list(
            self.stationary_enclosed_entities
        )
        stationary_names = {entity.name for entity in expanded_stationary_enclosed_entities}

        # Check if all stationary entities are in enclosed entities
        if not stationary_names.issubset(enclosed_names):
            missing_entities = stationary_names - enclosed_names
            raise ValueError(
                f"All entities in `stationary_enclosed_entities` must be present in `enclosed_entities`. "
                f"Missing entities: {', '.join(missing_entities)}"
            )

        return self

    @contextual_model_validator(mode="after")
    def _validate_spacing_requirements_by_entity_type(self, param_info: ParamsValidationInfo):
        """
        Validate spacing requirements based on entity type:
        - Sphere: only spacing_circumferential is required; spacing_axial and spacing_radial must not be specified
        - Cylinder/AxisymmetricBody: all three spacings are required
        """
        # Check if entity is a Sphere
        # pylint: disable=no-member
        expanded_entities = param_info.expand_entity_list(self.entities)
        has_sphere = any(isinstance(entity, Sphere) for entity in expanded_entities)
        has_cylinder_or_axisymmetric = any(
            isinstance(entity, (Cylinder, AxisymmetricBody)) for entity in expanded_entities
        )

        if has_sphere:
            if self.spacing_circumferential is None:
                raise ValueError(
                    "`spacing_circumferential` is required for `Sphere` entities in `RotationVolume`."
                )
            if self.spacing_axial is not None:
                raise ValueError(
                    "`spacing_axial` must not be specified for `Sphere` entities. "
                    "Sphere uses only `spacing_circumferential` for uniform surface spacing."
                )
            if self.spacing_radial is not None:
                raise ValueError(
                    "`spacing_radial` must not be specified for `Sphere` entities. "
                    "Sphere uses only `spacing_circumferential` for uniform surface spacing."
                )

        if has_cylinder_or_axisymmetric:
            if self.spacing_axial is None:
                raise ValueError(
                    "`spacing_axial` is required for `Cylinder` or `AxisymmetricBody` entities "
                    "in `RotationVolume`."
                )
            if self.spacing_radial is None:
                raise ValueError(
                    "`spacing_radial` is required for `Cylinder` or `AxisymmetricBody` entities "
                    "in `RotationVolume`."
                )
            if self.spacing_circumferential is None:
                raise ValueError(
                    "`spacing_circumferential` is required for `Cylinder` or `AxisymmetricBody` "
                    "entities in `RotationVolume`."
                )

        return self


@deprecated(
    "The `RotationCylinder` class is deprecated! Use `RotationVolume`,"
    "which supports `Cylinder`, `AxisymmetricBody`, and `Sphere` entities instead."
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


### BEGIN FARFIELDS ###


class _FarfieldBase(Flow360BaseModel):
    """Base class for farfield parameters."""

    domain_type: Optional[Literal["half_body_positive_y", "half_body_negative_y", "full_body"]] = (
        pd.Field(  # In the future, we will support more flexible half model types and full model via Union.
            None,
            description="""
            - half_body_positive_y: Trim to a half-model by slicing with the global Y=0 plane; keep the '+y' side for meshing and simulation.
            - half_body_negative_y: Trim to a half-model by slicing with the global Y=0 plane; keep the '-y' side for meshing and simulation.
            - full_body: Keep the full body for meshing and simulation without attempting to add symmetry planes.

            Warning: When using AutomatedFarfield or UserDefinedFarfield, setting `domain_type` overrides automatic symmetry plane detection.
            """,
        )
    )

    @contextual_field_validator("domain_type", mode="after")
    @classmethod
    def _validate_only_in_beta_mesher(cls, value, param_info: ParamsValidationInfo):
        """
        Ensure that domain_type is only used with the beta mesher and GAI.
        """
        if not value or (param_info.use_geometry_AI is True and param_info.is_beta_mesher is True):
            return value
        raise ValueError(
            "`domain_type` is only supported when using both GAI surface mesher and beta volume mesher."
        )

    @pd.field_validator("domain_type", mode="after")
    @classmethod
    def _validate_domain_type_bbox(cls, value):
        """
        Ensure that when domain_type is used, the model actually spans across Y=0.
        """
        validation_info = get_validation_info()
        if validation_info is None:
            return value

        if (
            value not in ("half_body_positive_y", "half_body_negative_y")
            or validation_info.global_bounding_box is None
            or validation_info.planar_face_tolerance is None
        ):
            return value

        y_min = validation_info.global_bounding_box[0][1]
        y_max = validation_info.global_bounding_box[1][1]

        largest_dimension = -float("inf")
        for dim in range(3):
            dimension = (
                validation_info.global_bounding_box[1][dim]
                - validation_info.global_bounding_box[0][dim]
            )
            largest_dimension = max(largest_dimension, dimension)

        tolerance = largest_dimension * validation_info.planar_face_tolerance

        # Check if model crosses Y=0
        crossing = y_min < -tolerance and y_max > tolerance
        if crossing:
            return value

        # If not crossing, check if it matches the requested domain
        if value == "half_body_positive_y":
            # Should be on positive side (y > 0)
            if y_min >= -tolerance:
                return value

        if value == "half_body_negative_y":
            # Should be on negative side (y < 0)
            if y_max <= tolerance:
                return value

        message = (
            f"The model does not cross the symmetry plane (Y=0) with tolerance {tolerance:.2g}. "
            f"Model Y range: [{y_min:.2g}, {y_max:.2g}]. "
            "Please check if `domain_type` is set correctly."
        )
        if getattr(validation_info, "entity_transformation_detected", False):
            add_validation_warning(message)
            return value
        raise ValueError(message)


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
        GenericVolume(
            name="__farfield_zone_name_not_properly_set_yet",
            private_attribute_id="farfield_zone_name_not_properly_set_yet",
        ),
        frozen=True,
        exclude=True,
    )
    relative_size: pd.PositiveFloat = pd.Field(
        default=50.0,
        description="Radius of the far-field (semi)sphere/cylinder relative to "
        "the max dimension of the geometry bounding box.",
    )
    enclosed_surfaces: Optional[EntityList[Surface]] = pd.Field(
        None,
        description=(
            "Geometry surfaces that, together with the farfield surface, form the boundary of the "
            "exterior farfield zone. Required when using CustomVolumes alongside an AutomatedFarfield. "
        ),
    )

    @property
    def farfield(self):
        """Returns the farfield boundary surface."""
        # Make sure the naming is the same here and what the geometry/surface mesh pipeline generates.
        return GhostSurface(name="farfield", private_attribute_id="farfield")

    @property
    def symmetry_plane(self) -> GhostSurface:
        """
        Returns the symmetry plane boundary surface.
        """
        if self.method == "auto":
            return GhostSurface(name="symmetric", private_attribute_id="symmetric")
        raise Flow360ValueError(
            "Unavailable for quasi-3d farfield methods. Please use `symmetry_planes` property instead."
        )

    @property
    def symmetry_planes(self):
        """Returns the symmetry plane boundary surface(s)."""
        # Make sure the naming is the same here and what the geometry/surface mesh pipeline generates.
        if self.method == "auto":
            return GhostSurface(name="symmetric", private_attribute_id="symmetric")
        if self.method in ("quasi-3d", "quasi-3d-periodic"):
            return [
                GhostSurface(name="symmetric-1", private_attribute_id="symmetric-1"),
                GhostSurface(name="symmetric-2", private_attribute_id="symmetric-2"),
            ]
        raise Flow360ValueError(f"Unsupported method: {self.method}")

    @contextual_field_validator("method", mode="after")
    @classmethod
    def _validate_quasi_3d_periodic_only_in_legacy_mesher(
        cls, values, param_info: ParamsValidationInfo
    ):
        """
        Check mesher and AutomatedFarfield method compatibility
        """
        if param_info.is_beta_mesher and values == "quasi-3d-periodic":
            raise ValueError("Only legacy mesher can support quasi-3d-periodic")
        return values


class UserDefinedFarfield(_FarfieldBase):
    """
    Setting for user defined farfield zone generation.
    This means the "farfield" boundaries are coming from the supplied geometry file
    and meshing will take place inside this "geometry".

    **Important:** By default, the volume mesher will grow boundary layers on :class:`~flow360.UserDefinedFarfield`.
    Use :class:`~flow360.PassiveSpacing` to project or disable boundary layer growth.

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
        if self.domain_type not in (None, "half_body_positive_y", "half_body_negative_y"):
            # We allow None here to allow auto detection of domain type from bounding box.
            raise Flow360ValueError(
                "Symmetry plane of user defined farfield is only supported when domain_type "
                "is `half_body_positive_y`, `half_body_negative_y`, or None (auto detection)."
            )
        return GhostSurface(name="symmetric", private_attribute_id="symmetric")


# pylint: disable=no-member
class StaticFloor(Flow360BaseModel):
    """Class for static wind tunnel floor with friction patch."""

    type_name: Literal["StaticFloor"] = pd.Field(
        "StaticFloor", description="Static floor with friction patch.", frozen=True
    )
    friction_patch_x_range: LengthType.Range = pd.Field(
        default=(-3, 6) * u.m, description="(Minimum, maximum) x of friction patch."
    )
    friction_patch_width: LengthType.Positive = pd.Field(
        default=2 * u.m, description="Width of friction patch."
    )


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
    central_belt_x_range: LengthType.Range = pd.Field(
        default=(-2, 2) * u.m, description="(Minimum, maximum) x of central belt."
    )
    central_belt_width: LengthType.Positive = pd.Field(
        default=1.2 * u.m, description="Width of central belt."
    )


class WheelBelts(CentralBelt):
    """Class for wind tunnel floor with one central belt and four wheel belts."""

    type_name: Literal["WheelBelts"] = pd.Field(
        "WheelBelts",
        description="Floor with central belt and four wheel belts.",
        frozen=True,
    )
    # No defaults for the below; user must specify
    front_wheel_belt_x_range: LengthType.Range = pd.Field(
        description="(Minimum, maximum) x of front wheel belt."
    )
    front_wheel_belt_y_range: LengthType.PositiveRange = pd.Field(
        description="(Inner, outer) y of front wheel belt."
    )
    rear_wheel_belt_x_range: LengthType.Range = pd.Field(
        description="(Minimum, maximum) x of rear wheel belt."
    )
    rear_wheel_belt_y_range: LengthType.PositiveRange = pd.Field(
        description="(Inner, outer) y of rear wheel belt."
    )

    @pd.model_validator(mode="after")
    def _validate_wheel_belt_ranges(self):
        if self.front_wheel_belt_x_range[1] >= self.rear_wheel_belt_x_range[0]:
            raise ValueError(
                f"Front wheel belt maximum x ({self.front_wheel_belt_x_range[1]}) "
                f"must be less than rear wheel belt minimum x ({self.rear_wheel_belt_x_range[0]})."
            )

        # Central belt is centered at y=0 and extends from -width/2 to +width/2
        # It must fit within the inner edges of the wheel belts
        front_wheel_inner_edge = self.front_wheel_belt_y_range[0]
        rear_wheel_inner_edge = self.rear_wheel_belt_y_range[0]

        # Validate central belt width against front wheel belt inner edge
        if self.central_belt_width > 2 * front_wheel_inner_edge:
            raise ValueError(
                f"Central belt width ({self.central_belt_width}) "
                f"must be less than or equal to twice the front wheel belt inner edge "
                f"(2 × {front_wheel_inner_edge} = {2 * front_wheel_inner_edge})."
            )

        # Validate central belt width against rear wheel belt inner edge
        if self.central_belt_width > 2 * rear_wheel_inner_edge:
            raise ValueError(
                f"Central belt width ({self.central_belt_width}) "
                f"must be less than or equal to twice the rear wheel belt inner edge "
                f"(2 × {rear_wheel_inner_edge} = {2 * rear_wheel_inner_edge})."
            )

        return self


# pylint: disable=no-member
class WindTunnelFarfield(_FarfieldBase):
    """
    Settings for analytic wind tunnel farfield generation.
    The user only needs to provide tunnel dimensions and floor type and dimensions, rather than a geometry.

    **Important:** By default, the volume mesher will grow boundary layers on :class:`~flow360.WindTunnelFarfield`.
    Use :class:`~flow360.PassiveSpacing` to project or disable boundary layer growth.

    Example
    -------
        >>> fl.WindTunnelFarfield(
            width = 10 * fl.u.m,
            height = 5 * fl.u.m,
            inlet_x_position = -10 * fl.u.m,
            outlet_x_position = 20 * fl.u.m,
            floor_z_position = 0 * fl.u.m,
            floor_type = fl.CentralBelt(
                central_belt_x_range = (-1, 4) * fl.u.m,
                central_belt_width = 1.2 * fl.u.m
            )
        )
    """

    model_config = pd.ConfigDict(ignored_types=(classproperty,))

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
    ] = pd.Field(
        default_factory=StaticFloor,
        description="Floor type of the wind tunnel.",
        discriminator="type_name",
    )

    # up direction not yet supported; assume +Z

    @property
    def symmetry_plane(self) -> GhostSurface:
        """
        Returns the symmetry plane boundary surface for half body domains.
        """
        if self.domain_type not in ("half_body_positive_y", "half_body_negative_y"):
            raise Flow360ValueError(
                "Symmetry plane for wind tunnel farfield is only supported when domain_type "
                "is `half_body_positive_y` or `half_body_negative_y`."
            )
        return GhostSurface(name="symmetric", private_attribute_id="symmetric")

    # pylint: disable=no-self-argument
    @classproperty
    def left(cls):
        """Return the ghost surface representing the tunnel's left wall."""
        return WindTunnelGhostSurface(name="windTunnelLeft", private_attribute_id="windTunnelLeft")

    @classproperty
    def right(cls):
        """Return the ghost surface representing the tunnel's right wall."""
        return WindTunnelGhostSurface(
            name="windTunnelRight", private_attribute_id="windTunnelRight"
        )

    @classproperty
    def inlet(cls):
        """Return the ghost surface corresponding to the wind tunnel inlet."""
        return WindTunnelGhostSurface(
            name="windTunnelInlet", private_attribute_id="windTunnelInlet"
        )

    @classproperty
    def outlet(cls):
        """Return the ghost surface corresponding to the wind tunnel outlet."""
        return WindTunnelGhostSurface(
            name="windTunnelOutlet", private_attribute_id="windTunnelOutlet"
        )

    @classproperty
    def ceiling(cls):
        """Return the ghost surface for the tunnel ceiling."""
        return WindTunnelGhostSurface(
            name="windTunnelCeiling", private_attribute_id="windTunnelCeiling"
        )

    @classproperty
    def floor(cls):
        """Return the ghost surface for the tunnel floor."""
        return WindTunnelGhostSurface(
            name="windTunnelFloor", private_attribute_id="windTunnelFloor"
        )

    @classproperty
    def friction_patch(cls):
        """Return the ghost surface for the floor friction patch used by static floors."""
        return WindTunnelGhostSurface(
            name="windTunnelFrictionPatch",
            used_by=["StaticFloor"],
            private_attribute_id="windTunnelFrictionPatch",
        )

    @classproperty
    def central_belt(cls):
        """Return the ghost surface used by central and wheel belt floor types."""
        return WindTunnelGhostSurface(
            name="windTunnelCentralBelt",
            used_by=["CentralBelt", "WheelBelts"],
            private_attribute_id="windTunnelCentralBelt",
        )

    @classproperty
    def front_wheel_belts(cls):
        """Return the ghost surface for the front wheel belt region."""
        return WindTunnelGhostSurface(
            name="windTunnelFrontWheelBelt",
            used_by=["WheelBelts"],
            private_attribute_id="windTunnelFrontWheelBelt",
        )

    @classproperty
    def rear_wheel_belts(cls):
        """Return the ghost surface for the rear wheel belt region."""
        return WindTunnelGhostSurface(
            name="windTunnelRearWheelBelt",
            used_by=["WheelBelts"],
            private_attribute_id="windTunnelRearWheelBelt",
        )

    # pylint: enable=no-self-argument

    @staticmethod
    def _get_valid_ghost_surfaces(
        floor_string: Optional[str] = "all", domain_string: Optional[str] = None
    ) -> list[WindTunnelGhostSurface]:
        """
        Returns a list of valid ghost surfaces given a floor type as a string
        or ``all``, and the domain type as a string.
        """
        common_ghost_surfaces = [
            WindTunnelFarfield.inlet,
            WindTunnelFarfield.outlet,
            WindTunnelFarfield.ceiling,
            WindTunnelFarfield.floor,
        ]
        if domain_string != "half_body_negative_y":
            common_ghost_surfaces += [WindTunnelFarfield.right]
        if domain_string != "half_body_positive_y":
            common_ghost_surfaces += [WindTunnelFarfield.left]
        for ghost_surface_type in [
            WindTunnelFarfield.friction_patch,
            WindTunnelFarfield.central_belt,
            WindTunnelFarfield.front_wheel_belts,
            WindTunnelFarfield.rear_wheel_belts,
        ]:
            if floor_string == "all" or floor_string in ghost_surface_type.used_by:
                common_ghost_surfaces += [ghost_surface_type]
        return common_ghost_surfaces

    @pd.model_validator(mode="after")
    def _validate_inlet_is_less_than_outlet(self):
        if self.inlet_x_position >= self.outlet_x_position:
            raise ValueError(
                f"Inlet x position ({self.inlet_x_position}) "
                f"must be less than outlet x position ({self.outlet_x_position})."
            )
        return self

    @pd.model_validator(mode="after")
    def _validate_central_belt_ranges(self):
        # friction patch
        if isinstance(self.floor_type, StaticFloor):
            if self.floor_type.friction_patch_width >= self.width:
                raise ValueError(
                    f"Friction patch width ({self.floor_type.friction_patch_width}) "
                    f"must be less than wind tunnel width ({self.width})"
                )
            if self.floor_type.friction_patch_x_range[0] <= self.inlet_x_position:
                raise ValueError(
                    f"Friction patch minimum x ({self.floor_type.friction_patch_x_range[0]}) "
                    f"must be greater than inlet x ({self.inlet_x_position})"
                )
            if self.floor_type.friction_patch_x_range[1] >= self.outlet_x_position:
                raise ValueError(
                    f"Friction patch maximum x ({self.floor_type.friction_patch_x_range[1]}) "
                    f"must be less than outlet x ({self.outlet_x_position})"
                )
        # central belt
        elif isinstance(self.floor_type, CentralBelt):
            if self.floor_type.central_belt_width >= self.width:
                raise ValueError(
                    f"Central belt width ({self.floor_type.central_belt_width}) "
                    f"must be less than wind tunnel width ({self.width})"
                )
            if self.floor_type.central_belt_x_range[0] <= self.inlet_x_position:
                raise ValueError(
                    f"Central belt minimum x ({self.floor_type.central_belt_x_range[0]}) "
                    f"must be greater than inlet x ({self.inlet_x_position})"
                )
            if self.floor_type.central_belt_x_range[1] >= self.outlet_x_position:
                raise ValueError(
                    f"Central belt maximum x ({self.floor_type.central_belt_x_range[1]}) "
                    f"must be less than outlet x ({self.outlet_x_position})"
                )
        return self

    @pd.model_validator(mode="after")
    def _validate_wheel_belts_ranges(self):
        if isinstance(self.floor_type, WheelBelts):
            if self.floor_type.front_wheel_belt_y_range[1] >= self.width * 0.5:
                raise ValueError(
                    f"Front wheel outer y ({self.floor_type.front_wheel_belt_y_range[1]}) "
                    f"must be less than half of wind tunnel width ({self.width * 0.5})"
                )
            if self.floor_type.rear_wheel_belt_y_range[1] >= self.width * 0.5:
                raise ValueError(
                    f"Rear wheel outer y ({self.floor_type.rear_wheel_belt_y_range[1]}) "
                    f"must be less than half of wind tunnel width ({self.width * 0.5})"
                )
            if self.floor_type.front_wheel_belt_x_range[0] <= self.inlet_x_position:
                raise ValueError(
                    f"Front wheel minimum x ({self.floor_type.front_wheel_belt_x_range[0]}) "
                    f"must be greater than inlet x ({self.inlet_x_position})"
                )
            if self.floor_type.rear_wheel_belt_x_range[1] >= self.outlet_x_position:
                raise ValueError(
                    f"Rear wheel maximum x ({self.floor_type.rear_wheel_belt_x_range[1]}) "
                    f"must be less than outlet x ({self.outlet_x_position})"
                )
        return self

    @contextual_model_validator(mode="after")
    def _validate_requires_geometry_ai(self, param_info: ParamsValidationInfo):
        """Ensure WindTunnelFarfield is only used when GeometryAI is enabled."""
        if not param_info.use_geometry_AI:
            raise ValueError("WindTunnelFarfield is only supported when Geometry AI is enabled.")
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
    include_crinkled_slices: bool = pd.Field(
        default=False,
        description="Generate crinkled slices in addition to flat slices.",
    )
    cutoff_radius: Optional[LengthType.Positive] = pd.Field(
        default=None,
        description="Cutoff radius of the slice output. If not specified, "
        "the slice extends to the boundaries of the volume mesh.",
    )
    output_type: Literal["MeshSliceOutput"] = pd.Field("MeshSliceOutput", frozen=True)


class CustomZones(Flow360BaseModel):
    """
    :class:`CustomZones` class for creating volume zones from custom volumes or seedpoint volumes.
    Names of the generated volume zones will be the names of the custom volumes.

    Example
    -------

      >>> fl.CustomZones(name="Custom zones", entities=[custom_volume1, custom_volume2], )

    ====
    """

    type: Literal["CustomZones"] = pd.Field("CustomZones", frozen=True)
    name: str = pd.Field("Custom zones", description="Name of the `CustomZones` meshing setting.")
    entities: EntityList[CustomVolume, SeedpointVolume] = pd.Field(
        description="The custom volume zones to be generated."
    )
    element_type: Literal["mixed", "tetrahedra"] = pd.Field(
        default="mixed",
        description="The element type to be used for the generated volume zones."
        + " - mixed: Mesher will automatically choose the element types used."
        + " - tetrahedra: Only tetrahedra element type will be used for the generated volume zones.",
    )
