"""
Meshing settings that applies to volumes.
"""

from abc import ABCMeta
from typing import Literal, Optional

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    GenericVolume,
    GhostSurface,
    Surface,
)
from flow360.component.simulation.unit_system import LengthType


class UniformRefinement(Flow360BaseModel):
    """Uniform spacing refinement inside specified region of mesh."""

    name: Optional[str] = pd.Field(None)
    refinement_type: Literal["UniformRefinement"] = pd.Field("UniformRefinement", frozen=True)
    entities: EntityList[Box, Cylinder] = pd.Field(
        description=":class:`UniformRefinement` can be applied to :class:`~flow360.Box` "
        + "and :class:`~flow360.Cylinder` regions."
    )
    # pylint: disable=no-member
    spacing: LengthType.Positive = pd.Field(description="The required refinement spacing.")


class CylindricalRefinementBase(Flow360BaseModel, metaclass=ABCMeta):
    """Base class for all refinements that requires spacing in axia, radial and circumferential directions."""

    # pylint: disable=no-member
    spacing_axial: LengthType.Positive = pd.Field(description="Spacing along the axial direction.")
    spacing_radial: LengthType.Positive = pd.Field(
        description="Spacing along the radial direction."
    )
    spacing_circumferential: LengthType.Positive = pd.Field(
        description="Spacing along the circumferential direction."
    )


class AxisymmetricRefinement(CylindricalRefinementBase):
    """
    - The mesh inside the :class:`AxisymmetricRefinement` is semi-structured.
    - The :class:`AxisymmetricRefinement` cannot enclose/intersect with other objects.
    - Users could create a donut-shape :class:`AxisymmetricRefinement` and place their hub/centerbody in the middle.
    - :class:`AxisymmetricRefinement` can be used for resolving the strong flow gradient
       along the axial direction for the actuator or BET disks.
    - The spacings along the axial, radial and circumferential directions can be adjusted independently.
    """

    name: Optional[str] = pd.Field(None)
    refinement_type: Literal["AxisymmetricRefinement"] = pd.Field(
        "AxisymmetricRefinement", frozen=True
    )
    entities: EntityList[Cylinder] = pd.Field()


class RotationCylinder(CylindricalRefinementBase):
    """
    - The mesh on :class:`RotationCylinder` is guaranteed to be concentric.
    - The :class:`RotationCylinder` is designed to enclose other objects, but it can’t intersect with other objects.
    - Users could create a donut-shape :class:`RotationCylinder` and put their stationary centerbody in the middle.
    - This type of volume zone can be used to generate volume zone compatible with :class:`~flow360.Rotation` model.
    """

    # Note: Please refer to
    # Note: https://www.notion.so/flexcompute/Python-model-design-document-
    # Note: 78d442233fa944e6af8eed4de9541bb1?pvs=4#c2de0b822b844a12aa2c00349d1f68a3

    type: Literal["RotationCylinder"] = pd.Field("RotationCylinder", frozen=True)
    name: Optional[str] = pd.Field(None, description="Name to display in the GUI.")
    entities: EntityList[Cylinder] = pd.Field()
    enclosed_entities: Optional[EntityList[Cylinder, Surface]] = pd.Field(
        None,
        description="Entities enclosed by :class:`RotationCylinder`. "
        + "Can be `Surface` and/or other :class:`~flow360.Cylinder` (s).",
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
                "Only single instance is allowed in entities for each RotationCylinder."
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

        cgns_max_zone_name_length = 32
        max_cylinder_name_length = cgns_max_zone_name_length - len("rotatingBlock-")
        for entity in values.stored_entities:
            if len(entity.name) > max_cylinder_name_length:
                raise ValueError(
                    f"The name ({entity.name}) of `Cylinder` entity in `RotationCylinder` "
                    + f"exceeds {max_cylinder_name_length} characters limit."
                )
        return values


class AutomatedFarfield(Flow360BaseModel):
    """
    Settings for automatic farfield volume zone generation.
    """

    type: Literal["AutomatedFarfield"] = pd.Field("AutomatedFarfield", frozen=True)
    name: Optional[str] = pd.Field(None)
    method: Literal["auto", "quasi-3d"] = pd.Field(
        default="auto",
        frozen=True,
        description="""
        - auto: The mesher will Sphere or semi-sphere will be generated based on the bounding box of the geometry.
            - Full sphere if min{Y} < 0 and max{Y} > 0.
            - +Y semi sphere if min{Y} = 0 and max{Y} > 0.
            - -Y semi sphere if min{Y} < 0 and max{Y} = 0.
        - quasi-3d: Thin disk will be generated for quasi 3D cases.
                    Both sides of the farfield disk will be treated as “symmetric plane”.
        """,
    )
    private_attribute_entity: GenericVolume = pd.Field(
        GenericVolume(name="__farfield_zone_name_not_properly_set_yet"), frozen=True, exclude=True
    )

    @property
    def farfield(self):
        """Returns the farfield boundary surface."""
        return GhostSurface(name="farfield")

    @property
    def symmetry_planes(self):
        """Returns the symmetry plane boundary surface(s)."""
        if self.method == "auto":
            return GhostSurface(name="symmetric")
        if self.method == "quasi-3d":
            return [
                GhostSurface(name="symmetric-1"),
                GhostSurface(name="symmetric-2"),
            ]
        raise ValueError(f"Unsupported method: {self.method}")


class UserDefinedFarfield(Flow360BaseModel):
    """
    Setting for user defined farfield zone generation.
    This means the "farfield" boundaires are comming from the supplied geometry file
    and meshing will take place inside this "geometry".
    """

    type: Literal["UserDefinedFarfield"] = pd.Field("UserDefinedFarfield", frozen=True)
    name: Optional[str] = pd.Field(None)
