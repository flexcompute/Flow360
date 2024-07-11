"""
Meshing settings that applies to volumes.
"""

from abc import ABCMeta
from typing import Literal, Optional, Union

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
from flow360.component.simulation.utils import _model_attribute_unlock


class UniformRefinement(Flow360BaseModel):
    """Uniform spacing refinement."""

    name: Optional[str] = pd.Field(None)
    refinement_type: Literal["UniformRefinement"] = pd.Field("UniformRefinement", frozen=True)
    entities: EntityList[Box, Cylinder] = pd.Field()
    # pylint: disable=no-member
    spacing: LengthType.Positive = pd.Field()


class CylindricalRefinementBase(Flow360BaseModel, metaclass=ABCMeta):
    """Base class for all refinements that requires spacing in axia, radial and circumferential directions."""

    # pylint: disable=no-member
    spacing_axial: LengthType.Positive = pd.Field()
    spacing_radial: LengthType.Positive = pd.Field()
    spacing_circumferential: LengthType.Positive = pd.Field()


class AxisymmetricRefinement(CylindricalRefinementBase):
    """
    Note:
    - This basically creates the "rotorDisks" type of volume refinement that we used to have.



    - We may provide a helper function to automatically determine what is inside the encloeud_objects list based on
    the mesh data. But this currently is out of scope due to the estimated efforts.
    """

    name: Optional[str] = pd.Field(None)
    refinement_type: Literal["AxisymmetricRefinement"] = pd.Field(
        "AxisymmetricRefinement", frozen=True
    )
    entities: EntityList[Cylinder] = pd.Field()
    # pylint: disable=no-member


VolumeRefinementTypes = Union[UniformRefinement, AxisymmetricRefinement]


class RotationCylinder(CylindricalRefinementBase):
    """
    This is the original SlidingInterface. This will create new volume zones
    Will add RotationSphere class in the future.
    Please refer to
    https://www.notion.so/flexcompute/Python-model-design-document-
    78d442233fa944e6af8eed4de9541bb1?pvs=4#c2de0b822b844a12aa2c00349d1f68a3

    - `enclosed_entities` is actually just a way of specifying the enclosing patches of a volume zone.
    Therefore in the future when supporting arbitrary-axisymmetric shaped sliding interface, we may not need this
    attribute at all. For example if the new class already has an entry to list all the enclosing patches.

    """

    type: Literal["RotationCylinder"] = pd.Field("RotationCylinder", frozen=True)
    name: Optional[str] = pd.Field(None)
    entities: EntityList[Cylinder] = pd.Field()
    enclosed_entities: Optional[EntityList[Cylinder, Surface]] = pd.Field(
        None,
        description="""Entities enclosed by this sliding interface. Can be faces, boxes and/or other cylinders etc.
        This helps determining the volume zone boundary.""",
    )
    private_attribute_displayed_name: Optional[str] = pd.Field(None)


class AutomatedFarfield(Flow360BaseModel):
    """
    - auto: The mesher will Sphere or semi-sphere will be generated based on the bounding box of the geometry

        - Full sphere if min{Y} < 0 and max{Y} > 0

        - +Y semi sphere if min{Y} = 0 and max{Y} > 0

        - -Y semi sphere if min{Y} < 0 and max{Y} = 0

    - quasi-3d: Thin disk will be generated for quasi 3D cases.
                Both sides of the farfield disk will be treated as “symmetric plane”.

    - user-defined: The farfield shape is provided by the user in ESP.
                    Note: "user-defined" are left out due to scarce usage and will not be implemented.
    """

    type: Literal["AutomatedFarfield"] = pd.Field("AutomatedFarfield", frozen=True)
    name: Optional[str] = pd.Field(None)
    method: Literal["auto", "quasi-3d"] = pd.Field(default="auto", frozen=True)
    private_attribute_entity: GenericVolume = pd.Field(
        GenericVolume(name="__farfield_zone_name_not_properly_set_yet"), frozen=True, exclude=True
    )

    def _set_up_zone_entity(self, found_rotating_zones: bool):
        with _model_attribute_unlock(self.private_attribute_entity, "name"):
            # pylint: disable=assigning-non-slot
            if found_rotating_zones:
                self.private_attribute_entity.name = "stationaryBlock"
            else:
                self.private_attribute_entity.name = "fluid"

        with _model_attribute_unlock(
            self.private_attribute_entity, "private_attribute_zone_boundary_names"
        ):
            # Is setting private_attribute_zone_boundary_names meaningful at all?
            # We did not include other potential boundaries like walls or sliding interfaces
            # so it is most likely incomplete.
            # Besides in casePipeline we will overwrite private_attribute_zone_boundary_names
            # with volume mesh metadata anyway.
            self.private_attribute_entity.private_attribute_zone_boundary_names.append("farfield")
            if self.method == "auto":
                self.private_attribute_entity.private_attribute_zone_boundary_names.append(
                    "symmetric"
                )
            elif self.method == "quasi-3d":
                self.private_attribute_entity.private_attribute_zone_boundary_names.append(
                    "symmetric-1"
                )
                self.private_attribute_entity.private_attribute_zone_boundary_names.append(
                    "symmetric-2"
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
