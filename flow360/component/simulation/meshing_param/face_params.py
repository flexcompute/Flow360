"""Face based meshing parameters for meshing."""

from typing import Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.unit_system import LengthType


class SurfaceRefinement(Flow360BaseModel):
    """
    These affects surface meshing.

    Note:
    - `None` entities will be expanded (or just ignored and convert to global default, depending on implementation)
    before submission. This is supposed to be applied to all the matching entities. We allow this so that we do not
    need to have dedicated field for global settings. This is also consistent with the `FluidDynamics` class' design.

    - For `SurfaceRefinement` we may need validation to detect if default has been set or not. This is because we need
    these defaults so that the when face name is not present, what config we ues. Depending on how we go down the road.
    """

    name: Optional[str] = pd.Field(None)
    refinement_type: Literal["SurfaceRefinement"] = pd.Field("SurfaceRefinement", frozen=True)
    entities: EntityList[Surface] = pd.Field(alias="faces")
    # pylint: disable=no-member
    max_edge_length: LengthType.Positive = pd.Field(
        description="Local maximum edge length for surface cells."
    )


class PassiveSpacing(Flow360BaseModel):
    """
    Passively control the mesh spacing either through other face's meshing
    setting or doing nothing to change existing surface mesh at all.
    """

    name: Optional[str] = pd.Field(None)
    type: Literal["projected", "unchanged"] = pd.Field()
    refinement_type: Literal["PassiveSpacing"] = pd.Field("PassiveSpacing", frozen=True)
    entities: EntityList[Surface] = pd.Field(alias="faces")


class BoundaryLayer(Flow360BaseModel):
    """
    These affects volume meshing.
    """

    name: Optional[str] = pd.Field(None)
    refinement_type: Literal["BoundaryLayer"] = pd.Field("BoundaryLayer", frozen=True)
    entities: EntityList[Surface] = pd.Field(alias="faces")
    # pylint: disable=no-member
    first_layer_thickness: LengthType.Positive = pd.Field(
        description="First layer thickness for volumetric anisotropic layers for given face."
    )


SurfaceRefinementTypes = Union[SurfaceRefinement, BoundaryLayer, PassiveSpacing]
