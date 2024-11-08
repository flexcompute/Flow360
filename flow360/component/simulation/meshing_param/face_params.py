"""Face based meshing parameters for meshing."""

from typing import Literal, Optional

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.unit_system import LengthType


class SurfaceRefinement(Flow360BaseModel):
    """
    Setting for refining surface elements for given `Surface`.
    """

    name: Optional[str] = pd.Field(None)
    refinement_type: Literal["SurfaceRefinement"] = pd.Field("SurfaceRefinement", frozen=True)
    entities: EntityList[Surface] = pd.Field(alias="faces")
    # pylint: disable=no-member
    max_edge_length: LengthType.Positive = pd.Field(
        description="Maximum edge length of surface cells."
    )


class PassiveSpacing(Flow360BaseModel):
    """
    Passively control the mesh spacing either through adjecent `Surface`'s meshing
    setting or doing nothing to change existing surface mesh at all.
    """

    name: Optional[str] = pd.Field(None)
    type: Literal["projected", "unchanged"] = pd.Field(
        description="""
        1. When set to *projected*, turn off anisotropic layers growing for this `Surface`. 
        Project the anisotropic spacing from the neighboring volumes to this face.

        2. When set to *unchanged*, turn off anisotropic layers growing for this `Surface`. 
        The surface mesh will remain unaltered when populating the volume mesh.
        """
    )
    refinement_type: Literal["PassiveSpacing"] = pd.Field("PassiveSpacing", frozen=True)
    entities: EntityList[Surface] = pd.Field(alias="faces")


class BoundaryLayer(Flow360BaseModel):
    """
    Setting for growing anisotropic layers orthogonal to the specified `Surface` (s).
    """

    name: Optional[str] = pd.Field(None)
    refinement_type: Literal["BoundaryLayer"] = pd.Field("BoundaryLayer", frozen=True)
    entities: EntityList[Surface] = pd.Field(alias="faces")
    # pylint: disable=no-member
    first_layer_thickness: LengthType.Positive = pd.Field(
        description="First layer thickness for volumetric anisotropic layers grown from given `Surface` (s)."
    )
