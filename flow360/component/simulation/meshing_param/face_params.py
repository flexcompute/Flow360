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
    entities: Optional[EntityList[Surface]] = pd.Field(None, alias="faces")
    # pylint: disable=no-member
    max_edge_length: LengthType.Positive = pd.Field(
        description="Local maximum edge length for surface cells."
    )


class BoundaryLayer(Flow360BaseModel):
    """
    These affects volume meshing.
    Note:
    - We do not support per volume specification of these settings so the entities will be **obsolete** for now.
    Should we have it at all in the release?

    - `None` entities will be expanded (or just ignored and convert to global default, depending on implementation)
    before submission. This is supposed to be applied to all the matching entities. We allow this so that we do not
    need to have dedicated field for global settings. This is also consistent with the `FluidDynamics` class' design.
    """

    name: Optional[str] = pd.Field(None)
    refinement_type: Literal["BoundaryLayer"] = pd.Field("BoundaryLayer", frozen=True)
    type: Literal["aniso", "projectAnisoSpacing", "none"] = pd.Field(default="aniso")
    entities: Optional[EntityList[Surface]] = pd.Field(None, alias="faces")
    # pylint: disable=no-member
    first_layer_thickness: LengthType.Positive = pd.Field(
        description="First layer thickness for volumetric anisotropic layers."
    )
    # pylint: disable=no-member
    growth_rate: Optional[pd.PositiveFloat] = pd.Field(
        None, description="Growth rate for volume prism layers.", ge=1
    )  # Note:  Per face specification is actually not supported.
    # This is a global setting in mesher similar to curvature_resolution_angle.

    @pd.model_validator(mode="after")
    def _add_global_default_growth_rate(self):
        """
        [CAPABILITY-LIMITATION]
        Add **global** default for `growth_rate`.
        Cannot add default in field definition because that may imply it can be set per surface.
        self.entities is None indicates that this is a global setting.
        """
        if self.entities is None and self.growth_rate is None:
            self.growth_rate = 1.2
        return self


SurfaceRefinementTypes = Union[SurfaceRefinement, BoundaryLayer]
