from typing import List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.meshing_param.edge_params import SurfaceEdgeRefinement
from flow360.component.simulation.meshing_param.face_params import (
    SurfaceRefinementTypes,
)
from flow360.component.simulation.meshing_param.volume_params import (
    RotationCylinder,
    VolumeRefinementTypes,
)


class MeshingParams(Flow360BaseModel):
    """
    Meshing parameters for volume and/or surface mesher.

    In `Simulation` this only contains what the user specifies. `Simulation` can derive and add more items according to other aspects of simulation. (E.g. BETDisk volume -> ZoneRefinement)

    Meshing related but may and maynot (user specified) need info from `Simulation`:
    1. Add rotational zones.
    2. Add default BETDisk refinement.

    Affects volume meshing:
    - farfield
    - refinement_factor
    - gap_treatment_strength
    - `class` BoundaryLayer
    - `class` UniformRefinement
    - `class` AxisymmetricRefinement
    - `class` RotationCylinder

    Affects surface meshing:
    - surface_layer_growth_rate
    - `class` SurfaceRefinement
    - `class` SurfaceEdgeRefinement
    """

    # Volume **defaults**:
    farfield: Optional[Literal["auto", "quasi-3d", "user-defined"]] = pd.Field(
        None, description="Type of farfield generation."
    )
    refinement_factor: Optional[pd.PositiveFloat] = pd.Field(
        None,
        description="If refinementFactor=r is provided all spacings in refinement regions and first layer thickness will be adjusted to generate r-times finer mesh.",
    )
    gap_treatment_strength: Optional[float] = pd.Field(
        None,
        ge=0,
        le=1,
        description="Narrow gap treatment strength used when two surfaces are in close proximity. Use a value between 0 and 1, where 0 is no treatment and 1 is the most conservative treatment. This parameter has a global impact where the anisotropic transition into the isotropic mesh. However, the impact on regions without close proximity is negligible.",
    )

    surface_layer_growth_rate: Optional[float] = pd.Field(
        None, ge=1, description="Global growth rate of the anisotropic layers grown from the edges."
    )  # Conditionally optional

    refinements: Optional[
        List[Union[SurfaceEdgeRefinement, SurfaceRefinementTypes, VolumeRefinementTypes]]
    ] = pd.Field(
        None,
        description="Additional fine-tunning for refinements.",
    )  # Note: May need discriminator for performance??
    # Will add more to the Union
    volume_zones: Optional[List[Union[RotationCylinder]]] = pd.Field(
        None, description="Creation of new volume zones."
    )
