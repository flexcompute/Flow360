from typing import List, Literal, Optional, Union

import pydantic as pd
from edge_params import SurfaceEdgeRefinement
from face_params import FaceRefinement
from volume_params import AxisymmetricRefinement, ZoneRefinementTypes

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.unique_list import UniqueItemList


class MeshingParameters(Flow360BaseModel):
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
    - `class` BoundaryLayerRefinement
    - `class` UniformRefinement
    - `class` AxisymmetricRefinement

    Affects surface meshing:
    - surface_layer_growth_rate
    - `class` FaceRefinement
    - `class` SurfaceEdgeRefinement
    """

    # Volume **defaults**:
    farfield: Literal["auto", "quasi-3d", "user-defined"] = pd.Field(
        description="Type of farfield generation."
    )
    refinement_factor: pd.PositiveFloat = pd.Field(
        description="If refinementFactor=r is provided all spacings in refinement regions and first layer thickness will be adjusted to generate r-times finer mesh."
    )
    gap_treatment_strength: float = pd.Field(
        ge=0,
        le=1,
        description="Narrow gap treatment strength used when two surfaces are in close proximity. Use a value between 0 and 1, where 0 is no treatment and 1 is the most conservative treatment. This parameter has a global impact where the anisotropic transition into the isotropic mesh. However, the impact on regions without close proximity is negligible.",
    )

    surface_layer_growth_rate: float = pd.Field(
        ge=1, description="Global growth rate of the anisotropic layers grown from the edges."
    )

    refinements: Optional[
        List[Union[SurfaceEdgeRefinement, FaceRefinement, ZoneRefinementTypes]]
    ] = pd.Field(
        None, description="Additional fine-tunning for refinement and specifications."
    )  # Note: May need discriminator for performance??
