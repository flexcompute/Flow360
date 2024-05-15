from typing import List, Optional, Union

import pydantic as pd
from edge_params import EdgeRefinementTypes
from face_params import FaceRefinement
from volume_params import Farfield, ZoneRefinementTypes

from flow360.component.simulation.base_model import Flow360BaseModel


class MeshingParameters(Flow360BaseModel):
    """
    Meshing parameters for volume and/or surface mesher.

    In `Simulation` this only contains what the user specifies. `Simulation` can derive and add more items according to other aspects of simulation. (E.g. BETDisk volume -> ZoneRefinement)

    Meshing related but may and maynot (user specified) need info from `Simulation`:
    1. Add rotational zones.
    2. Add default BETDisk refinement.

    Attributes:
    ----------
    farfield: Optional[Farfield]
        Farfield type for meshing.
    refinement_factor: Optional[pd.PositiveFloat]
        If refinementFactor=r is provided all spacings in refinement regions and first layer thickness will be adjusted to generate r-times finer mesh. For example, if refinementFactor=2, all spacings will be divided by 2**(1/3), so the resulting mesh will have approximately 2 times more nodes.
    gap_treatment_strength: Optional[float]
        Narrow gap treatment strength used when two surfaces are in close proximity. Use a value between 0 and 1, where 0 is no treatment and 1 is the most conservative treatment. This parameter has a global impact where the anisotropic transition into the isotropic mesh. However, the impact on regions without close proximity is negligible.
    refinements: Optional[List[Union[EdgeRefinementTypes, FaceRefinement, ZoneRefinementTypes]]]
        Refinements for meshing.
    """

    # Global fields:
    farfield: Optional[Farfield] = pd.Field()
    refinement_factor: Optional[pd.PositiveFloat] = pd.Field()
    gap_treatment_strength: Optional[float] = pd.Field(ge=0, le=1)

    refinements: Optional[List[Union[EdgeRefinementTypes, FaceRefinement, ZoneRefinementTypes]]] = (
        pd.Field()
    )  # Note: May need discriminator for performance??
