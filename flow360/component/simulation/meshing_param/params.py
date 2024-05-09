from typing import List, Optional

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
        edge_refinement (Optional[List[EdgeRefinementTypes]]): edge (1D) refinement
        face_refinement (Optional[List[FaceRefinement]]): face (2D) refinement
        zone_refinement (Optional[List[ZoneRefinement]]): zone (3D) refinement
    """

    # Global fields:
    farfield: Optional[Farfield] = pd.Field()
    refinement_factor: Optional[pd.PositiveFloat] = pd.Field()
    gap_treatment_strength: Optional[float] = pd.Field(ge=0, le=1)

    edge_refinement: Optional[List[EdgeRefinementTypes]] = pd.Field()
    face_refinement: Optional[List[FaceRefinement]] = pd.Field()
    zone_refinement: Optional[List[ZoneRefinementTypes]] = pd.Field()
