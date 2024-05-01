from typing import List, Optional, Union

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel

from .zones import BoxZone, CylindricalZone


class FaceRefinement(Flow360BaseModel):
    max_edge_length: float
    pass


class EdgeRefinement(Flow360BaseModel):
    pass


class ZoneRefinement:
    """
    Volumetric 3D meshing refinement
    """

    shape: Union[CylindricalZone, BoxZone] = pd.Field()
    spacing: float
    first_layer_thickness: float


class MeshingParameters(Flow360BaseModel):
    """
    Meshing parameters for volume and/or surface mesher.

    In `Simulation` this only contains what the user specifies. `Simulation` can derive and add more items according to other aspects of simulation. (E.g. BETDisk volume -> ZoneRefinement)

    Meshing related but may and maynot (user specified) need info from `Simulation`:
    1. Add rotational zones.
    2. Add default BETDisk refinement.

    Attributes:
        edge_refinement (Optional[List[EdgeRefinement]]): edge (1D) refinement
        face_refinement (Optional[List[FaceRefinement]]): face (2D) refinement
        zone_refinement (Optional[List[ZoneRefinement]]): zone (3D) refinement
    """

    edge_refinement: Optional[List[EdgeRefinement]] = pd.Field()
    face_refinement: Optional[List[FaceRefinement]] = pd.Field()
    zone_refinement: Optional[List[ZoneRefinement]] = pd.Field()
