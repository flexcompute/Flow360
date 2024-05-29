from typing import List, Literal, Optional, Union

import pydantic as pd
from edge_params import SurfaceEdgeRefinement
from face_params import FaceRefinement
from volume_params import CylindricalRefinement, ZoneRefinementTypes

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
    - volume_layer_growth_rate
    - volume_layer_first_layer_thickness
    - `class` UniformRefinement
    - `class` CylindricalRefinement

    Affects surface meshing:
    - max_edge_length
    - curvature_resolution_angle
    - surface_layer_growth_rate
    - `class` FaceRefinement
    - `class` AnisoSurfaceEdge
    - `class` ProjectAnisoSurfaceEdge
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
    volume_layer_growth_rate: pd.PositiveFloat = pd.Field(
        gt=1, description="Growth rate for volume prism layers."
    )
    volume_layer_first_layer_thickness: pd.PositiveFloat = pd.Field(
        description="Default first layer thickness for volumetric anisotropic layers."
    )

    # Face **defaults**
    max_edge_length: LengthType.PositiveFloat = pd.Field(
        description="Global maximum edge length for surface cells. This value will be overwritten by the local specification if provided."
    )  # Note: Applied to faces without faceName and therefore has to exist in root level.
    curvature_resolution_angle: pd.PositiveFloat = pd.Field(
        description="""
        Global maximum angular deviation in degrees. This value will restrict:
        (1) The angle between a cell’s normal and its underlying surface normal
        (2) The angle between a line segment’s normal and its underlying curve normal
        """
    )
    surface_layer_growth_rate: float = pd.Field(
        ge=1, description="Growth rate of the anisotropic layers grown from the edges."
    )

    refinements: Optional[
        List[Union[SurfaceEdgeRefinement, FaceRefinement, ZoneRefinementTypes]]
    ] = pd.Field(
        description="Additional fine-tunning for refinement and specifications."
    )  # Note: May need discriminator for performance??
