from typing import List, Optional, Union, Literal

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel

from .volumes import BoxZone, CylindricalZone


class MeshingBase(Flow360BaseModel):
    entities = pd.Field()


####:: Edge refinement ::####


class Aniso(MeshingBase):
    """Aniso edge"""

    type: str = pd.Field("aniso", frozen=True)
    method: Literal["angle", "height", "aspectRatio"] = pd.Field()
    value: pd.PositiveFloat = pd.Field()
    adapt: Optional[bool] = pd.Field()


class ProjectAniso(MeshingBase):
    """ProjectAniso edge"""

    type: str = pd.Field("projectAnisoSpacing", frozen=True)
    adapt: Optional[bool] = pd.Field()


class UseAdjacent(MeshingBase):
    """UseAdjacent edge"""

    type: str = pd.Field("useAdjacent", frozen=True)
    adapt: Optional[bool] = pd.Field()


EdgeRefinementTypes = Union[Aniso, ProjectAniso, UseAdjacent]


####:: Face refinement ::####


class Farfield(Flow360BaseModel):
    """
    Farfield type for meshing
    """

    type: Literal["auto", "quasi-3d"] = pd.Field()


class FaceRefinement(MeshingBase):
    max_edge_length: pd.PositiveFloat = pd.Field()
    curvature_resolution_angle: float = pd.Field()
    growth_rate: float = pd.Field()


class ZoneRefinement(MeshingBase):

    shape: Union[CylindricalZone, BoxZone] = pd.Field()
    spacing: float = pd.Field()
    first_layer_thickness: float
    refinement_factor: Optional[pd.PositiveFloat] = pd.Field()
    growth_rate: Optional[pd.PositiveFloat] = pd.Field()


class RotationalModelBase(Flow360BaseModel):
    """:class: RotorDisk"""

    inner_radius: Optional[pd.NonNegativeFloat] = pd.Field(default=0)
    outer_radius: pd.PositiveFloat = pd.Field()
    thickness: pd.PositiveFloat = pd.Field()
    center: Coordinate = pd.Field()
    spacing_axial: PositiveFloat = pd.Field(alias="spacingAxial")
    spacing_radial: PositiveFloat = pd.Field(alias="spacingRadial")
    spacing_circumferential: PositiveFloat = pd.Field(alias="spacingCircumferential")


class RotorDisk(RotationalModelBase):
    """:class: RotorDisk"""

    axis_thrust: Axis = pd.Field(alias="axisThrust")


class SlidingInterface(RotationalModelBase):
    """:class: SlidingInterface for meshing"""

    axis_of_rotation: Axis = pd.Field(alias="axisOfRotation")
    enclosed_objects: Optional[List[str]] = pd.Field(alias="enclosedObjects", default=[])


ZoneRefinementTypes = []


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

    farfield: Optional[Farfield] = pd.Field()

    edge_refinement: Optional[List[EdgeRefinementTypes]] = pd.Field()
    face_refinement: Optional[List[FaceRefinement]] = pd.Field()
    zone_refinement: Optional[List[ZoneRefinementTypes]] = pd.Field()
