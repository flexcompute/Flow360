from typing import List, Optional, Union

import pydantic as pd
from mesh import MeshingParameters
from operating_condition import OperatingConditionTypes
from references import ReferenceGeometry
from surfaces import SurfaceTypes
from volumes import VolumeTypes

from flow360.component.flow360_params.time_stepping import (
    SteadyTimeStepping,
    UnsteadyTimeStepping,
)
from flow360.component.simulation.base_model import Flow360BaseModel

from .inputs import Geometry, SurfaceMesh, VolumeMesh
from .outputs import OutputTypes


class UserDefinedDynamics(Flow360BaseModel):
    pass


class Simulation(Flow360BaseModel):
    """
    Simulation interface for user to submit a simulation starting from certain stage (geometry/surface mesh/volume mesh)

    Attributes:
        name (str): Name of simulation.
        tags (List[str]): List of tags to help classify the simulation.
    -----
        - Different stages of the simulation that can either come from cloud or local files. As the simulation progresses, each of these will get populated/updated if not specified at the beginning. All these attributes should have methods to compute/update/retrieve the params. Only one/zero of them can be specified in the `Simulation` constructor.

        geometry (Optional[Geometry]): Geometry.
        surface_mesh (Optional[SurfaceMesh]): Surface mesh.
        volume_mesh (Optional[VolumeMesh]): Volume mesh.

    -----
        meshing (Optional[MeshingParameters]): Contains all the user specified meshing parameters that either enrich or modify the existing surface/volume meshing parameters from starting points.

    -----
        - Global settings that gets applied by default to all volumes/surfaces. However per-volume/per-surface values will **always** overwrite global ones.

        reference_geometry (Optional[ReferenceGeometry]): Global geometric reference values.
        operating_condition (Optional[OperatingConditionTypes]): Global operating condition.
    -----
        - `volumes` and `surfaces` describes the physical problem **numerically**. Therefore `volumes` may/maynot necessarily have to map to grid volume zones (e.g. BETDisk). For now `surfaces` are used exclusivly for boundary conditions.

        volumes (Optional[List[VolumeTypes]]): Numerics/physics defined on a volume.
        surfaces (Optional[List[SurfaceTypes]]): Numerics/physics defined on a surface.
    -----
        - Other configurations that are orthogonal to all previous items.

        time_stepping (Optional[Union[SteadyTimeStepping, UnsteadyTimeStepping]]): Temporal aspects of simulation.
        user_defined_dynamics (Optional[UserDefinedDynamics]): Additional user-specified dynamics on top of the existing ones or how volumes/surfaces are intertwined.
        outputs (Optional[List[OutputTypes]]): Surface/Slice/Volume/Isosurface outputs.

    Limitations:
        Sovler capability:
            - Cannot specify multiple reference_geometry/operating_condition in volumes.
    """

    name: str = pd.Field()
    tags: Optional[List[str]] = pd.Field()
    #
    geometry: Optional[Geometry] = pd.Field()
    surface_mesh: Optional[SurfaceMesh] = pd.Field()
    volume_mesh: Optional[VolumeMesh] = pd.Field()
    #
    meshing: Optional[MeshingParameters] = pd.Field()

    reference_geometry: Optional[ReferenceGeometry] = pd.Field()
    operating_condition: Optional[OperatingConditionTypes] = pd.Field()
    #
    """
    meshing->edge_refinement, face_refinement, zone_refinement, volumes and surfaces should be class which has the:
    1. __getitem__ to allow [] access
    2. __setitem__ to allow [] assignment
    3. by_name(pattern:str) to use regexpr/glob to select all zones/surfaces with matched name
    3. by_type(pattern:str) to use regexpr/glob to select all zones/surfaces with matched type
    """
    volumes: Optional[List[VolumeTypes]] = pd.Field()
    surfaces: Optional[List[SurfaceTypes]] = pd.Field()
    """
    Below can be mostly reused with existing models 
    """
    time_stepping: Optional[Union[SteadyTimeStepping, UnsteadyTimeStepping]] = pd.Field()
    user_defined_dynamics: Optional[UserDefinedDynamics] = pd.Field()
    """
    Support for user defined expression?
    If so:
        1. Move over the expression validation functions.
        2. Have camelCase to snake_case naming converter for consistent user experience.
    Limitations:
        1. No per volume zone output. (single volume output)
    """
    outputs: Optional[List[OutputTypes]] = pd.Field()

    def __init__(self, **kwargs):
        pass

    def to_surface_meshing_params(self): ...

    def to_volume_meshing_params(self): ...

    def to_solver_params(self): ...

    def run(self) -> str:
        return "f113d93a-c61a-4438-84af-f760533bbce4"
