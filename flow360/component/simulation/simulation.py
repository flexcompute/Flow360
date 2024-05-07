from typing import List, Optional, Union

import pydantic as pd

from flow360.component_v2.case import Case

## Warning: pydantic V1
from flow360.component.flow360_params.unit_system import (
    UnitSystemType,
    unit_system_manager,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.inputs import Geometry
from flow360.component.simulation.mesh import MeshingParameters
from flow360.component.simulation.operating_condition import OperatingConditionTypes
from flow360.component.simulation.outputs import OutputTypes
from flow360.component.simulation.references import ReferenceGeometry
from flow360.component.simulation.starting_points.volume_mesh import VolumeMesh
from flow360.component.simulation.surfaces import SurfaceTypes
from flow360.component.simulation.time_stepping import (
    SteadyTimeStepping,
    UnsteadyTimeStepping,
)
from flow360.component.simulation.volumes import VolumeTypes
from flow360.component.surface_mesh import SurfaceMesh
from flow360.error_messages import use_unit_system_msg
from flow360.exceptions import Flow360ConfigError, Flow360RuntimeError
from flow360.log import log
from flow360.user_config import UserConfig


class Simulation(SimulationParams):
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

    # Resources
    geometry: Optional[Geometry] = pd.Field(default=None)
    surface_mesh: Optional[SurfaceMesh] = pd.Field(default=None)
    volume_mesh: Optional[VolumeMesh] = pd.Field(default=None)
    case: Optional[Case] = pd.Field(default=None)

    def __init__(self, **kwargs):  # Ref: _init_with_context
        # self.unit_system = unit_system_manager.copy_current()

        # if self.unit_system is None:
        # raise Flow360RuntimeError(use_unit_system_msg)

        super().__init__(**kwargs)

    def _get_simulation_params(self):
        return SimulationParams(
            meshing=self.meshing,
            reference_geometry=self.reference_geometry,
            operating_condition=self.operating_condition,
            volumes=self.volumes,
            surfaces=self.surfaces,
            time_stepping=self.time_stepping,
            user_defined_dynamics=self.user_defined_dynamics,
            outputs=self.outputs,
        )

    def run(self) -> str:
        self.volume_mesh.volume_mesh_draft.submit()
        self.case = Case(
            name="Simulation_om6wing",
            params=self._get_simulation_params(),
            volume_mesh_id=self.volume_mesh.volume_mesh_draft.id,
            solver_version="TestUserJSON-24.3.0",
        )
        self.case.case_draft.submit()
        return self.case.case_draft.id
