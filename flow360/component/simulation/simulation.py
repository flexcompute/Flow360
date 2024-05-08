from typing import List, Optional, Union

import pydantic as pd

from flow360.component.case import Case

## Warning: pydantic V1
from flow360.component.flow360_params.unit_system import (
    UnitSystemType,
    unit_system_manager,
)
from flow360.component.simulation.inputs import Geometry, GeometryDraft
from flow360.component.simulation.references import ReferenceGeometry
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.surface_mesh import SurfaceMesh, SurfaceMeshDraft
from flow360.component.volume_mesh import VolumeMesh, VolumeMeshDraft


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

    Limitations:
        Sovler capability:
            - Cannot specify multiple reference_geometry/operating_condition in volumes.
    """

    # Resources
    geometry: Optional[Geometry] = pd.Field(default=None)
    surface_mesh: Optional[SurfaceMesh] = pd.Field(default=None)
    volume_mesh: Optional[VolumeMesh] = pd.Field(default=None)
    case: Optional[Case] = pd.Field(default=None)

    def __init__(
        self,
        geometry: Union[Geometry, GeometryDraft],
        surface_mesh: Union[SurfaceMesh, SurfaceMeshDraft],
        volume_mesh: Union[VolumeMesh, VolumeMeshDraft],
        **kwargs,
    ):  # Ref: _init_with_context
        if volume_mesh is not None:
            if isinstance(volume_mesh, VolumeMeshDraft):
                self.volume_mesh = volume_mesh.submit()
            elif 

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
        volume_mesh_draft = VolumeMesh.from_file()
        self.volume_mesh.volume_mesh_draft.submit()
        self.case = Case(
            name="Simulation_om6wing",
            params=self._get_simulation_params(),
            volume_mesh_id=self.volume_mesh.volume_mesh_draft.id,
            solver_version="TestUserJSON-24.3.0",
        )
        self.case.case_draft.submit()
        return self.case.case_draft.id
