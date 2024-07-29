import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    SurfaceRefinement,
)
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.examples import Airplane

fl.Env.dev.active()

geometry_draft = Geometry.from_file(
    Airplane.geometry,
    name="testing-airplane-csm-geometry",
)
geometry = geometry_draft.submit(solver_version="workbench-24.7")
with SI_unit_system:
    params = SimulationParams(
        meshing=MeshingParams(
            refinements=[
                SurfaceRefinement(max_edge_length=0.8),
                # BoundaryLayer(first_layer_thickness=0.8),
            ],
            volume_zones=[AutomatedFarfield()],
        ),
    )
# geometry.generate_surface_mesh(params=params, async_mode=False)
geometry.generate_volume_mesh(params=params, async_mode=False)
