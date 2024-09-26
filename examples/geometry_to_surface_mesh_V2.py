import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.component.simulation import cloud
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    SurfaceRefinement,
)
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.examples import Airplane
from flow360 import log

fl.Env.dev.active()

log.set_logging_level("DEBUG")

geometry_draft = Geometry.from_file(
    Airplane.geometry, solver_version="workbench-24.9.2"
)
geometry = geometry_draft.submit()
# geometry = Geometry.from_cloud("geo-6e45ad20-74be-48d2-9092-d4de968df4f8")
geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag(tag_name="groupName")
with SI_unit_system:
    params = SimulationParams(
        meshing=MeshingParams(
            defaults=MeshingDefaults(
                boundary_layer_first_layer_thickness=1,
                surface_edge_growth_rate=1.41,
                surface_max_edge_length=1.1111,
            ),
            volume_zones=[AutomatedFarfield()],
        ),
    )
cloud.generate_surface_mesh(geometry, params=params, draft_name="TestGrouping", async_mode=True)
# print(geometry._meta_class)
