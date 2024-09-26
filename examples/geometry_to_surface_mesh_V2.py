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

fl.Env.dev.active()

# geometry_draft = Geometry.from_file(
#     Airplane.geometry, solver_version="workbenchMeshGrouping-24.9.1"
# )
# geometry = geometry_draft.submit()
geometry = Geometry.from_cloud("geo-e89fe565-24a2-4777-b563-2ec1d3d2a133")
# geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag(tag_name="groupName")
with SI_unit_system:
    params = SimulationParams(
        meshing=MeshingParams(
            defaults=MeshingDefaults(
                boundary_layer_first_layer_thickness=1,
                surface_edge_growth_rate=1.4,
                surface_max_edge_length=1.0111,
            ),
            volume_zones=[AutomatedFarfield()],
        ),
    )
cloud.generate_surface_mesh(geometry, params=params, draft_name="TestGrouping", async_mode=False)
# print(geometry._meta_class)
