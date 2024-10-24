from flow360.component.project import Project
from flow360.component.simulation.meshing_param.params import MeshingParams, MeshingDefaults
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.environment import dev
from flow360.component.simulation.unit_system import SI_unit_system

dev.active()

project = Project.from_cloud("prj-5159c384-fc07-4bd1-b30f-e4ea0dcafa2d")
print(project.get_simulation_json())

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

project.set_default_params(params)
project.run_case()
