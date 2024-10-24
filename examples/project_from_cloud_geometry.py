from flow360.component.project import Project
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.models.surface_models import Freestream, Wall
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput
from flow360.component.simulation.primitives import ReferenceGeometry
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady
from flow360.component.simulation.unit_system import SI_unit_system, u
from flow360.environment import dev

dev.active()

project = Project.from_cloud("prj-f3569ba5-16a3-4e41-bfd2-b8840df79835")
print(project.get_simulation_json())

geometry = project.geometry
geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag("faceId")

with SI_unit_system:
    params = SimulationParams(
        meshing=MeshingParams(
            defaults=MeshingDefaults(
                boundary_layer_first_layer_thickness=0.001, surface_max_edge_length=1
            ),
            volume_zones=[AutomatedFarfield()],
        ),
        reference_geometry=ReferenceGeometry(),
        operating_condition=AerospaceCondition(velocity_magnitude=100, alpha=5 * u.deg),
        time_stepping=Steady(max_steps=1000),
        models=[
            Wall(
                surfaces=[geometry["*"]],
                name="Wall",
            ),
            Freestream(surfaces=[AutomatedFarfield().farfield], name="Freestream"),
        ],
        outputs=[
            SurfaceOutput(surfaces=geometry["*"], output_fields=["Cp", "Cf", "yPlus", "CfVec"])
        ],
    )

project.run_surface_mesher(params=params, draft_name="Case of Simple Airplane from Python")
