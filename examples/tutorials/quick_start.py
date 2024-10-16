import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.component.simulation import cloud

from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.models.surface_models import Freestream, Wall
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.primitives import ReferenceGeometry
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady
from flow360.component.simulation.outputs.outputs import SurfaceOutput
from flow360.component.simulation.unit_system import SI_unit_system, u
from flow360.examples import Airplane

fl.Env.preprod.active()

SOLVER_VERSION = "workbench-24.9.3"

geometry_draft = Geometry.from_file(Airplane.geometry, project_name='Simple Airplane from Python', solver_version=SOLVER_VERSION)
geometry = geometry_draft.submit()
# you can use this if geometry was submitted eariler:
# geometry = Geometry(id=<provide-geo-id>)

geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag("groupName")


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
        outputs=[SurfaceOutput(surfaces=geometry["*"], output_fields=['Cp', 'Cf', 'yPlus', 'CfVec'])]
    )

case = cloud.run_case(geometry, params=params, draft_name="Case of Simple Airplane from Python", async_mode=True)

