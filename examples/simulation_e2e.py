import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.component.simulation import cloud
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    SurfaceRefinement,
)
from flow360.component.simulation.meshing_param.params import MeshingParams, MeshingDefaults
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.models.surface_models import Wall, Freestream
from flow360.component.simulation.primitives import ReferenceGeometry
from flow360.component.simulation.time_stepping.time_stepping import Steady
from flow360.component.simulation.operating_condition.operating_condition import AerospaceCondition
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.examples import Airplane

fl.UserConfig.set_profile("auto_test_1")
fl.Env.dev.active()

SOLVER_VERSION = "workbenchMeshGrouping-24.9.1"

# geometry_draft = Geometry.from_file(Airplane.geometry, solver_version=SOLVER_VERSION)
# geometry = geometry_draft.submit()
# geometry = Geometry(id='geo-745976a8-6c28-4c92-bb04-451b692e2eec')
geometry = Geometry(id="geo-fdd29aea-67ab-49cc-9c30-073eb9e24c77")
geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag("groupName")

print(geometry.info.entity_info)


with SI_unit_system:
    params = SimulationParams(
        meshing=MeshingParams(
            defaults=MeshingDefaults(
                boundary_layer_first_layer_thickness=0.001, surface_max_edge_length=1
            ),
            volume_zones=[AutomatedFarfield()],
        ),
        reference_geometry=ReferenceGeometry(),
        operating_condition=AerospaceCondition(velocity_magnitude=100),
        time_stepping=Steady(max_steps=1000),
        models=[
            Wall(
                surfaces=[geometry["leftWing"], geometry["rightWing"], geometry["fuselage"]],
                name="Wall",
            ),
            Freestream(surfaces=[AutomatedFarfield().farfield], name="Freestream"),
        ],
    )
cloud.run_case(geometry, params=params, draft_name="Testing Grouping", async_mode=True)
# geometry.generate_volume_mesh(params=params, async_mode=False)
# print(geometry._meta_class)
