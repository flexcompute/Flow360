import flow360.component.v1 as fl
from flow360.component.project import Project
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    UniformRefinement,
)
from flow360.component.simulation.models.surface_models import Freestream, Wall
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput
from flow360.component.simulation.primitives import Box, ReferenceGeometry
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady
from flow360.component.simulation.unit_system import SI_unit_system, u
from flow360.examples import Airplane

fl.Env.dev.active()

project = Project.from_file(
    Airplane.geometry, name="Python Project (Geometry, from file, multiple runs)"
)

geometry = project.geometry
geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag("groupName")

with SI_unit_system:
    params = SimulationParams(
        meshing=MeshingParams(
            defaults=MeshingDefaults(
                boundary_layer_first_layer_thickness=0.001, surface_max_edge_length=1
            ),
            volume_zones=[AutomatedFarfield()],
            refinements=[
                UniformRefinement(
                    entities=[
                        Box.from_principal_axes(
                            name="MyBox",
                            center=(0, 1, 2),
                            size=(4, 5, 6),
                            axes=((2, 2, 0), (-2, 2, 0)),
                        ),
                    ],
                    spacing=1.5,
                ),
            ],
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

# Run the mesher once
project.generate_surface_mesh(params=params, name="Surface mesh 1")
surface_mesh_1 = project.surface_mesh

# Tweak some parameter in the params
params.meshing.defaults.surface_max_edge_length = 2 * u.m

# Run the mesher again
project.generate_surface_mesh(params=params, name="Surface mesh 2")
surface_mesh_2 = project.surface_mesh

assert surface_mesh_1.id != surface_mesh_2.id

# Check available surface mesh IDs in the project
ids = project.get_cached_surface_meshes()
print(ids)
