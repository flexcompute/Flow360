import flow360 as fl
from flow360.examples import Airplane

project = fl.Project.from_geometry(
    Airplane.geometry,
    name="Python Project (Geometry, from file, multiple runs)",
)

geometry = project.geometry
geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag("groupName")

with fl.SI_unit_system:
    params = fl.SimulationParams(
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                boundary_layer_first_layer_thickness=0.001, surface_max_edge_length=1
            ),
            volume_zones=[fl.AutomatedFarfield()],
            refinements=[
                fl.UniformRefinement(
                    entities=[
                        fl.Box.from_principal_axes(
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
        reference_geometry=fl.ReferenceGeometry(),
        operating_condition=fl.AerospaceCondition(velocity_magnitude=100, alpha=5 * fl.u.deg),
        time_stepping=fl.Steady(max_steps=1000),
        models=[
            fl.Wall(
                surfaces=[geometry["*"]],
            ),
            fl.Freestream(surfaces=[fl.AutomatedFarfield().farfield]),
        ],
        outputs=[
            fl.SurfaceOutput(surfaces=geometry["*"], output_fields=["Cp", "Cf", "yPlus", "CfVec"])
        ],
    )

# Run the mesher once
project.generate_surface_mesh(params=params, name="Surface mesh 1")
surface_mesh_1 = project.surface_mesh

# Tweak some parameter in the params
params.meshing.defaults.surface_max_edge_length = 2 * fl.u.m

# Run the mesher again
project.generate_surface_mesh(params=params, name="Surface mesh 2")
surface_mesh_2 = project.surface_mesh

# Check available surface mesh IDs in the project
ids = project.get_surface_mesh_ids()
print(ids)
