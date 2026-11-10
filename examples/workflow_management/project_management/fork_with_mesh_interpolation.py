"""Illustrate interpolate a coarse mesh's simulation to a fine mesh."""

import flow360 as fl
from flow360.examples import Airplane


def get_project_with_refinement_factor(refinement_factor: float, run_case: bool):
    project = fl.Project.from_geometry(
        Airplane.geometry,
        name=f"Airplane with refinement_factor = {refinement_factor}",
    )
    geo = project.geometry
    with fl.SI_unit_system:
        far_field_zone = fl.AutomatedFarfield()
        params = fl.SimulationParams(
            meshing=fl.MeshingParams(
                refinement_factor=refinement_factor,
                defaults=fl.MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                    surface_max_edge_length=1,
                ),
                volume_zones=[far_field_zone],
            ),
            reference_geometry=fl.ReferenceGeometry(),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=100,  # Velocity of 100 m/s
                alpha=5 * fl.u.deg,  # Angle of attack of 5 degrees
            ),
            time_stepping=fl.Steady(max_steps=1000),
            models=[
                fl.Wall(
                    surfaces=[geo["*"]],
                ),
                fl.Freestream(
                    surfaces=[far_field_zone.farfield],
                ),
            ],
            outputs=[
                fl.SurfaceOutput(
                    surfaces=geo["*"],
                    output_fields=[
                        "Cp",
                        "Cf",
                        "yPlus",
                        "CfVec",
                    ],
                )
            ],
        )
    if run_case:
        project.run_case(params=params, name=f"Case-{refinement_factor}")
    else:
        project.generate_volume_mesh(params=params, name=f"VolumeMesh-{refinement_factor}")
    return project


project_coarse_mesh = get_project_with_refinement_factor(refinement_factor=0.5, run_case=True)
project_fine_mesh = get_project_with_refinement_factor(refinement_factor=1.0, run_case=False)

case_with_coarse_mesh = project_coarse_mesh.case

project_coarse_mesh.run_case(
    params=case_with_coarse_mesh.params,
    name="Interpolation fork onto fine mesh",
    fork_from=case_with_coarse_mesh,
    interpolate_to_mesh=project_fine_mesh.volume_mesh,
)
