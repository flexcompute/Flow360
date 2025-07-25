import flow360 as fl
from flow360.examples import IsolatedPropeller
from flow360.user_config import UserConfig
UserConfig.set_profile("demo")

IsolatedPropeller.get_files()

project = fl.Project.from_geometry(IsolatedPropeller.geometry, name="time-average-isosurfaces")

geometry = project.geometry
geometry.group_edges_by_tag("edgeId")
geometry.group_faces_by_tag("faceName")


with fl.SI_unit_system:
    rotating_cylinder = fl.Cylinder(
        name="Rotating zone", center=[0, 0, 0], axis=[1, 0, 0], outer_radius=2, height=0.8
    )
    refinement_cylinder = fl.Cylinder(
        name="Refinement zone", center=[1.9, 0, 0], axis=[1, 0, 0], outer_radius=2, height=4
    )
    slice = fl.Slice(name="Slice", normal=[1, 0, 0], origin=[0.6, 0, 0])
    volume_zone_rotating_cylinder = fl.RotationCylinder(
        name="Rotation cylinder",
        spacing_axial=0.05,
        spacing_radial=0.05,
        spacing_circumferential=0.05,
        entities=[rotating_cylinder],
        enclosed_entities=[geometry["*"]],
    )
    farfield = fl.AutomatedFarfield(name="Farfield")
    params = fl.SimulationParams(
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                surface_max_edge_length=1, boundary_layer_first_layer_thickness=0.1 * fl.u.mm
            ),
            refinements=[
                fl.UniformRefinement(
                    name="Uniform refinement", spacing=0.025, entities=[refinement_cylinder]
                )
            ],
            volume_zones=[farfield, volume_zone_rotating_cylinder],
        ),
        reference_geometry=fl.ReferenceGeometry(
            area=1, moment_center=[0, 0, 0], moment_length=[1, 1, 1]
        ),
        operating_condition=fl.AerospaceCondition(velocity_magnitude=20),
        models=[
            fl.Wall(name="Wall", surfaces=[geometry["*"]]),
            fl.Freestream(name="Freestream", surfaces=[farfield.farfield]),
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(relative_tolerance=0.01),
                turbulence_model_solver=fl.SpalartAllmaras(
                    relative_tolerance=0.01, hybrid_model=fl.DetachedEddySimulation()
                ),
            ),
            fl.Rotation(
                name="Rotation",
                volumes=[rotating_cylinder],
                spec=fl.AngularVelocity(value=200 * fl.u.rpm),
            ),
        ],
        time_stepping=fl.Unsteady(
            steps=600,
            step_size=0.0025,
            max_pseudo_steps=35,
            CFL=fl.AdaptiveCFL(
                min=0.1, max=10000, max_relative_change=1, convergence_limiting_factor=0.5
            ),
        ),
        outputs=[
            fl.SurfaceOutput(
                name="Surface output",
                output_fields=["Cp", "yPlus", "Cf", "CfVec"],
                surfaces=[geometry["*"]],
            ),
            fl.SliceOutput(
                name="Slice output",
                output_fields=["qcriterion", "vorticity", "Mach"],
                slices=[slice],
            ),
            fl.TimeAverageIsosurfaceOutput(
                isosurfaces=[fl.Isosurface(
                        name='q_criterion_avg',
                        field='qcriterion',
                        iso_value=0.0004128
                    )],
                start_step=420,
                output_fields=['velocity_magnitude']
            )
        ],
    )


project.run_case(params, name="Isolated propeller case")
