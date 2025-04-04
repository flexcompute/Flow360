import flow360 as fl
from flow360.examples import Tutorial2DCRM

Tutorial2DCRM.get_files()

project = fl.Project.from_geometry(Tutorial2DCRM.geometry, name="Tutorial 2D CRM from Python")
geometry = project.geometry

# show face and edge groupings
geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag("faceName")
geometry.group_edges_by_tag("edgeName")


with fl.SI_unit_system:
    cylinders = [
        fl.Cylinder(
            name=f"cylinder{i+1}",
            axis=[0, 1, 0],
            center=[0.7, 0.5, 0],
            outer_radius=outer_radius,
            height=1.0,
        )
        for i, outer_radius in enumerate([1.1, 2.2, 3.3, 4.5])
    ]
    cylinder5 = fl.Cylinder(
        name="cylinder5", axis=[-1, 0, 0], center=[6.5, 0.5, 0], outer_radius=6.5, height=10
    )
    farfield = fl.AutomatedFarfield(name="farfield", method="quasi-3d")
    params = fl.SimulationParams(
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                surface_edge_growth_rate=1.17,
                surface_max_edge_length=1.1,
                curvature_resolution_angle=12 * fl.u.deg,
                boundary_layer_growth_rate=1.17,
                boundary_layer_first_layer_thickness=1.8487111e-06,
            ),
            refinement_factor=1.35,
            gap_treatment_strength=0.5,
            volume_zones=[farfield],
            refinements=[
                fl.UniformRefinement(name="refinement1", spacing=0.1, entities=[cylinders[0]]),
                fl.UniformRefinement(name="refinement2", spacing=0.15, entities=[cylinders[1]]),
                fl.UniformRefinement(name="refinement3", spacing=0.225, entities=[cylinders[2]]),
                fl.UniformRefinement(name="refinement4", spacing=0.275, entities=[cylinders[3]]),
                fl.UniformRefinement(name="refinement5", spacing=0.325, entities=[cylinder5]),
                fl.SurfaceRefinement(name="wing", max_edge_length=0.74, faces=[geometry["wing"]]),
                fl.SurfaceRefinement(
                    name="flap-slat",
                    max_edge_length=0.55,
                    faces=[geometry["flap"], geometry["slat"]],
                ),
                fl.SurfaceRefinement(
                    name="trailing",
                    max_edge_length=0.36,
                    faces=[
                        geometry["*Trailing"],
                    ],
                ),
                fl.SurfaceEdgeRefinement(
                    name="edges",
                    method=fl.HeightBasedRefinement(value=0.0007),
                    edges=[
                        geometry["*trailingEdge"],
                        geometry["*leadingEdge"],
                    ],
                ),
                fl.SurfaceEdgeRefinement(
                    name="symmetry", method=fl.ProjectAnisoSpacing(), edges=[geometry["symmetry"]]
                ),
            ],
        ),
        reference_geometry=fl.ReferenceGeometry(
            moment_center=[0.25, 0, 0], moment_length=[1, 1, 1], area=0.01
        ),
        operating_condition=fl.operating_condition_from_mach_reynolds(
            mach=0.2,
            reynolds=5e6,
            project_length_unit=1 * fl.u.m,
            temperature=272.1,
            alpha=16 * fl.u.deg,
            beta=0 * fl.u.deg,
        ),
        time_stepping=fl.Steady(
            max_steps=3000, CFL=fl.RampCFL(initial=20, final=300, ramp_steps=500)
        ),
        models=[
            fl.Wall(
                surfaces=[
                    geometry["*"],
                ],
                name="wall",
            ),
            fl.Freestream(surfaces=farfield.farfield, name="Freestream"),
            fl.SlipWall(surfaces=farfield.symmetry_planes, name="slipwall"),
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-11,
                    linear_solver=fl.LinearSolver(max_iterations=35),
                    kappa_MUSCL=0.33,
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    absolute_tolerance=1e-10,
                    linear_solver=fl.LinearSolver(max_iterations=25),
                    equation_evaluation_frequency=1,
                ),
            ),
        ],
        outputs=[
            fl.VolumeOutput(
                name="fl.VolumeOutput",
                output_fields=[
                    "primitiveVars",
                    "vorticity",
                    "residualNavierStokes",
                    "residualTurbulence",
                    "Cp",
                    "Mach",
                    "qcriterion",
                    "mut",
                ],
            ),
            fl.SurfaceOutput(
                name="fl.SurfaceOutput",
                surfaces=geometry["*"],
                output_fields=["primitiveVars", "Cp", "Cf", "CfVec", "yPlus"],
            ),
        ],
    )


project.run_case(params=params, name="Case of tutorial 2D CRM from Python")
