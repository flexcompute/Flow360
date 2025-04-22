import json

import flow360 as fl
from flow360.examples import TutorialBETDisk

TutorialBETDisk.get_files()

project = fl.Project.from_geometry(TutorialBETDisk.geometry, name="Tutorial BETDisk from Python")
geometry = project.geometry

# show face and edge groupings
geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag("faceName")
geometry.group_edges_by_tag("edgeName")


bet = fl.BETDisk.from_file(TutorialBETDisk.extra["disk0"])

with fl.SI_unit_system:
    cylinder1 = fl.Cylinder(
        name="cylinder1",
        axis=[1, 0, 0],
        center=[-2.0, 5.0, 0],
        outer_radius=4.0,
        inner_radius=0,
        height=0.6,
    )
    cylinder2 = fl.Cylinder(
        name="cylinder2",
        axis=[1, 0, 0],
        center=[0, 5, 0],
        outer_radius=4.1,
        inner_radius=0,
        height=5,
    )
    farfield = fl.AutomatedFarfield()
    params = fl.SimulationParams(
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                surface_edge_growth_rate=1.2,
                surface_max_edge_length=0.5,
                curvature_resolution_angle=30 * fl.u.deg,
                boundary_layer_growth_rate=1.15,
                boundary_layer_first_layer_thickness=1e-06,
            ),
            volume_zones=[farfield],
            refinements=[
                fl.AxisymmetricRefinement(
                    name="BET_Disk",
                    spacing_axial=0.02,
                    spacing_radial=0.03,
                    spacing_circumferential=0.06,
                    entities=cylinder1,
                ),
                fl.UniformRefinement(name="cylinder_refinement", spacing=0.1, entities=[cylinder2]),
                fl.SurfaceRefinement(
                    name="tip",
                    max_edge_length=0.01,
                    faces=[
                        geometry["tip"],
                    ],
                ),
                fl.SurfaceEdgeRefinement(
                    name="aniso",
                    method=fl.HeightBasedRefinement(value=0.0003),
                    edges=[
                        geometry["wingTrailingEdge"],
                        geometry["wingLeadingEdge"],
                    ],
                ),
                fl.SurfaceEdgeRefinement(
                    name="projectAnisoSpacing",
                    method=fl.ProjectAnisoSpacing(),
                    edges=[
                        geometry["rootAirfoilEdge"],
                        geometry["tipAirfoilEdge"],
                    ],
                ),
            ],
        ),
        reference_geometry=fl.ReferenceGeometry(
            moment_center=[0.375, 0, 0],
            moment_length=[1.26666666, 1.26666666, 1.26666666],
            area=12.5,
        ),
        operating_condition=fl.AerospaceCondition.from_mach(
            mach=0.182,
            alpha=5 * fl.u.deg,
            reference_mach=0.54,
        ),
        time_stepping=fl.Steady(
            max_steps=10000, CFL=fl.RampCFL(initial=1, final=100, ramp_steps=2000)
        ),
        outputs=[
            fl.VolumeOutput(
                output_fields=[
                    "primitiveVars",
                    "betMetrics",
                    "qcriterion",
                ],
            ),
            fl.SurfaceOutput(
                surfaces=geometry["*"],
                output_fields=[
                    "primitiveVars",
                    "Cp",
                    "Cf",
                    "CfVec",
                ],
            ),
        ],
        models=[
            fl.Wall(
                surfaces=[
                    geometry["wing"],
                    geometry["tip"],
                ],
            ),
            fl.Freestream(surfaces=farfield.farfield),
            fl.SlipWall(surfaces=farfield.symmetry_planes),
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-12,
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    absolute_tolerance=1e-10,
                    update_jacobian_frequency=1,
                    equation_evaluation_frequency=1,
                ),
            ),
            bet,
        ],
    )


project.run_case(params=params, name="Case of tutorial BETDisk from Python")
