import math

import flow360 as fl
from flow360.examples import TutorialDynamicDerivatives

TutorialDynamicDerivatives.get_files()
project = fl.Project.from_geometry(
    TutorialDynamicDerivatives.geometry,
    name="Tutorial Calculating Dynamic Derivatives using Sliding Interfaces from Python",
)
geometry = project.geometry


# show face groupings
geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag("faceName")
geometry.group_edges_by_tag("edgeName")

with fl.SI_unit_system:
    cylinder = fl.Cylinder(
        name="cylinder",
        axis=[0, 1, 0],
        center=[0, 0, 0],
        inner_radius=0,
        outer_radius=1.0,
        height=2.5,
    )
    sliding_interface = fl.RotationCylinder(
        spacing_axial=0.04,
        spacing_radial=0.04,
        spacing_circumferential=0.04,
        entities=cylinder,
        enclosed_entities=geometry["wing"],
    )
    farfield = fl.AutomatedFarfield()
    params = fl.SimulationParams(
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                surface_max_edge_length=0.03 * fl.u.m,
                curvature_resolution_angle=8 * fl.u.deg,
                surface_edge_growth_rate=1.15,
                boundary_layer_first_layer_thickness=1e-6,
                boundary_layer_growth_rate=1.15,
            ),
            refinement_factor=1.0,
            volume_zones=[sliding_interface, farfield],
            refinements=[
                fl.SurfaceEdgeRefinement(
                    name="leadingEdge",
                    method=fl.AngleBasedRefinement(value=1 * fl.u.degree),
                    edges=geometry["leadingEdge"],
                ),
                fl.SurfaceEdgeRefinement(
                    name="trailingEdge",
                    method=fl.HeightBasedRefinement(value=0.001),
                    edges=geometry["trailingEdge"],
                ),
            ],
        ),
        reference_geometry=fl.ReferenceGeometry(
            moment_center=[0, 0, 0],
            moment_length=[1, 1, 1],
            area=2,
        ),
        operating_condition=fl.AerospaceCondition(
            velocity_magnitude=50,
        ),
        time_stepping=fl.Steady(
            max_steps=10000, CFL=fl.RampCFL(initial=1, final=100, ramp_steps=1000)
        ),
        outputs=[
            fl.VolumeOutput(
                output_fields=[
                    "Mach",
                ],
            ),
            fl.SurfaceOutput(
                surfaces=geometry["*"],
                output_fields=[
                    "Cp",
                    "CfVec",
                ],
            ),
        ],
        models=[
            fl.Rotation(
                volumes=cylinder,
                spec=fl.AngularVelocity(0 * fl.u.rad / fl.u.s),
            ),
            fl.Freestream(surfaces=farfield.farfield),
            fl.Wall(surfaces=geometry["wing"]),
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-9,
                    linear_solver=fl.LinearSolver(max_iterations=35),
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    absolute_tolerance=1e-8,
                    linear_solver=fl.LinearSolver(max_iterations=25),
                ),
            ),
        ],
    )

# Run steady case with a fixed sliding interface for initializing the flow field.
project.run_case(
    params=params,
    name="Tutorial Calculating Dynamic Derivatives using Sliding Interfaces (Steady)",
)
parent_case = fl.Case.from_cloud(project.case.id)

# Update the parameters for the unsteady case.
with fl.SI_unit_system:
    params.time_stepping = fl.Unsteady(
        max_pseudo_steps=80,
        steps=400,
        step_size=0.01 * 2.0 * math.pi / 20.0 * fl.u.s,
        CFL=fl.RampCFL(initial=1, final=1e8, ramp_steps=20),
    )
    params.models[0].spec = fl.AngleExpression("0.0349066 * sin(0.05877271 * t)")

# Run unsteady case with an oscillating sliding interface for collecting the data.
project.run_case(
    params=params,
    name="Tutorial Calculating Dynamic Derivatives using Sliding Interfaces (Unsteady)",
    fork_from=parent_case,
)
