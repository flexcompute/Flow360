import flow360 as fl
from flow360.examples import TutorialCHTSolver

fl.Env.preprod.active()

TutorialCHTSolver.get_files()
project = fl.Project.from_file(
    TutorialCHTSolver.mesh_filename, name="Tutorial CHT Solver from Python"
)
volume_mesh = project.volume_mesh

with fl.SI_unit_system:
    thermal_state = fl.ThermalState(
        temperature=288.15 * fl.u.K,
        material=fl.Air(
            dynamic_viscosity=fl.Sutherland(
                reference_temperature=288.15 * fl.u.K,
                reference_viscosity=4.2925193198151646e-8 * fl.u.flow360_viscosity_unit,
                effective_temperature=110.4 * fl.u.K,
            )
        ),
    )
    operating_condition = fl.AerospaceCondition().from_mach(mach=0.1, thermal_state=thermal_state)
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            moment_center=[0, 0, 0] * fl.u.m,
            moment_length=[1, 1, 1] * fl.u.m,
            area=1 * fl.u.m**2,
        ),
        operating_condition=operating_condition,
        time_stepping=fl.Steady(
            max_steps=10000, CFL=fl.RampCFL(initial=1, final=100, ramp_steps=1000)
        ),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-9,
                    linear_solver=fl.LinearSolver(max_iterations=35),
                    order_of_accuracy=2,
                    kappa_MUSCL=-1,
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    absolute_tolerance=1e-8,
                    linear_solver=fl.LinearSolver(max_iterations=25),
                    equation_evaluation_frequency=4,
                    order_of_accuracy=2,
                ),
            ),
            fl.Solid(
                entities=volume_mesh["solid"],
                heat_equation_solver=fl.HeatEquationSolver(
                    absolute_tolerance=1e-11,
                    linear_solver=fl.LinearSolver(
                        max_iterations=25,
                        absolute_tolerance=1e-12,
                    ),
                    equation_evaluation_frequency=10,
                ),
                material=fl.SolidMaterial(
                    name="copper",
                    thermal_conductivity=398 * fl.u.W / (fl.u.m * fl.u.K),
                ),
                volumetric_heat_source=5e3 * fl.u.W / (0.01257 * fl.u.m**3),
            ),
            fl.Wall(
                surfaces=volume_mesh["fluid/centerbody"],
            ),
            fl.Freestream(
                surfaces=volume_mesh["fluid/farfield"],
            ),
            fl.Wall(surfaces=volume_mesh["solid/adiabatic"]),
        ],
        outputs=[
            fl.VolumeOutput(
                name="fl.VolumeOutput",
                output_format="both",
                output_fields=[
                    "primitiveVars",
                    "T",
                    "Cp",
                    "Mach",
                ],
            ),
            fl.SurfaceOutput(
                name="fl.SurfaceOutput",
                surfaces=volume_mesh["*"],
                output_format="both",
                output_fields=["primitiveVars", "T", "Cp", "Cf", "CfVec"],
            ),
        ],
    )

project.run_case(params=params, name="Tutorial CHT Solver from Python")
