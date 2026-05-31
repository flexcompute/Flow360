import flow360 as fl
from flow360.examples import Cylinder2D

Cylinder2D.get_files()


project = fl.Project.from_volume_mesh(
    Cylinder2D.mesh_filename, name="Unsteady 2D Cylinder from Python"
)

vm = project.volume_mesh


with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=20, moment_center=[0, 0, 0], moment_length=[1, 1, 1]
        ),
        operating_condition=fl.AerospaceCondition.from_mach_reynolds(
            reynolds_mesh_unit=50, mach=0.1, project_length_unit=fl.u.m
        ),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-9, linear_solver=fl.LinearSolver(max_iterations=25)
                ),
                turbulence_model_solver=fl.NoneSolver(),
            ),
            fl.Wall(surfaces=[vm["fluid/wall"]]),
            fl.Freestream(surfaces=[vm["fluid/farfield"]]),
            fl.SlipWall(surfaces=[vm["fluid/periodic_0_l"], vm["fluid/periodic_0_r"]]),
        ],
        time_stepping=fl.Unsteady(
            max_pseudo_steps=40,
            step_size=2,
            steps=20,
        ),
        outputs=[
            fl.SurfaceOutput(output_fields=["Cp"], surfaces=[vm["*"]]),
            fl.VolumeOutput(
                output_fields=[
                    "primitiveVars",
                    "vorticity",
                    "residualNavierStokes",
                    "T",
                    "Cp",
                    "mut",
                ],
            ),
        ],
    )

project.run_case(params, "Unsteady 2D Cylinder case from Python")
