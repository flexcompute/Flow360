import pandas as pd

import flow360 as fl
from flow360.examples import Quadcopter

Quadcopter.get_files()

project = fl.Project.from_volume_mesh(
    Quadcopter.mesh_filename, name="Aeroacoustic results from Python"
)

vm = project.volume_mesh

with fl.SI_unit_system:
    rotation_zone_1 = vm["zone_r1"]
    rotation_zone_1.center = [-0.125, 0.125, 0.0055]
    rotation_zone_1.axis = [0, 0, 1]

    rotation_zone_2 = vm["zone_r2"]
    rotation_zone_2.center = [-0.125, -0.125, 0.0055]
    rotation_zone_2.axis = [0, 0, -1]

    rotation_zone_3 = vm["zone_r3"]
    rotation_zone_3.center = [0.125, -0.125, 0.0055]
    rotation_zone_3.axis = [0, 0, 1]

    rotation_zone_4 = vm["zone_r4"]
    rotation_zone_4.center = [0.125, 0.125, 0.0055]
    rotation_zone_4.axis = [0, 0, -1]

    omega = 6000 * fl.u.rpm

    # Time step size will be calculated based on predetermined degrees per time step (3 deg for this run)
    deg_per_time_step_0 = 3.0 * fl.u.deg
    time_step_0 = deg_per_time_step_0 / omega.to("deg/s")

    # Amount of time steps will be adjusted to satisfy the required amount of revolutions (5 rev for this run)
    revolution_time_0 = 360 * fl.u.deg / omega.to("deg/s")
    steps_0 = int(5 * revolution_time_0 / time_step_0)

    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=0.0447726728530549,
            moment_center=[0, 0, 0],
            moment_length=[0.11938, 0.11938, 0.11938],
        ),
        operating_condition=fl.AerospaceCondition.from_mach(
            mach=0,
            thermal_state=fl.ThermalState(temperature=293.15),
            reference_mach=0.21868415800906676,
        ),
        time_stepping=fl.Unsteady(
            step_size=time_step_0,
            steps=steps_0,
        ),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-10,
                    relative_tolerance=0.1,
                    order_of_accuracy=1,
                    linear_solver=fl.LinearSolver(max_iterations=30),
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    absolute_tolerance=1e-8,
                    relative_tolerance=0.1,
                    order_of_accuracy=1,
                    linear_solver=fl.LinearSolver(max_iterations=20),
                ),
            ),
            fl.Wall(
                surfaces=[
                    vm["zone_s/airframe"],
                    vm["zone_r1/blade1"],
                    vm["zone_r2/blade2"],
                    vm["zone_r3/blade3"],
                    vm["zone_r4/blade4"],
                ],
            ),
            fl.Freestream(surfaces=vm["zone_s/farfield"]),
            fl.Rotation(
                name="Rotation",
                volumes=[rotation_zone_1, rotation_zone_2, rotation_zone_3, rotation_zone_4],
                spec=fl.AngularVelocity(value=omega),
            ),
        ],
    )

case = project.run_case(params, "First order run")


case.params.models[0].navier_stokes_solver.order_of_accuracy = 2
case.params.models[0].navier_stokes_solver.linear_solver = fl.LinearSolver(max_iterations=25)

case.params.models[0].turbulence_model_solver.order_of_accuracy = 2
case.params.models[0].turbulence_model_solver.linear_solver = fl.LinearSolver(max_iterations=25)

deg_per_time_step_1 = 0.404496 * fl.u.deg
time_step_1 = deg_per_time_step_1 / omega.to("deg/s")

revolution_time_1 = 360 * fl.u.deg / omega.to("deg/s")
steps_1 = int(5 * revolution_time_1 / time_step_1)

case.params.time_stepping.step_size = time_step_1
case.params.time_stepping.steps = steps_1

case_fork_1 = project.run_case(case.params, "Second order run", fork_from=case)

case_fork_1.params.outputs = [
    fl.AeroAcousticOutput(
        observers=[
            fl.Observer(position=[0, -1.905, 0] * fl.u.m, group_name="1"),
            fl.Observer(position=[0, -1.7599905, -0.72901194] * fl.u.m, group_name="1"),
            fl.Observer(position=[0, -1.3470384, -1.3470384] * fl.u.m, group_name="1"),
            fl.Observer(position=[0.9525, -0.9525, -1.3470384] * fl.u.m, group_name="1"),
            fl.Observer(position=[1.3470384, 0, -1.3470384] * fl.u.m, group_name="1"),
            fl.Observer(position=[0, 0, 1.905] * fl.u.m, group_name="1"),
            fl.Observer(position=[0, -0.37164706, 1.8683959] * fl.u.m, group_name="1"),
            fl.Observer(position=[0, -1.868396, 0.37164707] * fl.u.m, group_name="1"),
            fl.Observer(position=[1.295, 0, -0.767] * fl.u.m, group_name="2"),
        ],
        write_per_surface_output=True,
    )
]

case_fork_2 = project.run_case(case_fork_1.params, "Final run", fork_from=case_fork_1)

case_fork_2.wait()

results = case_fork_2.results

total_acoustics = results.aeroacoustics
print(total_acoustics)

# There are also surface specific aeroacoustic output files
blade_1_acoustics = results.download_file_by_name(
    "results/surface_zone_r1_blade1_acoustics_v3.csv", to_folder="aeroacoustic_results"
)
blade_1_acoustics = pd.read_csv(blade_1_acoustics)
print(blade_1_acoustics)
blade_2_acoustics = results.download_file_by_name(
    "results/surface_zone_r2_blade2_acoustics_v3.csv", to_folder="aeroacoustic_results"
)
blade_2_acoustics = pd.read_csv(blade_2_acoustics)
print(blade_2_acoustics)
blade_3_acoustics = results.download_file_by_name(
    "results/surface_zone_r3_blade3_acoustics_v3.csv", to_folder="aeroacoustic_results"
)
blade_3_acoustics = pd.read_csv(blade_3_acoustics)
print(blade_3_acoustics)
blade_4_acoustics = results.download_file_by_name(
    "results/surface_zone_r4_blade4_acoustics_v3.csv", to_folder="aeroacoustic_results"
)
blade_4_acoustics = pd.read_csv(blade_4_acoustics)
print(blade_4_acoustics)
