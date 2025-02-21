import os
import json

import flow360 as fl
from flow360.examples import BETDisk

BETDisk.get_files()

# project = fl.Project.from_file(
#     files=fl.VolumeMeshFile(BETDisk.mesh_filename),
#     name="BET disk case from Python",
#     length_unit="inch"
# )
project = fl.Project.from_cloud(project_id="prj-d71a1697-7a7c-4765-8131-7500c4a6432e")

vm = project.volume_mesh

bet = json.loads(open(BETDisk.extra["disk0"]).read())

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=16286.016316209487 * fl.u.inch**2,
            moment_center=[450, 0, 0] * fl.u.inch,
            moment_length=[72, 1200, 1200] * fl.u.inch
        ),
        operating_condition=fl.AerospaceCondition.from_mach(mach=0.04),
        time_stepping=fl.Steady(
            max_steps=200,
            CFL=fl.RampCFL(
                initial=1,
                final=200,
                ramp_steps=200
            )
        ),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-11,
                    linear_solver=fl.LinearSolver(max_iterations=35),
                    kappa_MUSCL=0.33
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    absolute_tolerance=1e-10,
                    linear_solver=fl.LinearSolver(max_iterations=25),
                    update_jacobian_frequency=2,
                    equation_evaluation_frequency=1
                )
            ),
            fl.BETDisk(**bet),
            fl.Wall(
                name="NoSlipWall",
                surfaces=vm["fluid/body"]
            ),
            fl.Freestream(
                name="Freestream",
                surfaces=vm["fluid/farfield"]
            )
        ],
        outputs=[
            fl.SurfaceOutput(
                name="SurfaceOutput",
                surfaces=vm["fluid/body"],
                output_fields=["Cp", "Cf", "CfVec", "yPlus", "nodeForcesPerUnitArea"]
            ),
            fl.VolumeOutput(
                name="VolumeOutput",
                output_fields=["primitiveVars", "Mach"]
            )
        ]
    )

case = project.run_case(params, "BET case from Python")


case.wait()


results = case.results
bet_forces_non_dim = results.bet_forces.as_dataframe()
print(results.bet_forces)

# convert results to SI system:
results.bet_forces.to_base("SI")
bet_forces_si = results.bet_forces.as_dataframe()
print(results.bet_forces)

bet_forces_radial_distribution = results.bet_forces_radial_distribution.as_dataframe()
print(results.bet_forces_radial_distribution)

bet_forces_radial_distribution.plot(
    x="Disk0_All_Radius",
    y=["Disk0_Blade0_All_ThrustCoeff", "Disk0_Blade0_All_TorqueCoeff"],
    xlim=(0, 200),
    xlabel="Pseudo Step",
    figsize=(10, 7),
    title="Actuator Disk radial distribution"
)

# download resuts:
results.set_destination(use_case_name=True)
results.download(bet_forces=True, bet_forces_radial_distribution=True, overwrite=True)

# save converted results to a new CSV file:
results.bet_forces.to_file(os.path.join(case.name, "bet_forces_in_SI.csv"))
