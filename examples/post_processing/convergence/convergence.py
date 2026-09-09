from pylab import show

import flow360 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

project = fl.Project.from_volume_mesh(
    OM6wing.mesh_filename,
    name="Convergence results from Python",
)

vm = project.volume_mesh

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=1.15315, moment_center=[0, 0, 0], moment_length=[1.47602, 0.801672, 1.47602]
        ),
        operating_condition=fl.AerospaceCondition(velocity_magnitude=286, alpha=3.06 * fl.u.deg),
        time_stepping=fl.Steady(
            max_steps=5000, CFL=fl.RampCFL(initial=1, final=100, ramp_steps=1000)
        ),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-9, linear_solver=fl.LinearSolver(max_iterations=35)
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    absolute_tolerance=1e-8, linear_solver=fl.LinearSolver(max_iterations=25)
                ),
            ),
            fl.Wall(surfaces=vm["2"]),
            fl.SlipWall(surfaces=vm["1"]),
            fl.Freestream(surfaces=vm["3"]),
        ],
        outputs=[
            fl.SurfaceOutput(surfaces=vm["1"], output_fields=["Cp", "CfVec"]),
            fl.VolumeOutput(output_fields=["Cp", "Mach", "qcriterion"]),
        ],
    )

case = project.run_case(params, "Convergence case from Python")


# wait until the case finishes execution
case.wait()

results = case.results

# nonlinear residuals contain convergence information
nonlinear_residuals = results.nonlinear_residuals.as_dataframe()
print(nonlinear_residuals)

nonlinear_residuals.plot(
    x="pseudo_step",
    y=["0_cont", "1_momx", "2_momy", "3_momz", "4_energ", "5_nuHat"],
    xlim=(0, None),
    xlabel="Pseudo Step",
    secondary_y=["5_nuHat"],
    figsize=(10, 7),
    title="Nonlinear residuals",
)

max_residual_location = results.max_residual_location.as_dataframe()
print(max_residual_location)

max_residual_location.plot(
    x="pseudo_step",
    y=[
        "max_cont_res",
        "max_momx_res",
        "max_momy_res",
        "max_momz_res",
        "max_energ_res",
        "max_nuHat_res",
    ],
    xlabel="Pseudo Step",
    xlim=(0, None),
    ylim=(-25, None),
    secondary_y=["max_nuHat_res"],
    figsize=(10, 7),
    title="Max residual location",
)
show()

cfl = results.cfl.as_dataframe()
print(cfl)

cfl.plot(
    x="pseudo_step",
    y=["0_NavierStokes_cfl", "1_SpalartAllmaras_cfl"],
    xlim=(0, None),
    xlabel="Pseudo Step",
    figsize=(10, 7),
    title="CFL",
)
show()

results.set_destination(use_case_name=True)
results.download(
    nonlinear_residuals=True,
    linear_residuals=True,
    cfl=True,
    minmax_state=True,
    max_residual_location=True,
)
