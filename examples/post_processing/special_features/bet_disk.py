import os

from pylab import show

import flow360 as fl
from flow360.examples import BETExampleData
from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_items import (
    Chart3D,
    FrontCamera,
    Inputs,
    Settings,
    Summary,
)

BETExampleData.get_files()

project = fl.Project.from_volume_mesh(
    BETExampleData.mesh_filename,
    name="BET Disk results from Python",
    length_unit="inch",
)

vm = project.volume_mesh

bet = fl.BETDisk.from_file(BETExampleData.extra["disk0"])

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=16286.016316209487 * fl.u.inch**2,
            moment_center=[450, 0, 0] * fl.u.inch,
            moment_length=[72, 1200, 1200] * fl.u.inch,
        ),
        operating_condition=fl.AerospaceCondition.from_mach(mach=0.04),
        time_stepping=fl.Steady(),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-11,
                    linear_solver=fl.LinearSolver(max_iterations=35),
                    kappa_MUSCL=0.33,
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    absolute_tolerance=1e-10,
                    linear_solver=fl.LinearSolver(max_iterations=25),
                    update_jacobian_frequency=2,
                    equation_evaluation_frequency=1,
                ),
            ),
            bet,
            fl.Wall(surfaces=vm["fluid/body"]),
            fl.Freestream(surfaces=vm["fluid/farfield"]),
        ],
        outputs=[
            fl.SliceOutput(
                slices=[fl.Slice(name="slice_x", normal=(1, 0, 0), origin=(0, 0, 0))],
                output_fields=["betMetrics"],
            )
        ],
    )

case = project.run_case(params, "BET Disk case from Python")


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
    xlim=(0, 150),
    xlabel="Radius",
    figsize=(10, 7),
    title="BET Disk radial distribution",
)
show()

# download resuts:
results.set_destination(use_case_name=True)
results.download(bet_forces=True, bet_forces_radial_distribution=True, overwrite=True)

# save converted results to a new CSV file:
results.bet_forces.to_file(os.path.join(case.name, "bet_forces_in_SI.csv"))

cases = [case]

front_camera_slice = FrontCamera(dimension=350, dimension_dir="height")

bet_slice_screenshot = Chart3D(
    section_title="BET effective AoA",
    items_in_row=2,
    force_new_page=True,
    show="slices",
    include=["slice_x"],
    field="betMetrics_AlphaDegrees",
    limits=(-18, 0),
    camera=front_camera_slice,
    fig_name="slice_x",
)

report = ReportTemplate(
    title="BET results screenshots",
    items=[Summary(), Inputs(), bet_slice_screenshot],
    settings=Settings(dpi=150),
)

report = report.create_in_cloud("BET, dpi=default", cases)

report.wait()
report.download("report.pdf")
