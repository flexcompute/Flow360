import os
from typing import List

from pylab import plot, show, xlabel, ylabel

import flow360 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

project = fl.Project.from_volume_mesh(OM6wing.mesh_filename, name="Alpha sweep results from Python")

vm = project.volume_mesh

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=1.15315084119231,
            moment_center=[0, 0, 0],
            moment_length=[1.47602, 0.801672958512342, 1.47602],
        ),
        operating_condition=fl.AerospaceCondition(velocity_magnitude=286, alpha=3.06 * fl.u.deg),
        time_stepping=fl.Steady(max_steps=500),
        models=[
            fl.Wall(surfaces=vm["1"]),
            fl.SlipWall(surfaces=vm["2"]),
            fl.Freestream(surfaces=vm["3"]),
        ],
    )


# ": List[fl.Case]" is just for type hints
case_list: List[fl.Case] = []
alpha_range = range(-6, 15, 2)
for alpha in alpha_range:
    params.operating_condition.alpha = alpha * fl.u.deg
    case = project.run_case(params, f"alpha-sweep-OM6wing-alpha={alpha}")
    case_list.append(case)

project.print_project_tree()

# wait for all cases to finish processing
[case.wait() for case in case_list]


# calculate average using dataframe structure and pandas functions
def average_last_10_percent(df, column):
    last_10_percent = df.tail(int(len(df) * 0.1))
    average = last_10_percent[column].mean()
    return average


CL_list = []
CD_list = []

for case in case_list:
    total_forces = case.results.total_forces

    average_CL = average_last_10_percent(total_forces.as_dataframe(), "CL")
    CL_list.append(average_CL)

    average_CD = average_last_10_percent(total_forces.as_dataframe(), "CD")
    CD_list.append(average_CD)

# download all data:
results_folder = "alpha_sweep_example"
for case in case_list:
    results = case.results
    results.download(
        total_forces=True,
        nonlinear_residuals=True,
        destination=os.path.join(results_folder, case.name),
    )

# plot CL / CD
plot(CD_list, CL_list)
xlabel("CD")
ylabel("CL")
show()
