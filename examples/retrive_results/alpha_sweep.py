import os
from typing import List

from pylab import *

import flow360 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

# create a folder
folder = fl.Folder.create("alpha-sweep-example").submit()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh")
volume_mesh = volume_mesh.submit()
print(volume_mesh)

# read case configuration from JSON file:
params = fl.Flow360Params(OM6wing.case_json)

# ": List[fl.Case]" is just for type hints
case_list: List[fl.Case] = []


alpha_range = range(-6, 15, 2)
for alpha in alpha_range:
    params.freestream.alpha = alpha
    case = fl.Case.create(f"alpha-sweep-OM6wing-alpha={alpha}", params, volume_mesh.id)
    case = case.submit()
    case.move_to_folder(folder)
    case_list.append(case)


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
