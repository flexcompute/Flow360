import flow360.v1 as fl
from flow360.examples import OM6wing

fl.Env.dev.active()

OM6wing.get_files()

project = fl.Project.from_volume_mesh(OM6wing.mesh_filename, name="OM6wing Quick Start from Python")

# # submit case using json file
params = fl.Flow360Params(OM6wing.case_json)
case = fl.Case.create("OM6wing", params, volume_mesh.id, solver_version="release-23.1.1.0")
case = case.submit()
print(case)

case2 = case.retry(name="OM6wing-adaptive-CFL", solver_version="release-23.2.1.0")
case2.params.time_stepping.CFL = fl.AdaptiveCFL()
case2.params.time_stepping.max_pseudo_steps = 1000
case2 = case2.submit()
print(case2)
