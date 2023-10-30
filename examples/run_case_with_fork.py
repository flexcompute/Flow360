import flow360 as fl
from flow360.examples import OM6wing
from flow360 import log

OM6wing.get_files()
fl.Env.dev.active()
log.set_logging_level('DEBUG')

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh", tags=["tag"])
volume_mesh = volume_mesh.submit()
print(volume_mesh)

volume_mesh2 = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh2", tags=["tag"])
volume_mesh2 = volume_mesh2.submit()
print(volume_mesh2)


# submit case using json file
params = fl.Flow360Params(OM6wing.case_json)
case = fl.Case.create("OM6wing", params, volume_mesh.id, solver_version='release-23.3.2.0')
case = case.submit()

# fork a case
case_fork_1 = case.fork(interpolate_on_mesh_id=volume_mesh2.id)
case_fork_1.name = "OM6wing-fork-1"
case_fork_1.params.time_stepping.max_pseudo_steps = 300
case_fork_1 = case_fork_1.submit()

# # create another fork
# case_fork_2 = case_fork_1.fork("OM6wing-fork-2")
# case_fork_2.params.time_stepping.max_pseudo_steps = 200

# case_fork_1 = case_fork_1.submit()
# case_fork_2 = case_fork_2.submit()

# # create fork by providing parent case id:
# case_fork = fl.Case.create("case-fork", case.params, parent_id=case.id)
# case_fork.submit()
