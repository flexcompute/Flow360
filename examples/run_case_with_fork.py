import flow360.component.v1xxx as fl
from flow360.examples import OM6wing

OM6wing.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh", tags=["tag"])
volume_mesh = volume_mesh.submit()
print(volume_mesh)


# submit case using json file
params = fl.Flow360Params(OM6wing.case_json)
case = fl.Case.create("OM6wing", params, volume_mesh.id)
case = case.submit()

# fork a case
case_fork_1 = case.fork()
case_fork_1.name = "OM6wing-fork-1"
case_fork_1.params.time_stepping.max_pseudo_steps = 300

# create another fork
case_fork_2 = case_fork_1.fork("OM6wing-fork-2")
case_fork_2.params.time_stepping.max_pseudo_steps = 200

case_fork_1 = case_fork_1.submit()
case_fork_2 = case_fork_2.submit()

# create fork by providing parent case id:
case_fork = fl.Case.create("case-fork", case.params, parent_id=case.id)
case_fork.submit()
