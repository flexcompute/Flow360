from flow360 import VolumeMesh, Case
from flow360 import Flow360MeshParams, MeshBoundary, Flow360Params

from testcases import OM6test

OM6test.get_files()

# submit mesh with manual configuration
meshParams = Flow360MeshParams(boundaries=MeshBoundary(noSlipWalls=[1]))
volumeMesh = VolumeMesh.from_file(OM6test.mesh_filename, meshParams, name="OM6wing-mesh")
print(volumeMesh)


# submit case using json file
params = Flow360Params.from_file(OM6test.case_json)
case = Case.new("OM6wing", params, volumeMesh.id)
case.submit()

# fork a case
case_fork_1 = case.fork()
case_fork_1.name = "OM6wing-fork-1"
case_fork_1.params.time_stepping.max_pseudo_steps = 300

# create another fork
case_fork_2 = case_fork_1.fork("OM6wing-fork-2")
case_fork_2.params.time_stepping.max_pseudo_steps = 200

case_fork_1.submit()
case_fork_2.submit()

# create fork by providing parent case id:
case_fork = Case.new("case-fork", case.params, parent_id=case.id)
case_fork.submit()
