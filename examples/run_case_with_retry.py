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

# retry from submited case
case_retry_1 = case.retry()
case_retry_1.params.time_stepping.max_pseudo_steps = 400
case_retry_1.name = "OM6wing-retry-1"

# retry from not yet submitted case
case_retry_2 = case_retry_1.retry()
case_retry_2.name = "Om6wing-retry-2"
case_retry_2.params.time_stepping.max_pseudo_steps = 300

case_retry_1.submit()
case_retry_2.submit()
