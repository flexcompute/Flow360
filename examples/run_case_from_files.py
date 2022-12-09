from flow360 import VolumeMesh, Case
from flow360 import Flow360MeshParams, Flow360Params

from testcases import OM6test

OM6test.get_files()

# submit mesh using mesh JSON from file
meshParams = Flow360MeshParams.from_file(OM6test.mesh_json)
volumeMesh = VolumeMesh.from_file(OM6test.mesh_filename, meshParams, name="OM6wing-mesh")
print(volumeMesh)


# submit case using json file
params = Flow360Params.from_file(OM6test.case_json)
case = Case.new("OM6wing", params, volumeMesh.id)
case.submit()
print(case)
