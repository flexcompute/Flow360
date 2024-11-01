from flow360.component.volume_mesh import VolumeMesh
from flow360.component.case import Case
from flow360.examples import OM6wing
from flow360.component.v1.flow360_params import Flow360Params

OM6wing.get_files()

# submit mesh
volume_mesh = VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh")
# volume_mesh = volume_mesh.submit()
print(volume_mesh)

# # submit case using json file
params = Flow360Params(OM6wing.case_json)
case = Case.create("OM6wing", params, volume_mesh.id)
case = case.submit()
print(case)
