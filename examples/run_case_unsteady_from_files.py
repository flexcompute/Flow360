from flow360.examples import Cylinder2D
from flow360.component.volume_mesh import VolumeMesh
from flow360.component.v1.flow360_params import Flow360Params
from flow360.component.case import Case

Cylinder2D.get_files()

# submit mesh
volume_mesh = VolumeMesh.from_file(Cylinder2D.mesh_filename, name="cylinder2d-mesh")
volume_mesh = volume_mesh.submit()
print(volume_mesh)

# # submit case using json file
params = Flow360Params(Cylinder2D.case_json)
case = Case.create("cylinder2d-Re100", params, volume_mesh.id)
case = case.submit()
print(case)
