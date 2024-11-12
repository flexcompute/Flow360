import flow360.v1 as fl
from flow360.examples import Cylinder2D

Cylinder2D.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(Cylinder2D.mesh_filename, name="cylinder2d-mesh")
volume_mesh = volume_mesh.submit()
print(volume_mesh)

# # submit case using json file
params = fl.Flow360Params(Cylinder2D.case_json)
case = fl.Case.create("cylinder2d-Re100", params, volume_mesh.id)
case = case.submit()
print(case)
