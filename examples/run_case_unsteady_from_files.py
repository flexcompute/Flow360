import flow360 as fl
from flow360.examples import Cylinder

Cylinder.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(Cylinder.mesh_filename, name="cylinder-mesh")
volume_mesh = volume_mesh.submit()
print(volume_mesh)

# # submit case using json file
params = fl.Flow360Params(Cylinder.case_json)
case = fl.Case.create("cylinder-Re100", params, volume_mesh.id)
case = case.submit()
print(case)
