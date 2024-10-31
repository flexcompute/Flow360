import flow360.component.v1 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh")
volume_mesh = volume_mesh.submit()
print(volume_mesh)

# # submit case using json file
params = fl.Flow360Params(OM6wing.case_json)
case = fl.Case.create("OM6wing", params, volume_mesh.id)
case = case.submit()
print(case)
