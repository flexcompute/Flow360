import flow360 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

# submit mesh using mesh JSON from file
volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh")
volume_mesh = volume_mesh.submit()
print(volume_mesh)

params = fl.Flow360Params(OM6wing.case_yaml)
case = volume_mesh.new_case("om6wing-from-yaml", params)
case = case.submit()
print(case)
