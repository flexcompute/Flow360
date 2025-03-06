import flow360.v1 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh")
volume_mesh = volume_mesh.submit()
print(volume_mesh)

project = fl.Project.from_volume_mesh(OM6wing.mesh_filename, name="OM6Wing yaml input from Python")

params = fl.SimulationParams.from_file("data/om6wing_params.json")

project.run_case(params, name="OM6Wing yaml input case from Python")
