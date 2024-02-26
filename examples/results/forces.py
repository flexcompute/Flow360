import flow360 as fl
from flow360.examples import Forces

Forces.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(Forces.mesh_filename, name="Forces-mesh")
volume_mesh = volume_mesh.submit()

# submit case using json file
params = fl.Flow360Params(Forces.case_json)
case = volume_mesh.create_case("Forces-example", params)
case = case.submit()

# wait until the case finishes execution
case.wait()

results = case.results

print(results.total_forces.as_dataframe())

print(results.surface_forces.as_dataframe())

print(results.force_distribution.as_dataframe())
