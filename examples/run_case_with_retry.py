import flow360 as fl
from flow360.examples import OM6wing

OM6wing.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh")
volume_mesh = volume_mesh.submit()
print(volume_mesh)


# submit case using json file
params = fl.Flow360Params(OM6wing.case_json)
case = fl.Case.create("OM6wing", params, volume_mesh.id)
case.submit()

# retry from submited case
case_retry_1 = case.retry()
case_retry_1.params.time_stepping.max_pseudo_steps = 400
case_retry_1.params.boundaries["1"] = fl.WallFunction()
case_retry_1.name = "OM6wing-retry-1"

# retry from not yet submitted case
case_retry_2 = case_retry_1.retry()
case_retry_2.name = "Om6wing-retry-2"
case_retry_2.params.time_stepping.max_pseudo_steps = 300

case_retry_1.submit()
case_retry_2.submit()
