import time

import flow360 as fl
from flow360.component.simulation.primitives import GenericVolume, Surface
from flow360.component.volume_mesh import VolumeMeshV2
from flow360.examples import OM6wing

fl.Env.dev.active()

# Creating and uploading a volume mesh from file
volume_mesh_draft = VolumeMeshV2.from_file(
    OM6wing.mesh_filename,
    project_name="wing-volume-mesh-python-upload",
    solver_version="workbench-24.9.2",
    tags=["python"],
)

volume_mesh = volume_mesh_draft.submit()

# Going to sleep after upload, let pipelines finish...
time.sleep(10)

# Loading volume mesh from cloud
volume_mesh = VolumeMeshV2.from_cloud(volume_mesh.id)

print(volume_mesh.boundary_names)
print(volume_mesh.zone_names)

entities = volume_mesh["1"]

print(entities[0])

assert isinstance(entities[0], Surface)
assert isinstance(entities[1], GenericVolume)
