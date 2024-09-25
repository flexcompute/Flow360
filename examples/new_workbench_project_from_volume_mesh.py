import time

import flow360 as fl
from flow360.component.volume_mesh import VolumeMeshV2
from flow360.examples import OM6wing

fl.Env.dev.active()

OM6wing.get_files()

print("Creating and uploading a volume mesh from file")
volume_mesh_draft = VolumeMeshV2.from_file(
    OM6wing.mesh_filename,
    project_name="wing-volume-mesh-python-upload",
    solver_version="workbench-24.9.2",
    tags=["python"],
)

volume_mesh = volume_mesh_draft.submit(compress=True)
