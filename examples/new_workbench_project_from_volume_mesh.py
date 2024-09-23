import flow360 as fl
from flow360.component.volume_mesh import VolumeMeshV2

fl.Env.dev.active()

# volume mesh
volume_mesh_draft = VolumeMeshV2.from_file(
    "data/om6wing/mesh.lb8.ugrid",
    project_name="wing-volume-mesh-python-upload",
    solver_version="workbench-24.9.2",
    tags=["python"],
)

volume_mesh = volume_mesh_draft.submit()
print(volume_mesh)
