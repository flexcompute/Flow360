import flow360 as fl
from flow360.examples import OM6wing

# select the company shared workspace
shared = fl.Workspace.get_shared()

# retrieve mesh files
OM6wing.get_files()

# submit the mesh directly into the shared workspace
project = fl.Project.from_volume_mesh(
    OM6wing.mesh_filename,
    name="Shared workspace mesh upload from Python",
    folder_or_workspace=shared,
)

# a folder created in the shared workspace works as a target too
shared_folder = fl.Folder.create(
    "Team meshes", parent_folder_or_workspace=shared
).submit()
project_in_folder = fl.Project.from_volume_mesh(
    OM6wing.mesh_filename,
    name="Shared workspace mesh upload into a folder",
    folder_or_workspace=shared_folder,
)
