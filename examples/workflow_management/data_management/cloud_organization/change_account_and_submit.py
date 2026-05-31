import flow360 as fl
from flow360.examples import OM6wing

# choose shared account interactively
fl.Accounts.choose_shared_account()

# retrieve mesh files
OM6wing.get_files()

# submit mesh
project = fl.Project.from_volume_mesh(
    OM6wing.mesh_filename,
    name="Account change mesh upload from Python",
)

# leave the account
fl.Accounts.leave_shared_account()
