import flow360.component.v1.modules as fl
from flow360.examples import OM6wing

fl.Env.dev.active()

# choose shared account interactively
fl.Accounts.choose_shared_account()

# retrieve mesh files
OM6wing.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh")
volume_mesh = volume_mesh.submit()

# leave the account
fl.Accounts.leave_shared_account()
