import flow360.v1 as fl
from flow360.log import log

meshes = fl.MyVolumeMeshes(limit=1000)

for volume_mesh in meshes:
    log.info(
        volume_mesh.short_description() + "solver_version = " + str(volume_mesh.solver_version)
    )
