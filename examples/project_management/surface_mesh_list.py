import flow360.v1 as fl
from flow360.log import log

for surface_mesh in fl.MySurfaceMeshes(limit=1000):
    log.info(
        surface_mesh.short_description() + "solver_version = " + str(surface_mesh.solver_version)
    )
