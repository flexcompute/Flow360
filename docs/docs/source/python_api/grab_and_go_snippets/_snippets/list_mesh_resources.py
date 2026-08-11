import flow360.v1 as fl

for surf_mesh in fl.MySurfaceMeshes(limit=1000):
    print(f"{surf_mesh.short_description()} solver_version = {surf_mesh.solver_version}")

for volume_mesh in fl.MyVolumeMeshes(limit=1000):
    print(f"{volume_mesh.short_description()} solver_version = {volume_mesh.solver_version}")
