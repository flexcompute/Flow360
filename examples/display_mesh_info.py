from flow360.component.volume_mesh import VolumeMeshList as MyVolumeMeshes

meshes = MyVolumeMeshes()
mesh = meshes[0]

for mesh in meshes:
    print(
        mesh.id,
        "status:",
        mesh.status,
        "| all boundaries:",
        mesh.all_boundaries,
        "| no slip walls:",
        mesh.no_slip_walls,
    )
