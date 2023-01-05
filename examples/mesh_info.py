from flow360.component.volume_mesh import VolumeMeshList

meshes = VolumeMeshList()
mesh = meshes[0].to_volume_mesh()

for item in meshes:
    mesh = item.to_volume_mesh()
    print(
        mesh.id,
        "status:",
        mesh.status,
        "| all boundaries:",
        mesh.all_boundaries,
        "| no slip walls:",
        mesh.no_slip_walls,
    )
