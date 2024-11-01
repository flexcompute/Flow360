import flow360.component.v1.modules as fl

meshes = fl.MyVolumeMeshes()
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
