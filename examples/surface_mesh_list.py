from flow360.component.surface_mesh import SurfaceMeshList as MySurfaceMeshes


my_meshes = MySurfaceMeshes()


for mesh in my_meshes:
    print(mesh.id, mesh.status, mesh.solver_version, mesh.info.created_at, mesh.name)
    print(mesh.params)

mesh = my_meshes[0]
print(mesh)
print(mesh.params)
