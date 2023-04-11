import flow360 as fl

my_meshes = fl.MySurfaceMeshes()


for mesh in my_meshes:
    print(mesh.id, mesh.status, mesh.solver_version, mesh.info.created_at, mesh.name)
    print(mesh.params)

mesh = my_meshes[0]
print(mesh)
print(mesh.params)
