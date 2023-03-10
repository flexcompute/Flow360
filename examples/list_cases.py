import flow360 as fl

# get all cases:
my_cases = fl.MyCases()

# print metadata for the latest case:
print(my_cases[0])

for case in my_cases:
    print(f"id: {case.id}, status: {case.status}, is case deleted: {case.info.deleted}")

# get parameters for the latest case:
case0 = my_cases[0]
print(case0.params)
print(case0.params.freestream.Mach)

# get all meshes:
meshes = fl.MyVolumeMeshes(limit=None, include_deleted=True)

meshes[0].info.surface_mesh_id
for mesh in meshes:
    print(
        f"mesh id: {mesh.id}, status: {mesh.status}, is mesh deleted: {mesh.info.deleted} {mesh.info.surface_mesh_id}"
    )

mesh = meshes[0]
# list of cases for a given mesh
mesh_cases = fl.MyCases(mesh_id=mesh.id, include_deleted=True)
for case in mesh_cases:
    print(f"id: {case.id}, status: {case.status}, is case deleted: {case.info.deleted}")


for mesh in meshes:
    if mesh.info.surface_mesh_id is not None:
        volume_meshes = fl.MyVolumeMeshes(surface_mesh_id=mesh.info.surface_mesh_id)
        print(volume_meshes)
        assert mesh.id in [m.id for m in volume_meshes]
        break
