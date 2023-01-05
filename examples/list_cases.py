from flow360.component.case import CaseList
from flow360.component.volume_mesh import VolumeMeshList


# get all cases:
my_cases = CaseList(include_deleted=True)

# print metadata for the latest case:
print(my_cases[0])

for case in my_cases:
    print(f"id: {case.id}, status: {case.status}, is case deleted: {case.deleted}")

# get parameters for the latest case:
case0 = my_cases[0].to_case()
print(case0.params)
print(case0.params.time_stepping.max_pseudo_steps)


# get all meshes:
meshes = VolumeMeshList()
for mesh in meshes:
    print(f"mesh id: {mesh.id}, status: {mesh.status}, is mesh deleted: {mesh.deleted}")

mesh = meshes[0]
print(mesh.to_volume_mesh())

# list of cases for a given mesh
mesh_cases = CaseList(mesh_id=mesh.id, include_deleted=True)
for case in mesh_cases:
    print(f"id: {case.id}, status: {case.status}, is case deleted: {case.deleted}")
