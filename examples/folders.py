import flow360 as fl
from flow360.examples import OM6wing

# create folder in ROOT level
folder_A = fl.Folder.create("folder-python-level-A").submit()
print(folder_A)

# create folder inside the above folder
folder_B = fl.Folder.create("folder-python-level-B", parent_folder=folder_A).submit()
print(folder_B)

# create folder in ROOT level and move inside folder_B
folder_C = fl.Folder.create("folder-python-level-C").submit()
folder_C = folder_C.move_to_folder(folder_B)
print(folder_C)


OM6wing.get_files()

# submit mesh
volume_mesh = fl.VolumeMesh.from_file(OM6wing.mesh_filename, name="OM6wing-mesh")
volume_mesh = volume_mesh.submit()
print(volume_mesh)

# submit case using json file
params = fl.Flow360Params(OM6wing.case_json)
case = fl.Case.create("OM6wing-in-folder-C", params, volume_mesh.id)
case = case.submit()

# move case to folder_C
case = case.move_to_folder(folder_C)
print(case.info)
