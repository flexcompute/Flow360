import flow360 as fl

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

project = fl.Project.from_cloud("PROJECT_ID_HERE")

with fl.SI_unit_system:
    params = fl.SimulationParams(...)

case = project.run_case(params)

# move case to folder_C
case = case.move_to_folder(folder_C)
print(case.info)
