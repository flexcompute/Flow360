import flow360 as fl

folder_A = fl.Folder.create("folder-python-level-A").submit()
folder_B = fl.Folder.create("folder-python-level-B", parent_folder=folder_A).submit()
folder_C = fl.Folder.create("folder-python-level-C", parent_folder=folder_B).submit()

folder_C = folder_C.move_to_folder(folder_A)
print(folder_C)