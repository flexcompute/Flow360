import flow360 as fl

folders = []
for i in range(5):
    folder = fl.Folder.create(f"batch-folder-{i}").submit()
    folders.append(folder)
    print(f"Created folder {i+1}: {folder.id}")

# Test moving multiple folders
target = fl.Folder.create("target-folder").submit()
for folder in folders:
    folder.move_to_folder(target)
    print(f"Moved {folder.id} to {target.id}")
