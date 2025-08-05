import flow360 as fl

folder = fl.Folder.create(
    "basic-folder-operations-test", tags=["operations-test", "example"]
).submit()
print(f"Created folder: {folder.id}")

info = folder.get_info()
print(f"Folder info: {info}")

print(f"Folder name: {folder.info.name}")
print(f"Parent folder ID: {folder.info.parent_folder_id}")
