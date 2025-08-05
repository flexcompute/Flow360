import flow360 as fl

folder = fl.Folder.create("metadata-test", tags=["test", "example"]).submit()

# Get and display metadata
info = folder.get_info(force=True)
print(f"Folder metadata:")
print(f"- ID: {info.id}")
print(f"- Name: {info.name}")
print(f"- User ID (Author): {info.user_id}")
print(f"- Parent Folder ID: {info.parent_folder_id}")
print(f"- Status: {info.status}")
print(f"- Tags: {info.tags}")
print(f"- Type: {info.type}")
print(f"- Deleted: {info.deleted}")
print(f"- Created At: {info.created_at}")
print(f"- Updated At: {info.updated_at}")
print(f"- Updated At: {info.updated_by}")
print(f"- Parent folders: {info.parent_folders}")
