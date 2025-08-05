import flow360 as fl

your_folder_id = ""
folder = fl.Folder(your_folder_id)

items = folder.get_items()

print(f"Found {len(items)} items in folder:")

for item in items:
    print(f"Name: {item['name']}, Type: {item['type']}, Size: {item.get('storageSize', 0)}")