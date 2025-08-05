import flow360 as fl

folder = fl.Folder.create("folder-items-test").submit()

# Insert or move items into folder here

items = folder.get_items()

print(f"Found {len(items)} items in folder:")

for item in items:
    print(f"Name: {item['name']}, Type: {item['type']}, Size: {item.get('storageSize', 0)}")
