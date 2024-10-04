import flow360 as fl
from flow360.cloud.rest_api import RestApi
fl.Env.preprod.active()


def format_size(size_in_bytes):
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 ** 2:
        return f"{size_in_bytes / 1024:.2f} kB"
    elif size_in_bytes < 1024 ** 3:
        return f"{size_in_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_in_bytes / (1024 ** 3):.2f} GB"

def get_folder_items(folder_id):
    payload = {
        "page": 0,
        "size": 1000,
        "filterFolderIds": folder_id,
        "filterExcludeSubfolders": True,
        "sortFields": ["storageSize"],
        "sortDirections": ["desc"],
        "expandFields": ['contentInfo']
    }
    data = RestApi('/v2/items').get(params=payload)
    return data.get('records', [])



def build_folder_tree(folders):
    folder_dict = {folder['id']: folder for folder in folders}
    tree = {}
    for folder in folders:
        folder_id = folder['id']
        parent_id = folder.get('parentFolderId')
        if parent_id is None or parent_id == "ROOT.FLOW360":
            tree[folder_id] = folder
            folder['subfolders'] = []
        else:
            parent_folder = folder_dict.get(parent_id)
            if parent_folder:
                if 'subfolders' not in parent_folder:
                    parent_folder['subfolders'] = []
                parent_folder['subfolders'].append(folder)

    return tree



def display_and_calculate_storage(folder, indent=0):
    print("  " * indent + f"- [FOLDER] {folder['name']}")
    total_storage = 0
    if 'subfolders' in folder:
        for subfolder in folder['subfolders']:
            total_storage += display_and_calculate_storage(subfolder, indent + 1)

    items = get_folder_items(folder['id'])
    for item in items:
        if item['type'] != "Folder":
            storage_size = item.get('storageSize', 0)
            total_storage += storage_size
            print("  " * (indent + 1) + f"- {item['name']} (Size: {format_size(storage_size)})")

    print("  " * (indent + 1) + f"Total Storage: {format_size(total_storage)}")
    return total_storage


payload = {
    "includeSubfolders": True,
    "page": 0,
    "size": 50
}

data = RestApi('/v2/folders').get(json=payload)
folder_tree = build_folder_tree(data['records'])

print("Folders and Subfolders Structure:")
folder_tree = {
    'id': "ROOT.FLOW360",
    'name': "ROOT.FLOW360",
    'subfolders': folder_tree.values()
}
display_and_calculate_storage(folder_tree)
