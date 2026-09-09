import flow360 as fl

# Creating folders in a parent folder
root_folder = fl.Folder.create(name="ParentFolder").submit()
new_folder= fl.Folder.create(name="new_folder",parent_folder=root_folder).submit()

# Looking for folders: find the folder from the name
def find_folder_by_name(root, target_name):
    """Walk the folder tree starting at `root` and return the first Folder
    object whose name matches `target_name`. Returns None if not found.
    """
    tree = root.get_folder_tree()

    def _recurse(node: dict):
        if node.get("name") == target_name:
            return fl.Folder(node["id"])
        for sub in node.get("subfolders", []):
            found = _recurse(sub)
            if found:
                return found
        return None

    return _recurse(tree)

# Example usage:
found_folder = find_folder_by_name(root_folder, "new_folder")
print(f"A folder was found by the name of:{found_folder.name}, and id={ found_folder.id}")

# Once you have created your structure you can use:
fl.Project.from_geometry("geometry.STEP", name="Example", folder=new_folder)
