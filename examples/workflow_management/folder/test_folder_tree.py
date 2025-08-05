import flow360 as fl

folder = fl.Folder.create("folder-items-test").submit()
# Insert or move items into folder here

tree = folder.get_folder_tree()


def print_tree(node, indent=0):
    if node:
        print("  " * indent + f"- {node['name']} ({node['id']})")
        for subfolder in node["subfolders"]:
            print_tree(subfolder, indent + 1)


print_tree(tree)
