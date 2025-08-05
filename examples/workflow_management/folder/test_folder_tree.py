import flow360 as fl

your_folder_id = ""
folder = fl.Folder(your_folder_id)
tree = folder.get_folder_tree()


def print_tree(node, indent=0):
    if node:
        print("  " * indent + f"- {node['name']} ({node['id']})")
        for subfolder in node["subfolders"]:
            print_tree(subfolder, indent + 1)


print_tree(tree)
