"""
Utility module for operating on dicts for the CLI
"""


def merge_overwrite(old: dict, new: dict, path=None):
    """
    Perform a deep merge of two dictionaries overwriting a with b in case of conflicts
    """
    if path is None:
        path = []
    for key in new:
        if key in old:
            if isinstance(old[key], dict) and isinstance(new[key], dict):
                merge_overwrite(old[key], new[key], path + [str(key)])
            elif old[key] != new[key]:
                old[key] = new[key]
        else:
            old[key] = new[key]
    return old
