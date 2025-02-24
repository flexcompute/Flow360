"""Implementation of the updater functions. The updated.py should just import functions from here."""


def fix_ghost_sphere_schema(*, params_as_dict: dict):
    """
    The previous ghost farfield has wrong schema (bug) and therefore needs data alternation.
    """

    def i_am_outdated_ghost_sphere(*, data: dict):
        """Identify if the current dict is a outdated ghost sphere."""
        if "type_name" in data.keys() and data["type_name"] == "GhostSphere":
            return True
        return False

    def recursive_fix_ghost_surface(*, data):
        if isinstance(data, dict):
            # 1. Check if this is a ghost sphere instance
            if i_am_outdated_ghost_sphere(data=data):
                data.pop("type_name")
                data["private_attribute_entity_type_name"] = "GhostSphere"

            # 2. Otherwise, recurse into each item in the dictionary
            for _, val in data.items():
                recursive_fix_ghost_surface(
                    data=val,
                )

        elif isinstance(data, list):
            # Recurse into each item in the list
            for _, item in enumerate(data):
                recursive_fix_ghost_surface(data=item)

    recursive_fix_ghost_surface(data=params_as_dict)
