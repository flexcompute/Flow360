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


def is_entity_dict(data: dict):
    """Check if current dict is an Entity item"""
    return data.get("name") and (
        data.get("private_attribute_registry_bucket_name")
        or data.get("private_attribute_entity_type_name")
    )


def populate_entity_id_with_name(*, params_as_dict: dict):
    """
    Recursively populates the entity item's private_attribute_id with its name if
    the private_attribute_id is none.
    """

    def recursive_populate_entity_id_with_name(*, data):
        if isinstance(data, dict):
            # Check if current dict is an Entity item
            if is_entity_dict(data=data):
                if "private_attribute_id" not in data or data["private_attribute_id"] is None:
                    data["private_attribute_id"] = data["name"]

            for value in data.values():
                recursive_populate_entity_id_with_name(data=value)

        elif isinstance(data, list):
            for element in data:
                recursive_populate_entity_id_with_name(data=element)

    recursive_populate_entity_id_with_name(data=params_as_dict)


def update_symmetry_ghost_entity_name_to_symmetric(*, params_as_dict: dict):
    """
    Recursively update ghost entity name from symmetric-* to symmetry-*
    """

    def recursive_update_symmetry_ghost_entity_name_to_symmetric(*, data):
        if isinstance(data, dict):
            # Check if current dict is an Entity item
            if (
                is_entity_dict(data=data)
                and data["private_attribute_entity_type_name"] == "GhostCircularPlane"
                and data["name"].startswith("symmetry")
            ):
                data["name"] = data["name"].replace("symmetry", "symmetric")

            for value in data.values():
                recursive_update_symmetry_ghost_entity_name_to_symmetric(data=value)

        elif isinstance(data, list):
            for element in data:
                recursive_update_symmetry_ghost_entity_name_to_symmetric(data=element)

    recursive_update_symmetry_ghost_entity_name_to_symmetric(data=params_as_dict)
