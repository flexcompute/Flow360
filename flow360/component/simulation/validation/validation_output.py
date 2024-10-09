"""
Validation for output parameters
"""


def _check_unique_probe_type(value, probe_output_type_str: str):
    """check to ensure every entity has the same type"""
    if value is None:
        return value
    for entity in value.stored_entities:
        if type(entity) is not type(value.stored_entities[0]):
            raise ValueError(
                f"All entities in a single `{probe_output_type_str}` must have the same type: `Point` or `PointArray`."
            )
    return value
