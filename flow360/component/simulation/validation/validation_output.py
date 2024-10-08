def _check_entities_type(value, ProbeOutputTypeStr: str):
    """check to ensure every entity has the same type"""
    if value is None:
        return value
    for entity in value.stored_entities:
        if type(entity) is not type(value.stored_entities[0]):
            raise ValueError(
                f"All entities in a single `{ProbeOutputTypeStr}` must have the same type: `Point` or `PointArray`."
            )
    return value
