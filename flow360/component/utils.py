"""
Utility functions
"""
import uuid


# pylint: disable=redefined-builtin
def is_valid_uuid(id, ignore_none=False):
    """
    Checks if id is valid
    """
    if id is None and ignore_none:
        return
    try:
        uuid.UUID(str(id))
    except Exception as exc:
        raise ValueError(f"{id} is not a valid UUID.") from exc
