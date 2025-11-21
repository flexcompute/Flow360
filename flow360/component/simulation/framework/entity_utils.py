"""Shared utilities for entity operations."""

import uuid


def generate_uuid():
    """generate a unique identifier for non-persistent entities. Required by front end."""
    return str(uuid.uuid4())
