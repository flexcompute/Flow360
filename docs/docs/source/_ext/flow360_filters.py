"""Member filtering utilities for the Flow360 custom Sphinx autodoc extension."""

from __future__ import annotations

from typing import Annotated, get_args, get_origin

import pydantic as pd

EXCLUDED_MEMBERS = frozenset(
    {
        "model_fields",
        "model_config",
        "model_computed_fields",
        "model_fields_set",
        "model_constructor",
        "entity_bucket",
        "entity_type",
        "full_name",
        "used_entity_registry",
        "to_base",
        "SchemaConfig",
        "Config",
    }
)


def is_private_member(name: str) -> bool:
    """Returns True if the member name matches private naming conventions."""
    return name.startswith("private_attribute") or name.startswith("_") or name.endswith("_")


def is_frozen_field(obj) -> bool:
    """Returns True if the FieldInfo has ``frozen=True``."""
    if not isinstance(obj, pd.fields.FieldInfo):
        return False
    return getattr(obj, "frozen", False) is True


def is_excluded_member(name: str) -> bool:
    """Returns True if the name is in the hardcoded exclusion set."""
    return name in EXCLUDED_MEMBERS


def should_skip_member(name: str, annotation=None, obj=None) -> bool:
    """Master filtering function. Returns True if the member should be hidden from docs."""
    if is_private_member(name):
        return True

    if is_excluded_member(name):
        return True

    if is_frozen_field(obj):
        return True

    if isinstance(obj, pd.fields.FieldInfo):
        # PrivateAttr fields are FieldInfo instances with init=False and a private marker
        if hasattr(obj, "metadata"):
            for meta in obj.metadata:
                if isinstance(meta, type) and issubclass(meta, pd.fields.FieldInfo):
                    return True
        if getattr(obj, "init", None) is False and name.startswith("_"):
            return True

    return False
