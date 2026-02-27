"""
filters.py - Filter matching logic

Provides functions for matching node attributes against filter criteria.
Supports glob patterns (* and ?) for string matching.
"""

import re
from typing import Any, Dict


def glob_to_regex(pattern: str) -> re.Pattern:
    """
    Convert glob pattern to regex.

    Supports:
        * - matches any number of characters
        ? - matches exactly one character
    """
    # Escape special regex characters except * and ?
    escaped = re.escape(pattern)
    # Convert glob wildcards to regex
    regex_pattern = escaped.replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile(f"^{regex_pattern}$", re.IGNORECASE)


def matches_pattern(value: str, pattern: str) -> bool:
    """
    Check if a string value matches a glob pattern.
    """
    if value is None:
        return False

    # If no wildcards, do exact case-insensitive match
    if "*" not in pattern and "?" not in pattern:
        return value.lower() == pattern.lower()

    regex = glob_to_regex(pattern)
    return bool(regex.match(value))


def matches_criteria(node_attrs: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """
    Check if node attributes match all given criteria.

    System attributes (name, type, colorRGB, etc.) are matched at top level.
    Custom attributes should be passed via 'attributes' parameter:
        attributes={"groupName": "wing"}
    """
    for key, expected_value in criteria.items():
        # Handle custom attributes dict separately
        if key == "attributes" and isinstance(expected_value, dict):
            node_custom_attrs = node_attrs.get("attributes", {})
            for attr_key, attr_pattern in expected_value.items():
                actual_value = node_custom_attrs.get(attr_key)
                if actual_value is None:
                    return False
                if not matches_pattern(str(actual_value), str(attr_pattern)):
                    return False
            continue

        # System attributes - only match top-level keys
        actual_value = node_attrs.get(key)

        if actual_value is None:
            return False

        # Convert to string for pattern matching
        actual_str = str(actual_value)
        expected_str = str(expected_value)

        if not matches_pattern(actual_str, expected_str):
            return False

    return True


def is_face_node(node_attrs: Dict[str, Any]) -> bool:
    """Check if a node is a face (Face or FacePointer)."""
    node_type = node_attrs.get("type", "")
    return node_type in ("Face", "FacePointer")


def get_face_uuid(node_attrs: Dict[str, Any]):
    """Get the _Flow360UUID of a face node."""
    if "_Flow360UUID" in node_attrs:
        return node_attrs["_Flow360UUID"]

    if "attributes" in node_attrs:
        return node_attrs["attributes"].get("_Flow360UUID")

    return None
