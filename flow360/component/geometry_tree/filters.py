"""
filters.py - Filter matching logic

Provides functions for matching node attributes against filter criteria.
Supports glob patterns (* and ?) for string matching.
"""

import re
from typing import Any, Dict

from .node_type import NodeType


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

        # Type uses exact NodeType comparison
        if key == "type":
            if node_attrs.get("type") != expected_value:
                return False
            continue

        # Other system attributes - match via pattern
        actual_value = node_attrs.get(key)

        if actual_value is None:
            return False

        if not matches_pattern(str(actual_value), str(expected_value)):
            return False

    return True


def is_face_node(node_attrs: Dict[str, Any]) -> bool:
    """Check if a node is a face."""
    return node_attrs.get("type") == NodeType.FACE
