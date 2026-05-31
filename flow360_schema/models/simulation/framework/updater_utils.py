"""Utiliy functions for updater"""

import logging
import re
from functools import wraps
from numbers import Number

import numpy as np

logger = logging.getLogger(__name__)


def recursive_remove_key(data, key: str, *additional_keys: str):
    """Recursively remove one or more keys from nested dict/list structures in place.

    This function performs an in-place traversal without unnecessary allocations
    to preserve performance. It handles arbitrarily nested combinations of
    dictionaries and lists.
    """
    keys: tuple[str, ...] = (key,) + additional_keys

    stack = [data]
    while stack:
        current = stack.pop()

        if isinstance(current, dict):
            for item_key in keys:
                current.pop(item_key, None)
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


PYTHON_API_VERSION_REGEXP = r"^(\d+)\.(\d+)\.(\d+)(?:b(\d+))?$"


def compare_dicts(dict1, dict2, atol=1e-15, rtol=1e-10, ignore_keys=None):
    """Check two dictionaries are same or not"""
    if ignore_keys is None:
        ignore_keys = set()

    dict1_filtered = {k: v for k, v in dict1.items() if k not in ignore_keys}
    dict2_filtered = {k: v for k, v in dict2.items() if k not in ignore_keys}

    if dict1_filtered.keys() != dict2_filtered.keys():
        logger.debug(
            "dict keys not equal:\n dict1 %s\n dict2 %s",
            sorted(dict1_filtered.keys()),
            sorted(dict2_filtered.keys()),
        )
        return False

    for key in dict1_filtered:
        value1 = dict1_filtered[key]
        value2 = dict2_filtered[key]

        if not compare_values(value1, value2, atol, rtol, ignore_keys):
            logger.debug(
                "dict value of key %s not equal:\n dict1 %s\n dict2 %s",
                key,
                dict1[key],
                dict2[key],
            )
            return False

    return True


def compare_values(value1, value2, atol=1e-15, rtol=1e-10, ignore_keys=None):
    """Check two values are same or not"""
    if isinstance(value1, Number) and isinstance(value2, Number):
        return np.isclose(value1, value2, rtol, atol)

    if isinstance(value1, tuple):
        value1 = list(value1)

    if isinstance(value2, tuple):
        value2 = list(value2)

    if type(value1) != type(value2):
        return False

    if isinstance(value1, dict) and isinstance(value2, dict):
        return compare_dicts(value1, value2, atol, rtol, ignore_keys)
    if isinstance(value1, list) and isinstance(value2, list):
        return compare_lists(value1, value2, atol, rtol, ignore_keys)
    return value1 == value2


def compare_lists(list1, list2, atol=1e-15, rtol=1e-10, ignore_keys=None):
    """Check two lists are same or not"""
    if len(list1) != len(list2):
        return False

    def is_simple_type(item):
        return isinstance(item, (str, int, float, bool)) or (
            isinstance(item, Number) and not isinstance(item, (dict, list))
        )

    if list1 and all(is_simple_type(item) for item in list1) and all(is_simple_type(item) for item in list2):
        list1, list2 = sorted(list1), sorted(list2)

    for item1, item2 in zip(list1, list2, strict=False):
        if not compare_values(item1, item2, atol, rtol, ignore_keys):
            logger.debug("list value not equal:\n list1 %s\n list2 %s", item1, item2)
            return False

    return True


class Flow360Version:
    """
    Parser for the Flow360 Python API version.
    Expected pattern is `major.minor.patch` (integers).
    """

    __slots__ = ["major", "minor", "patch"]

    def __init__(self, version: str):
        """
        Initialize the version by parsing a string like '23.1.2'.
        Each of major, minor, patch should be numeric.
        """
        match = re.match(PYTHON_API_VERSION_REGEXP, version.strip())
        if not match:
            raise ValueError(f"Invalid version string: {version}")

        self.major = int(match.group(1))
        self.minor = int(match.group(2))
        self.patch = int(match.group(3))

    def _comparison_key(self) -> tuple[int, int, int]:
        return (
            self.major,
            self.minor,
            self.patch,
        )

    def __lt__(self, other):
        return self._comparison_key() < other._comparison_key()

    def __le__(self, other):
        return self._comparison_key() <= other._comparison_key()

    def __gt__(self, other):
        return self._comparison_key() > other._comparison_key()

    def __ge__(self, other):
        return self._comparison_key() >= other._comparison_key()

    def __eq__(self, other):
        if not isinstance(other, Flow360Version):
            return NotImplemented
        return self._comparison_key() == other._comparison_key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"


def deprecation_reminder(version: str):
    """
    If your_package.__version__ > version, raise.
    Otherwise, do nothing special.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flow360_schema import __version__ as _schema_version

            current = Flow360Version(_schema_version)
            target = Flow360Version(version)
            if current > target:
                raise ValueError(
                    f"[INTERNAL] This validator or function is detecting/handling deprecated schema that was"
                    f" scheduled to be removed since {version}. "
                    "Please deprecate the schema now, write updater and remove related checks."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
