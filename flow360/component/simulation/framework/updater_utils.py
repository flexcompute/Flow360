"""Utiliy functions for updater"""

import re
from functools import wraps
from numbers import Number

import numpy as np

from flow360.version import __version__

PYTHON_API_VERSION_REGEXP = r"^(\d+)\.(\d+)\.(\d+)(?:b(\d+))?$"


def compare_dicts(dict1, dict2, atol=1e-15, rtol=1e-10, ignore_keys=None):
    """Check two dictionaries are same or not"""
    if ignore_keys is None:
        ignore_keys = set()

    # Filter out the keys to be ignored
    dict1_filtered = {k: v for k, v in dict1.items() if k not in ignore_keys}
    dict2_filtered = {k: v for k, v in dict2.items() if k not in ignore_keys}

    if dict1_filtered.keys() != dict2_filtered.keys():
        print(f"dict keys not equal, dict1 {dict1_filtered.keys()}, dict2 {dict2_filtered.keys()}")
        return False

    for key in dict1_filtered:
        value1 = dict1_filtered[key]
        value2 = dict2_filtered[key]

        if not compare_values(value1, value2, atol, rtol, ignore_keys):
            print(f"dict value of key {key} not equal dict1 {dict1[key]}, dict2 {dict2[key]}")
            return False

    return True


def compare_values(value1, value2, atol=1e-15, rtol=1e-10, ignore_keys=None):
    """Check two values are same or not"""
    # Handle numerical comparisons first (including int vs float)
    if isinstance(value1, Number) and isinstance(value2, Number):
        return np.isclose(value1, value2, rtol, atol)

    # Handle type mismatches for non-numerical types
    # pylint: disable=unidiomatic-typecheck
    if type(value1) != type(value2):
        return False

    if isinstance(value1, Number) and isinstance(value2, Number):
        return np.isclose(value1, value2, rtol, atol)
    if isinstance(value1, dict) and isinstance(value2, dict):
        return compare_dicts(value1, value2, atol, rtol, ignore_keys)
    if isinstance(value1, list) and isinstance(value2, list):
        return compare_lists(value1, value2, atol, rtol, ignore_keys)
    return value1 == value2


def compare_lists(list1, list2, atol=1e-15, rtol=1e-10, ignore_keys=None):
    """Check two lists are same or not"""
    if len(list1) != len(list2):
        return False

    # Only sort if the lists contain simple comparable types (not dicts, lists, etc.)
    def is_simple_type(item):
        return isinstance(item, (str, int, float, bool)) or (
            isinstance(item, Number) and not isinstance(item, (dict, list))
        )

    if list1 and all(is_simple_type(item) for item in list1):
        list1, list2 = sorted(list1), sorted(list2)

    for item1, item2 in zip(list1, list2):
        if not compare_values(item1, item2, atol, rtol, ignore_keys):
            print(f"list value not equal list1 {item1}, list2 {item2}")
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
        # Match three groups of digits separated by dots
        match = re.match(PYTHON_API_VERSION_REGEXP, version.strip())
        if not match:
            raise ValueError(f"Invalid version string: {version}")

        self.major = int(match.group(1))
        self.minor = int(match.group(2))
        self.patch = int(match.group(3))

    def __lt__(self, other):
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other):
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)

    def __gt__(self, other):
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __ge__(self, other):
        return (self.major, self.minor, self.patch) >= (other.major, other.minor, other.patch)

    def __eq__(self, other):
        # Also check that 'other' is the same type or has the same attributes
        if not isinstance(other, Flow360Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

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
            current = Flow360Version(__version__)
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
