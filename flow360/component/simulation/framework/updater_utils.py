"""Utiliy functions for updater"""

from numbers import Number

import numpy as np


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

    if list1 and not isinstance(list1[0], dict):
        list1, list2 = sorted(list1), sorted(list2)

    for item1, item2 in zip(list1, list2):
        if not compare_values(item1, item2, atol, rtol, ignore_keys):
            print(f"list value not equal list1 {item1}, list2 {item2}")
            return False

    return True
