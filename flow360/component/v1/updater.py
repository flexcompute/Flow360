"""
Module containing updaters from version to version
"""

import re

from flow360.exceptions import Flow360NotImplementedError, Flow360RuntimeError


def _no_update(params_as_dict):
    return params_as_dict


UPDATE_MAP = [
    ("0.2.0b16", "0.2.0b17", _no_update),
    ("0.2.0b17", "0.2.0b18", _no_update),
    ("0.2.0b18", "23.3.0", _no_update),
    ("23.3.0", "23.3.*", _no_update),
    ("23.3.*", "24.2.*", _no_update),
    ("24.2.*", "24.2.*", _no_update),
    ("24.2.*", "24.3.*", _no_update),  # we should not allow to submit Flow360Params to version 24.3
    ("24.3.*", "24.11.*", _no_update),  # Note: Remove when we do not want V1 submit to 2.0 anymore
    ("24.11.*", "24.11.*", _no_update),  # Note: Remove when we do not want V1 submit to 2.0 anymore
    ("24.11.*", "25.2.0", _no_update),
    ("25.2.0", "25.2.1", _no_update),
    ("25.2.1", "25.2.2", _no_update),
    ("25.2.2", "25.2.3", _no_update),
    ("25.2.3", "25.4.0", _no_update),
    ("25.4.0", "25.4.1", _no_update),
    ("25.4.1", "25.5.2", _no_update),
    ("25.5.2", "25.5.3", _no_update),
    ("25.5.3", "25.5.4", _no_update),
]


def _version_match(version_1, version_2):
    pattern_1 = re.compile(version_1.replace(".", r"\.").replace("*", r".*"))
    pattern_2 = re.compile(version_2.replace(".", r"\.").replace("*", r".*"))
    return pattern_1.match(version_2) or pattern_2.match(version_1)


# pylint: disable=dangerous-default-value
def _find_update_path(version_from, version_to, update_map=UPDATE_MAP):
    path = []

    current_version = version_from
    while not _version_match(current_version, version_to):
        found_next_version = False

        for map_version_from, map_version_to, update_function in update_map:
            if (
                _version_match(map_version_from, current_version)
                and map_version_to != current_version
            ):
                next_version = map_version_to
                path.append(update_function)
                current_version = next_version
                found_next_version = True
                break

        if not found_next_version:
            raise Flow360NotImplementedError(
                f"No updater flow from {version_from} to {version_to} exists as of now"
            )

        if len(path) > len(update_map):
            raise Flow360RuntimeError(
                f"An error occured when trying to update from {version_from} to {version_to}. Contact support."
            )

    return path


def updater(version_from, version_to, params_as_dict):
    """
    Update parameters from version_from to version_to.

    Parameters
    ----------
    version_from : str
        The starting version.
    version_to : str
        The target version to update to.
    params_as_dict : dict
        A dictionary containing parameters to be updated.

    Returns
    -------
    dict
        Updated parameters as a dictionary.

    Raises
    ------
    Flow360NotImplementedError
        If no update path exists from version_from to version_to.

    Notes
    -----
    This function iterates through the update map starting from version_from and
    updates the parameters based on the update path found.
    """

    update_functions = _find_update_path(version_from=version_from, version_to=version_to)
    for fun in update_functions:
        params_as_dict = fun(params_as_dict)

    return params_as_dict
