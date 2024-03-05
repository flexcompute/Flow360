"""
Module containing updaters from version to version 
"""

from ...exceptions import Flow360NotImplementedError


def _no_update(params_as_dict):
    return params_as_dict


UPDATE_MAP = [
    ("0.2.0b16", "0.2.0b17", _no_update),
    ("0.2.0b17", "0.2.0b18", _no_update),
    ("0.2.0b18", "23.3.1", _no_update),
    ("23.3.1", "24.2.0b1", _no_update),
]


def _find_update_path(version_from, version_to):
    path = []

    current_version = version_from

    while current_version != version_to:
        found_next_version = False

        for update_info in UPDATE_MAP:
            if update_info[0] == current_version:
                next_version = update_info[1]
                update_function = update_info[2]

                path.append(update_function)
                current_version = next_version
                found_next_version = True
                break

        if not found_next_version:
            raise Flow360NotImplementedError(
                f"No updater flow between {version_from} and {version_to} exists as of now"
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
