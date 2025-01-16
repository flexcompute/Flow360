"""
Module containing updaters from version to version

TODO: remove duplication code with FLow360Params updater. 
"""

# pylint: disable=R0801

import re

from ....exceptions import Flow360NotImplementedError, Flow360RuntimeError
from .entity_base import generate_uuid
from .updater_utils import compare_dicts


def _no_update(params_as_dict):
    return params_as_dict


def _24_11_6_to_24_11_7_update(params_as_dict):
    # Check if PointArray has private_attribute_id. If not, generate the uuid and assign the id
    # to all occurance of the same PointArray
    if params_as_dict.get("outputs") is None:
        return params_as_dict

    point_array_list = []
    for output in params_as_dict["outputs"]:
        if output.get("entities", None):
            for entity in output["entities"]["stored_entities"]:
                if (
                    entity.get("private_attribute_entity_type_name") == "PointArray"
                    and entity.get("private_attribute_id") is None
                ):
                    new_uuid = generate_uuid()
                    entity["private_attribute_id"] = new_uuid
                    point_array_list.append(entity)

    if params_as_dict["private_attribute_asset_cache"].get("project_entity_info"):
        for idx, draft_entity in enumerate(
            params_as_dict["private_attribute_asset_cache"]["project_entity_info"]["draft_entities"]
        ):
            if draft_entity.get("private_attribute_entity_type_name") != "PointArray":
                continue
            for point_array in point_array_list:
                if compare_dicts(
                    dict1=draft_entity,
                    dict2=point_array,
                    ignore_keys=["private_attribute_id"],
                ):
                    params_as_dict["private_attribute_asset_cache"]["project_entity_info"][
                        "draft_entities"
                    ][idx] = point_array
                    continue

    return params_as_dict


UPDATE_MAP = [
    ("24.11.([0-5])$", "24.11.6", _no_update),
    ("24.11.6", "24.11.7", _24_11_6_to_24_11_7_update),
    ("24.11.7", "24.11.*", _no_update),
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
