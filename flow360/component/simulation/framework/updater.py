"""
Module containing updaters from version to version

TODO: remove duplication code with FLow360Params updater.
"""

# pylint: disable=R0801


import re
from typing import Any

from flow360.component.simulation.framework.entity_base import generate_uuid
from flow360.component.simulation.framework.updater_functions import (
    fix_ghost_sphere_schema,
    populate_entity_id_with_name,
    update_symmetry_ghost_entity_name_to_symmetric,
)
from flow360.component.simulation.framework.updater_utils import (
    Flow360Version,
    compare_dicts,
)
from flow360.log import log
from flow360.version import __version__


def _to_24_11_1(params_as_dict):
    # Check and remove the 'meshing' node if conditions are met
    if params_as_dict.get("meshing") is not None:
        meshing_defaults = params_as_dict["meshing"].get("defaults", {})
        bl_thickness = meshing_defaults.get("boundary_layer_first_layer_thickness")
        max_edge_length = meshing_defaults.get("surface_max_edge_length")
        if bl_thickness is None and max_edge_length is None:
            del params_as_dict["meshing"]

    # Iterate over models and update 'heat_spec' where necessary
    for model in params_as_dict.get("models", []):
        if model.get("type") == "Wall" and model.get("heat_spec") is None:
            model["heat_spec"] = {
                "type_name": "HeatFlux",
                "value": {"value": 0, "units": "W / m**2"},
            }

    # Check and remove the 'time_stepping' -> order_of_accuracy node
    if "time_stepping" in params_as_dict:
        params_as_dict["time_stepping"].pop("order_of_accuracy", None)

    update_symmetry_ghost_entity_name_to_symmetric(params_as_dict=params_as_dict)
    return params_as_dict


def _to_24_11_7(params_as_dict):
    # Check if PointArray has private_attribute_id. If not, generate the uuid and assign the id
    # to all occurrence of the same PointArray
    if params_as_dict.get("outputs") is None:
        return params_as_dict

    point_array_list = []
    for output in params_as_dict["outputs"]:
        if output.get("entities", None) and output["entities"].get("stored_entities", None):
            for entity in output["entities"]["stored_entities"]:
                if (
                    entity.get("private_attribute_entity_type_name") == "PointArray"
                    and entity.get("private_attribute_id") is None
                ):
                    new_uuid = generate_uuid()
                    entity["private_attribute_id"] = new_uuid
                    point_array_list.append(entity)

    if not params_as_dict["private_attribute_asset_cache"].get("project_entity_info"):
        return params_as_dict
    if not params_as_dict["private_attribute_asset_cache"]["project_entity_info"].get(
        "draft_entities"
    ):
        return params_as_dict
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

    update_symmetry_ghost_entity_name_to_symmetric(params_as_dict=params_as_dict)
    return params_as_dict


def _to_24_11_10(params_as_dict):
    fix_ghost_sphere_schema(params_as_dict=params_as_dict)
    return params_as_dict


def _to_24_11_12(params_as_dict):
    populate_entity_id_with_name(params_as_dict=params_as_dict)
    return params_as_dict


VERSION_MILESTONES = [
    (Flow360Version("24.11.1"), _to_24_11_1),
    (Flow360Version("24.11.7"), _to_24_11_7),
    (Flow360Version("24.11.10"), _to_24_11_10),
    (Flow360Version("24.11.12"), _to_24_11_12),
]  # A list of the Python API version tuple with there corresponding updaters.


# pylint: disable=dangerous-default-value
def _find_update_path(
    *,
    version_from: Flow360Version,
    version_to: Flow360Version,
    version_milestones: list[tuple[Flow360Version, Any]],
):

    if version_from == version_to:
        return []

    if version_from > version_to:
        raise ValueError(
            "Input `SimulationParams` have higher version than the target version and thus cannot be handled."
        )

    if version_from > version_milestones[-1][0]:
        raise ValueError(
            "Input `SimulationParams` have higher version than all known versions and thus cannot be handled."
        )

    if version_from == version_milestones[-1][0]:
        return []

    if version_to < version_milestones[0][0]:
        raise ValueError(
            "Trying to update `SimulationParams` to a version lower than any known version."
        )

    def _get_path_start():
        for index, item in enumerate(version_milestones):
            milestone_version = item[0]
            if milestone_version > version_from:
                # exclude equal because then it is already `milestone_version` version
                return index
        return None

    def _get_path_end():
        for index, item in enumerate(version_milestones):
            milestone_version = item[0]
            if milestone_version > version_to:
                return index - 1
        return len(version_milestones) - 1

    path_start = _get_path_start()
    path_end = _get_path_end()

    return [
        item[1] for index, item in enumerate(version_milestones) if path_start <= index <= path_end
    ]


def updater(version_from, version_to, params_as_dict) -> dict:
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
    log.debug(f"Input SimulationParam has version: {version_from}.")
    update_functions = _find_update_path(
        version_from=Flow360Version(version_from),
        version_to=Flow360Version(version_to),
        version_milestones=VERSION_MILESTONES,
    )
    for fun in update_functions:
        _to_version = re.search(r"_to_(\d+_\d+_\d+)", fun.__name__).group(1)
        log.debug(f"Updating input SimulationParam to {_to_version}...")
        params_as_dict = fun(params_as_dict)
    params_as_dict["version"] = str(version_to)
    return params_as_dict
