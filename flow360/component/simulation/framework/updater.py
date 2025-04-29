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
    def _add_private_attribute_id_for_point_array(params_as_dict: dict) -> dict:
        """
                Check if PointArray has private_attribute_id. If not, generate the uuid and assign the id
        to all occurrence of the same PointArray
        """
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
        return params_as_dict

    params_as_dict = _add_private_attribute_id_for_point_array(params_as_dict=params_as_dict)
    update_symmetry_ghost_entity_name_to_symmetric(params_as_dict=params_as_dict)
    return params_as_dict


# pylint: disable=invalid-name, too-many-branches
def _to_25_2_0(params_as_dict):
    # Migrates the old DDES turbulence model interface to the new hybrid_model format.
    for model in params_as_dict.get("models", []):
        turb_dict = model.get("turbulence_model_solver")
        if not turb_dict:
            continue

        run_ddes = turb_dict.pop("DDES", None)
        grid_size_for_LES = turb_dict.pop("grid_size_for_LES", None)

        if run_ddes:
            turb_dict["hybrid_model"] = {
                "shielding_function": "DDES",
                "grid_size_for_LES": grid_size_for_LES,
            }

    if params_as_dict.get("outputs") is not None:
        for output in params_as_dict["outputs"]:
            if output.get("output_type") == "VolumeOutput":
                items = output.get("output_fields", {}).get("items", [])
                for old, new in [
                    ("SpalartAllmaras_DDES", "SpalartAllmaras_hybridModel"),
                    ("kOmegaSST_DDES", "kOmegaSST_hybridModel"),
                ]:
                    if old in items:
                        items.remove(old)
                        items.append(new)

            # Convert the observers in the AeroAcousticOutput to new schema
            if output.get("output_type") == "AeroAcousticOutput":
                legacy_observers = output.get("observers", [])
                converted_observers = []
                for position in legacy_observers:
                    converted_observers.append(
                        {"group_name": "0", "position": position, "private_attribute_expand": None}
                    )
                output["observers"] = converted_observers

    # Add ramping to MassFlowRate and move velocity direction to TotalPressure
    for model in params_as_dict.get("models", []):
        if model.get("type") == "Inflow" and "velocity_direction" in model.keys():
            velocity_direction = model.pop("velocity_direction", None)
            model["spec"]["velocity_direction"] = velocity_direction

        if model.get("spec") and model["spec"].get("type_name") == "MassFlowRate":
            model["spec"]["ramp_steps"] = None

    return params_as_dict


def _to_24_11_10(params_as_dict):
    fix_ghost_sphere_schema(params_as_dict=params_as_dict)
    return params_as_dict


def _to_25_2_1(params_as_dict):
    ## We need a better mechanism to run updater function once.
    fix_ghost_sphere_schema(params_as_dict=params_as_dict)
    return params_as_dict


def _to_25_2_3(params_as_dict):
    populate_entity_id_with_name(params_as_dict=params_as_dict)
    return params_as_dict


def _to_25_4_1(params_as_dict):
    if params_as_dict.get("meshing") is None:
        return params_as_dict
    meshing_defaults = params_as_dict["meshing"].get("defaults", {})
    if meshing_defaults.get("geometry_relative_accuracy"):
        geometry_relative_accuracy = meshing_defaults.pop("geometry_relative_accuracy")
        meshing_defaults["geometry_accuracy"] = {"value": geometry_relative_accuracy, "units": "m"}
    return params_as_dict


VERSION_MILESTONES = [
    (Flow360Version("24.11.1"), _to_24_11_1),
    (Flow360Version("24.11.7"), _to_24_11_7),
    (Flow360Version("24.11.10"), _to_24_11_10),
    (Flow360Version("25.2.0"), _to_25_2_0),
    (Flow360Version("25.2.1"), _to_25_2_1),
    (Flow360Version("25.2.3"), _to_25_2_3),
    (Flow360Version("25.4.1"), _to_25_4_1),
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

    if version_from >= version_milestones[-1][0]:
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
        The target version to update to. This has to be equal or higher than `version_from`
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
    version_from_is_newer = Flow360Version(version_from) > Flow360Version(version_to)

    if version_from_is_newer:
        raise ValueError(
            f"[Internal] Misuse of updater, version_from ({version_from}) is higher than version_to ({version_to})"
        )
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
