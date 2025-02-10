# pylint: disable=invalid-name

"""Module for loading the monitors settings from Flow360 V0 configs and construct ProbeOutput for V2"""

import json
import os
from typing import List

from pydantic import validate_call

from flow360.component.simulation.outputs.output_entities import Point
from flow360.component.simulation.outputs.outputs import ProbeOutput
from flow360.log import log


@validate_call
def read_all_v0_monitors(*, file_path: str, mesh_unit) -> List[ProbeOutput]:
    """
    Read in the provided Flow360 ProbeOutput config.

    Returns a list of ProbeOutput instances
    """

    monitor_list = []
    data_dict = None

    if os.path.isfile(file_path) is False:
        raise FileNotFoundError(f"Supplied file: {file_path} cannot be found.")
    with open(file_path, "r", encoding="utf-8") as file:
        data_dict = json.load(file)

    if data_dict.get("monitorOutput") is None:
        log.warning("Input file does not contain `monitorOutput` key.")
        return monitor_list

    if data_dict["monitorOutput"].get("monitors") is None:
        log.warning("Input file does not contain the `monitors` key in the monitorOutput setting.")
        return monitor_list

    flow360_monitor_dict = data_dict["monitorOutput"]["monitors"]
    point_idx = 0

    for monitor_group_name, group_settings in flow360_monitor_dict.items():
        if group_settings["type"] != "probe":
            continue
        entities = []

        if not group_settings.get("outputFields") or len(group_settings["outputFields"]) == 0:
            raise ValueError(
                f"Invalid monitor settings: {monitor_group_name} monitor group does not specify any `outputFields`."
            )

        if (
            not group_settings.get("monitorLocations")
            or len(group_settings["monitorLocations"]) == 0
        ):
            raise ValueError(
                f"Invalid monitor settings: {monitor_group_name} monitor group does not specify any `monitorLocations`."
            )

        output_fields = group_settings["outputFields"]
        for location in group_settings["monitorLocations"]:
            entities.append(Point(name=f"Point-{point_idx}", location=location * mesh_unit))
            point_idx += 1
        monitor_list.append(
            ProbeOutput(name=monitor_group_name, entities=entities, output_fields=output_fields)
        )

    return monitor_list
