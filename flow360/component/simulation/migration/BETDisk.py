# pylint: disable=invalid-name

"""Module for loading the BETDisk settings from Flow360 V1 configs"""

import json
import os

from numpy import sqrt
from pydantic import validate_call

import flow360.component.simulation.units as u
from flow360.component.simulation.models.volume_models import BETDisk
from flow360.component.simulation.primitives import Cylinder
from flow360.component.simulation.unit_system import AbsoluteTemperatureType, LengthType
from flow360.log import log


def _parse_flow360_bet_disk_dict(
    *, flow360_bet_disk_dict: dict, mesh_unit, freestream_temperature, bet_disk_index: int = 0
):
    """
    Read in the provided Flow360 BETDisk config.
    This handles the conversion of **1** instance of BETDisk.
    For populating a list of BETDisks, use the function in SimulationParams.

    Returns the BETDisk and the cylinder entity used.
    """
    if "BETDisks" in flow360_bet_disk_dict:
        if len(flow360_bet_disk_dict["BETDisks"]) > 1:
            raise ValueError(
                "'BETDisks' list found in input file."
                " Please pass in single BETDisk setting at a time."
                " To read in all the BETDisks, use BETDisks.read_flow360_BETDisk_list()."
            )
        if len(flow360_bet_disk_dict["BETDisks"]) == 0:
            raise ValueError("Input file does not contain BETDisk setting.")
        flow360_bet_disk_dict = flow360_bet_disk_dict["BETDisks"][0]

    specific_heat_ratio = 1.4
    gas_constant = 287.0529 * u.m**2 / u.s**2 / u.K  # pylint: disable=no-member
    speed_of_sound = sqrt(specific_heat_ratio * gas_constant * freestream_temperature.to("K"))
    time_unit = mesh_unit / speed_of_sound

    key_mapping = {
        "rotationDirectionRule": "rotation_direction_rule",
        "numberOfBlades": "number_of_blades",
        "ReynoldsNumbers": "reynolds_numbers",
        "chordRef": "chord_ref",
        "nLoadingNodes": "n_loading_nodes",
        "sectionalRadiuses": "sectional_radiuses",
        "sectionalPolars": "sectional_polars",
        "MachNumbers": "mach_numbers",
        "liftCoeffs": "lift_coeffs",
        "dragCoeffs": "drag_coeffs",
        "tipGap": "tip_gap",
        "initialBladeDirection": "initial_blade_direction",
        "bladeLineChord": "blade_line_chord",
    }

    keys_to_remove = [
        "axisOfRotation",
        "centerOfRotation",
        "radius",
        "thickness",
    ]

    cylinder_dict = {
        "name": f"bet_cylinder{bet_disk_index+1}",
        "axis": flow360_bet_disk_dict["axisOfRotation"],
        "center": flow360_bet_disk_dict["centerOfRotation"] * mesh_unit,
        "inner_radius": 0 * mesh_unit,
        "outer_radius": flow360_bet_disk_dict["radius"] * mesh_unit,
        "height": flow360_bet_disk_dict["thickness"] * mesh_unit,
    }

    updated_bet_dict = {
        key_mapping.get(key, key): value
        for key, value in flow360_bet_disk_dict.items()
        if key not in keys_to_remove
    }

    updated_bet_dict["twists"] = [
        {
            "radius": twist["radius"] * mesh_unit,
            "twist": twist["twist"] * u.deg,  # pylint: disable = no-member
        }
        for twist in updated_bet_dict["twists"]
    ]

    if "tip_gap" not in updated_bet_dict:
        updated_bet_dict["tip_gap"] = "inf"

    if updated_bet_dict["tip_gap"] != "inf":
        updated_bet_dict["tip_gap"] = updated_bet_dict["tip_gap"] * mesh_unit

    updated_bet_dict["chords"] = [
        {"radius": chord["radius"] * mesh_unit, "chord": chord["chord"] * mesh_unit}
        for chord in updated_bet_dict["chords"]
    ]

    updated_bet_dict["sectional_polars"] = [
        {
            key_mapping.get(key, key): value
            for key, value in polars.items()
            if key not in keys_to_remove
        }
        for polars in updated_bet_dict["sectional_polars"]
    ]

    updated_bet_dict["alphas"] = updated_bet_dict["alphas"] * u.deg  # pylint: disable = no-member
    updated_bet_dict["omega"] = (
        updated_bet_dict["omega"] * u.rad / time_unit  # pylint: disable = no-member
    )

    log.info("Provided temperature was used to calculate the value of omega in rad/s.")
    log.info("You can print and correct the value and unit of `Omega` afterwards if needed.")

    updated_bet_dict["chord_ref"] = updated_bet_dict["chord_ref"] * mesh_unit
    updated_bet_dict["sectional_radiuses"] = updated_bet_dict["sectional_radiuses"] * mesh_unit

    if "blade_line_chord" in updated_bet_dict:
        updated_bet_dict["blade_line_chord"] = updated_bet_dict["blade_line_chord"] * mesh_unit
    return updated_bet_dict, cylinder_dict


def _load_flow360_json(*, file_path: str) -> dict:
    if os.path.isfile(file_path) is False:
        raise FileNotFoundError(f"Supplied file: {file_path} cannot be found.")
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


@validate_call
def read_single_v1_BETDisk(
    file_path: str,
    mesh_unit: LengthType.NonNegative,  # pylint: disable = no-member
    freestream_temperature: AbsoluteTemperatureType,
) -> BETDisk:
    """
    Constructs a single :class: `BETDisk` instance from a given V1 (legacy) Flow360 input.

    Parameters
    ----------
    file_path: str
        Path to Flow360 JSON file that contains a **single** BETDisk setting.
    mesh_unit: LengthType.NonNegative
        Length unit used for LengthType BETDisk parameters.
    time_unit: TimeType.Positive
        Time unit used for non-dimensionalization.

    Returns
    -------
    BETDisk
        An instance of :class:`BETDisk` completed with given inputs.

    Examples
    --------
    Create a BET disk with an XROTOR file.

    >>> param = fl.BETDisk.from_flow360(
    ...     file=fl.Flow360File(file_name="flow360.json")),
    ...     mesh_unit=param.length_unit,
    ...     time_unit=param.time_unit,
    ... )
    """

    try:
        bet_disk_dict, cylinder_dict = _parse_flow360_bet_disk_dict(
            flow360_bet_disk_dict=_load_flow360_json(file_path=file_path),
            mesh_unit=mesh_unit,
            freestream_temperature=freestream_temperature,
        )

        return BETDisk(**bet_disk_dict, entities=Cylinder(**cylinder_dict))
    except KeyError as err:
        raise ValueError(
            "The supplied Flow360 input for BETDisk hsa invalid format. Details: " + str(err) + "."
        ) from err


@validate_call
def read_all_v1_BETDisks(
    file_path: str,
    mesh_unit: LengthType.NonNegative,  # pylint: disable = no-member
    freestream_temperature: AbsoluteTemperatureType,
) -> list[BETDisk]:
    """
    Read in Legacy V1 Flow360.json and convert its BETDisks settings to a list of :class: `BETDisk` instances

    Parameters
    ----------
    file_path: str
        Path to the Flow360.json file.
    mesh_unit: LengthType.NonNegative
        Length unit used for LengthType BETDisk parameters.
    time_unit: TimeType.Positive
        Time unit used for non-dimensionalization.

    Examples
    --------
    Create a BET disk with an XROTOR file.

    >>> param = fl.BETDisk.from_flow360(
    ...     file=fl.Flow360File(file_name="flow360.json")),
    ...     mesh_unit=param.length_unit,
    ...     time_unit=param.time_unit,
    ... )
    """

    bet_list = []

    data_dict = _load_flow360_json(file_path=file_path)

    if "BETDisks" not in data_dict.keys():
        raise ValueError("Cannot find 'BETDisk' key in the supplied JSON file.")

    if not data_dict.get("BETDisks", None):
        raise ValueError("'BETDisk'in the supplied JSON file contains no info.")

    bet_disk_index = 0
    for item in data_dict.get("BETDisks"):
        bet_disk_dict, cylinder_dict = _parse_flow360_bet_disk_dict(
            flow360_bet_disk_dict=item,
            mesh_unit=mesh_unit,
            freestream_temperature=freestream_temperature,
        )
        bet_list.append(BETDisk(**bet_disk_dict, entities=Cylinder(**cylinder_dict)))
        bet_disk_index += 1
    return bet_list
