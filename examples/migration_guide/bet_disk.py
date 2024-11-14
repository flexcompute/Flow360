"""
This module helps the users to use their old json files 
containing BETDisks in the new version of Flow360.
"""

import json
from typing import Optional

import pydantic as pd

from flow360.component.simulation.models.volume_models import BETDisk
from flow360.component.simulation.primitives import Cylinder
from flow360.component.simulation.unit_system import AngularVelocityType, LengthType, u


def bet_disk_convert(
    file: str,
    save: Optional[bool] = False,
    length_unit: LengthType = u.m,
    omega_unit: AngularVelocityType = u.deg / u.s,
):
    """
    Convert old BETDisks input from a json file into the new BETDisk format.

    This function provides the updated version of BETDisk
    parameters used in describing simulation cases.

    Parameters
    ----------
    file : str
        Name of the json file containing BETDisk information. Default is None.
    save_to_file : bool, optional
        Choose whether to save the output to a file/files.
    length_unit : LengthType, optional
        BETDisk parameters length unit. Default is u.m
    omega_unit : AngularVelocityType, optional
        BETDisk omega's unit. Default is u.deg / u.s.

    Returns
    -------
    BETDisks, Cylinders
        BETDisks defined using the provided file.
        Cylinder defined using the provided file.

    Raises
    ------
    AttributeError
        If the input unit values don't exit in the 'unyt' module.
    TypeError
        If required file name is not provided.
    FileNotFoundError
        If the provided file does not exist.

    Examples
    --------
    Example usage:

    >>> BETDisks = BETDisk_convert(
    ...     file="xv15_bet_line_hover_good.json",
    ...     length_unit = u.ft,
    ...     save = True,
    ... )
    >>> print(json.dumps(BETDisks[0].model_dump(), indent=4))

    """

    betdisks, cylinders = convert(
        file,
        length_unit,
        omega_unit,
    )
    bet_disk_list = []
    cylinder_list = []
    amount = 0
    for number, bet in enumerate(betdisks):
        cyl = cylinders[number]
        cylinder = Cylinder(**cyl)
        bet_disk_instance = BETDisk(**bet, volumes=cylinder)
        bet_disk_list.append(bet_disk_instance)
        cylinder_list.append(cylinder)
        amount = number

from flow360.log import log
...
    log.info(f"Available BETDisks: {amount+1}")

    save_to_file(bet_disk_list, cylinder_list, save)

    return bet_disk_list, cylinder_list


def convert(
    file,
    length_unit,
    omega_unit,
):
    """
    Converting old json format to the new one as well as adding user defined units.
    """
    with open(file) as reader:
        read = json.load(reader)
        read["BETDisk"] = read["BETDisks"]
        del read["BETDisks"]
        betdisk = read["BETDisk"]

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
        }

        keys_to_remove = [
            "axisOfRotation",
            "centerOfRotation",
            "radius",
            "thickness",
        ]

        betdisks = []
        cylinders = []
        for number, data in enumerate(betdisk):
            cylinder = {
                "name": f"cylinder{number}",
                "axis": data["axisOfRotation"],
                "center": data["centerOfRotation"] * length_unit,
                "inner_radius": 0 * length_unit,
                "outer_radius": data["radius"] * length_unit,
                "height": data["thickness"] * length_unit,
            }
            updated_data = {
                key_mapping.get(key, key): value
                for key, value in data.items()
                if key not in keys_to_remove
            }
            polars = []
            for items in updated_data["sectional_polars"]:
                polar = {
                    key_mapping.get(key, key): value
                    for key, value in items.items()
                    if key not in keys_to_remove
                }
                polars.append(polar)
            updated_data["sectional_polars"] = polars
            updated_data["omega"] = updated_data["omega"] * omega_unit
            updated_data["chord_ref"] = updated_data["chord_ref"] * length_unit
            cylinders.append(cylinder)
            betdisks.append(updated_data)

    return betdisks, cylinders


def save_to_file(bet_disk_list, cylinder_list, save):
    """
    Saving the information about BETDisks and Cylinders to their respective files.
    """
    if save is True:
        for number, bet in enumerate(bet_disk_list):
            with open(f"Betdisk{number+1}.json", "w") as betdisk:
                betdisk.write(json.dumps(bet.model_dump(), indent=4))
            with open(f"Cylinder{number+1}.json", "w") as cylinder:
                cylinder.write(json.dumps(cylinder_list[number].model_dump(), indent=4))
