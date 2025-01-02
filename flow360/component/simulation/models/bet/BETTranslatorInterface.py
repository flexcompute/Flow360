"""
This Module is meant to be imported into scripts that translate the required information into a Flow360 input JSON file
with BET disk(s).

Explain XROTOR, DFDC, C81, Xfoil.

EXAMPLE useage:

EXAMPLE codes

"""

import json

# import sys
import os
from math import *
from os import path

import numpy as np
from scipy.interpolate import interp1d

import flow360.component.simulation.units as u

from .utils import *


########################################################################################################################
def read_in_xfoil_polar(polar_file):
    """
    Parameters
    ----------
    polar_file: path to the xfoil polar file.

    Returns
    -------
    alpha_list, mach_list, cl_list, cd_list
    """
    cl_alphas = []
    cl_values = {}  # dictionary of list with the machs as keys
    cd_values = {}  # dictionary of list with the machs as keys

    xfoil_fid = open(polar_file, "r")
    xfoil_fid.readline()  # skip the header
    for i in range(8):  # skip the first 9 lines
        line = xfoil_fid.readline()

    mach_num = line.strip().split(" ")[4]
    cl_values[mach_num] = []
    cd_values[mach_num] = []
    for i in range(4):  # skip the next 4 lines
        line = xfoil_fid.readline()
    while True:
        linecontents = line.strip().split(" ")

        c = linecontents.count(
            ""
        )  # remove all instances of '' because that number varies form file to file.
        for i in range(c):
            linecontents.remove("")

        cl_alphas.append(float(linecontents[0]))
        cl_values[mach_num].append(float(linecontents[1]))
        cd_values[mach_num].append(float(linecontents[2]))
        line = xfoil_fid.readline()
        if len(line) == 0:  # If we did all the alphas and we are done
            break
    # extrapolate alphas to +-180 deg and Use the flat plate Cl and CD outside of where we have values from Xfoil
    cl_alphas, cl_mach_nums, cl_values, cd_values = blend_polars_to_flat_plate(
        cl_alphas, [mach_num], cl_values, cd_values
    )

    # Now we interpolate the polar data to a constant set of alphas to make sure we have all the smae alphas across all mach and section
    # 10 deg steps from -180 ->-30 and from 30 to 180. 1 deg steps from -29 to 29
    deg_increment_ang = list(np.arange(-30, 30, 1).astype(float))

    alphas = (
        list(np.arange(-180, -30, 10).astype(float))
        + deg_increment_ang
        + list(np.arange(30, 190, 10).astype(float))
    )  # json doesn't like the numpy default int64 type so I make it a float

    cl_interp = interp1d(
        cl_alphas, cl_values[cl_mach_nums[0]], kind="linear"
    )  # method should be linear to make sure we still have 0 at the +- 180 values
    cd_interp = interp1d(
        cl_alphas, cd_values[cl_mach_nums[0]], kind="linear"
    )  # method should be linear to make sure we still have 0 at the +- 180 values
    cls = [0 for i in range(len(alphas))]
    cds = [0 for i in range(len(alphas))]
    for i, alpha in enumerate(alphas):  # interpolate the cl and cd over the new set of alphas
        cls[i] = float(cl_interp(alpha))
        cds[i] = float(cd_interp(alpha))

    xfoil_fid.close()

    return alphas, cl_mach_nums[0], cls, cds


########################################################################################################################
def blend_polars_to_flat_plate(cl_alphas, cl_mach_nums, cl_values, cd_values):
    """
    This function blends a given arbitrary set of CL and CD polars that are missing values to cover the whole -180 to 180
    range of angles. The resulting polars will have the missing values be replaced by the flat plate CL and CD.
    Parameters
    ----------
    cl_alphas: list of alpha angles
    cl_mach_nums: list of mach numbers
    cl_values: dict with dimensions nMach*nAlphas
    cd_values: dict with dimensions nMach*nAlphas

    Returns
    -------
    cl_alphas, cl_mach_nums, cl_values, cd_values with polars completed to +- 180
    """

    polar_alpha_step_blend = 10  # add a polar point every N alpha

    alpha_min = cl_alphas[0]
    alpha_max = cl_alphas[-1]
    if alpha_min < -180:
        raise ValueError(f"ERROR: alpha_min is smaller than -180: {alpha_min}")
    if alpha_max > 180:
        raise ValueError(f"ERROR: alpha_max is greater than 180: {alpha_max}")

    blend_window = 0.5  # 0.5 radians

    # create a point every 10 deg, how many points do we need.
    num_missing_alphas_min = round((alpha_min + 180) / polar_alpha_step_blend)
    num_missing_alphas_max = round((180 - alpha_max) / polar_alpha_step_blend)

    for i in range(num_missing_alphas_min - 1):  # add alphas at beginning of clAlphas list
        cl_alphas.insert(0, cl_alphas[0] - polar_alpha_step_blend)
        a = cl_alphas[0] * pi / 180  # smallest alpha in radians
        for i, mach in enumerate(cl_mach_nums):
            blend_val = blend_func_value(
                blend_window, a, alpha_min * pi / 180, "below_cl_min"
            )  # we are on the alphaCLmin side going up in CL
            # this follows the flat plate lift and drag equations times the blend val coefficient

            cl = cl_values[mach][0] * blend_val + (1 - blend_val) * cos(a) * 2 * pi * sin(a) / sqrt(
                1 + (2 * pi * sin(a)) ** 2
            )
            cd = (
                cd_values[mach][0] * blend_val
                + (1 - blend_val)
                * sin(a)
                * (2 * pi * sin(a)) ** 3
                / sqrt(1 + (2 * pi * sin(a)) ** 6)
                + 0.05
            )
            mach = str(mach)
            cl_values[mach].insert(0, cl)  # add a new cl value at the beginning
            cd_values[mach].insert(0, cd)  # add a new cd value at the beginning

    for i in range(num_missing_alphas_max - 1):  # add alphas at end of clAlphas list
        cl_alphas.append(cl_alphas[-1] + polar_alpha_step_blend)
        a = cl_alphas[-1] * pi / 180  # smallest alpha in radians
        for i, mach in enumerate(cl_mach_nums):
            blend_val = blend_func_value(
                blend_window, a, alpha_max * pi / 180, "above_cl_max"
            )  # we are on the alphaCLmin side going up in CL
            # this follows the flat plate lift and drag equations times the blend val coefficient

            cl = cl_values[mach][-1] * blend_val + (1 - blend_val) * cos(a) * 2 * pi * sin(
                a
            ) / sqrt(1 + (2 * pi * sin(a)) ** 2)
            cd = (
                cd_values[mach][-1] * blend_val
                + (1 - blend_val)
                * sin(a)
                * (2 * pi * sin(a)) ** 3
                / sqrt(1 + (2 * pi * sin(a)) ** 6)
                + 0.05
            )
            mach = str(mach)
            cl_values[mach].append(cl)  # add a new cl value at the beginning
            cd_values[mach].append(cd)  # add a new cl value at the beginning

    cl_alphas.insert(0, -180)  # make sure that the last value in the list is 180
    cl_alphas.append(180)  # make sure that the last value in the list is 180
    for i, mach in enumerate(cl_mach_nums):
        cl_values[mach].insert(0, 0)  # make sure Cl=0 at alpha -180
        cd_values[mach].insert(0, 0.05)  # Cd=0.05 is flat plate Cd at 180
        cl_values[mach].append(0)  # make sure Cl=0 at alpha -180
        cd_values[mach].append(0.05)  # Cd=0.05 is flat plate Cd at 180

    return cl_alphas, cl_mach_nums, cl_values, cd_values


###############################################################################################################
def read_in_c81_polar_c81_format(polar_file):
    """
    Read in the c81 format polar file
    This function checks that the list of Alphas is consistent across CL and CD and that the number of Machs is also consistent across Cl and CD.
    Parameters
    ----------
    polarFile

    Returns
    -------
    4 lists of floats: cl_alphas, cl_mach_nums, cl_values, cd_values
    """
    cl_alphas = []
    cd_alphas = []
    cl_values = {}  # dictionary of list with the machs as keys
    cd_values = {}  # dictionary of list with the machs as keys

    c81_fid = open(polar_file, "r")
    c81_fid.readline()  # skip the header
    line = c81_fid.readline()
    cl_mach_nums = line.strip().split(" ")
    cl_mach_nums = [
        float(i) for i in cl_mach_nums if i
    ]  # remove empty items and trailing \n in cl_mach_nums
    for mach in cl_mach_nums:
        cl_values[mach] = []
    line = c81_fid.readline()
    while True:
        # c81 format is as per this document https://cibinjoseph.github.io/C81-Interface/page/index.html
        # first 7 chars in string is A0A then 7 chars per cl value
        if (
            line[:7] == "       "
        ):  # If we did all the alphas and now that line starts with a bunch of spaces.
            break
        cl_alphas.append(float(line[:7]))

        for i, mach in enumerate(cl_mach_nums):
            index_beg = i * 7 + 7
            index_end = (i + 1) * 7 + 7
            cl_values[mach].append(float(line[index_beg:index_end]))
        line = c81_fid.readline()

    # Now do the CDs
    cd_mach_nums = line.strip().split(" ")
    # we already read the mach numbers line in the while loop above. so just split it.
    cd_mach_nums = [
        float(i) for i in cd_mach_nums if i
    ]  # remove empty items and trailing \n in clMachNums
    if cl_mach_nums != cd_mach_nums:  # if we have different lists of  machs
        raise ValueError(
            f"ERROR: in file {polar_file}, The machs in the Cl polar do not match the machs in the CD polar, we have {cl_mach_nums} Cl mach values and {cd_mach_nums} CD mach values:"
        )

    for mach in cd_mach_nums:
        cd_values[mach] = []
    line = c81_fid.readline()
    while True:
        # c81 format is as per this document https://cibinjoseph.github.io/C81-Interface/page/index.html
        # first 7 chars in string is A0A then 7 chars per cl value
        if (
            line[:7] == "       "
        ):  # If we did all the alphas and now that line starts with a bunch of spaces.
            break
        cd_alphas.append(float(line[:7]))

        for i, mach in enumerate(cd_mach_nums):
            index_beg = i * 7 + 7
            index_end = (i + 1) * 7 + 7
            cd_values[mach].append(float(line[index_beg:index_end]))
        line = c81_fid.readline()

    if cl_alphas != cd_alphas:  # if we have different  lists of alphas
        raise ValueError(
            f"ERROR: in file {polar_file}, The alphas in the Cl polar do not match the alphas in the CD polar. We have {cl_alphas} Cls and {cd_alphas} Cds"
        )

    # We also have the moment informatiomn in a c81 file but we ignore that for our purposes.

    return cl_alphas, cl_mach_nums, cl_values, cd_values


###############################################################################################################
def read_in_c81_polar_csv(polar_file):
    """
    # read in the c81 format polar file as a csv file
    # the script checks that the list of Alphas is consistent across CL and CD and that the number of Machs is also consistent across Cl and CD.
    Parameters
    ----------
    polarFile

    Returns
    -------
     4 lists of floats: cl_alphas, cl_mach_nums, cl_values, cd_values
    """

    cl_alphas = []
    cd_alphas = []
    cl_values = {}  # dictionary of list with the machs as keys
    cd_values = {}  # dictionary of list with the machs as keys

    c81_fid = open(polar_file, "r")
    c81_fid.readline()  # skip the header
    line = c81_fid.readline()
    cl_mach_nums = line.split(",")
    cl_mach_nums = [
        float(i.strip()) for i in cl_mach_nums if i
    ]  # remove empty items and trailing \n in cl_mach_nums
    # numClMachs=len(clMachNums) #number of machs we have
    for mach in cl_mach_nums:
        cl_values[mach] = []
    line = c81_fid.readline()
    while True:
        values = line.split(",")
        if values[0] == "":  # If we did all the alphas
            break
        cl_alphas.append(float(values[0]))
        for i, mach in enumerate(cl_mach_nums):
            cl_values[mach].append(float(values[i + 1].strip()))
        line = c81_fid.readline()

    # Now do the CDs
    cd_mach_nums = line.split(
        ","
    )  # we already read the mach numbers line in the while loop above. so just split it.
    cd_mach_nums = [
        float(i.strip()) for i in cd_mach_nums if i
    ]  # remove empty items and trailing \n in cl_mach_nums
    if cl_mach_nums != cd_mach_nums:  # if we have different lists of machs
        raise ValueError(
            f"ERROR: in file {polar_file}, The machs in the Cl polar do not match the machs in the CD polar, we have {cl_mach_nums} Cl mach values and {cd_mach_nums} CD mach values:"
        )

    for mach in cd_mach_nums:
        cd_values[mach] = []
    line = c81_fid.readline()
    while True:
        values = line.split(",")
        if values[0] == "":  # If we did all the alphas
            break
        cd_alphas.append(float(values[0]))
        for i, mach in enumerate(cd_mach_nums):
            cd_values[mach].append(float(values[i + 1].strip()))
        line = c81_fid.readline()

    if cl_alphas != cd_alphas:  # if we have different  lists of alphas
        raise ValueError(
            f"ERROR: in file {polar_file}, The alphas in the Cl polar do not match the alphas in the CD polar. We have {len(cl_alphas)} Cls and {len(cd_alphas)} Cds"
        )

    # We also have the moment information in a c81 file but we ignore that for our purposes.
    if (
        cl_alphas[0] != -180 and cl_alphas[-1] != 180
    ):  # if we don't have polars for the full circle of alpha angles.
        blend_polars_to_flat_plate(cl_alphas, cl_mach_nums, cl_values, cd_values)
    c81_fid.close()
    return cl_alphas, cl_mach_nums, cl_values, cd_values


###############################################################################################################
def read_in_xfoil_data(bet_disk, xfoil_polar_files):
    """
    This function reads in the Xfoil polars and assigns the resulting values correctly into the BET disk dictionary
    Parameters
    ----------
    bet_disk - Dictionary of values needed for the BET disk implementation
    xfoil_polar_files - list of xfoil polar files

    Returns
    -------
    bet_disk - same dictionary as was passed to function but with all the polar information added.
    """
    if len(xfoil_polar_files) != len(bet_disk["sectional_radiuses"]):
        raise ValueError(
            f'Error: There is an error in the number of polar files ({len(xfoil_polar_files)}) vs the number of sectional Radiuses ({len(bet_disk["sectionalRadiuses"])})'
        )

    bet_disk["sectional_polars"] = []
    bet_disk["mach_numbers"] = []

    mach_numbers = []

    for sec_idx, section in enumerate(bet_disk["sectional_radiuses"]):
        secpol = (
            {}
        )  # temporary dict to store all the section polars before assigning it to the right location.
        secpol["lift_coeffs"] = []
        secpol["drag_coeffs"] = []

        polar_files = xfoil_polar_files[sec_idx]
        mach_numbers_for_section = []
        for polar_file in polar_files:
            print(f"doing sectional_radius {section} with polar file {polar_file}")
            if not path.isfile(polar_file):
                raise ValueError(f"Error: xfoil format polar file {polar_file} does not exist.")
            alpha_list, mach_num, cl_values, cd_values = read_in_xfoil_polar(
                polar_file
            )  # read in xfoil data and use flat plate values outside of given polar range
            mach_numbers_for_section.append(float(mach_num))
            secpol["lift_coeffs"].append([cl_values])
            secpol["drag_coeffs"].append([cd_values])
        mach_numbers.append(mach_numbers_for_section)
        bet_disk["sectional_polars"].append(secpol)
    for i in range(
        len(mach_numbers) - 1
    ):  # check to make sure all N cross sections have the same list of mach numbers
        if mach_numbers[i] != mach_numbers[i + 1]:
            raise ValueError(
                f'ERROR: the mach numbers from the Xfoil polars need to be the same set for each cross section. Here sections {i} \
                    and {i+1} have the following sets of mach numbers:{secpol["mach_numbers"][i]} and {secpol["mach_numbers"][i+1]}'
            )
    bet_disk["alphas"] = alpha_list
    bet_disk["mach_numbers"] = mach_numbers[
        0
    ]  # they should all be the same set so just pick the first one.
    #    betDisk['sectionalPolars'].append(secpol)

    return bet_disk


###############################################################################################################
def read_in_c81_polars(bet_disk, c81_polar_files):
    """
    This function reads in the C81 polars and assigns the resulting values correctly into the BET disk dictionary
    Parameters
    ----------
    bet_disk - Dictionary of values needed for the BET disk implementation
    c81_polar_files - list of C81 polar files

    Returns
    -------
    bet_disk - same dictionary as was passed to function but with all the polar information added.
    """
    if len(c81_polar_files) != len(bet_disk["sectional_radiuses"]):
        raise ValueError(
            f'Error: There is an error in the number of polar files ({len(c81_polar_files)}) vs the number of sectional Radiuses ({len(bet_disk["sectionalRadiuses"])})'
        )

    bet_disk["sectional_polars"] = []
    for sec_idx, section in enumerate(bet_disk["sectional_radiuses"]):
        polar_file = c81_polar_files[sec_idx][0]  # Take the first element of that list.
        print(f"doing sectional_radius {section} with polar file {polar_file}")
        if not path.isfile(polar_file):
            raise ValueError(f"Error: c81 format polar file {polar_file} does not exist.")

        if "csv" in polar_file:  # if we are dealing with a csv file
            alpha_list, mach_list, cl_list, cd_list = read_in_c81_polar_csv(polar_file)

        else:
            # we are dealing with a genuine c81 file, then I need to handle it by splitting the list into certain sizes
            alpha_list, mach_list, cl_list, cd_list = read_in_c81_polar_c81_format(polar_file)
        if "mach_numbers" in bet_disk.keys() and bet_disk["mach_numbers"] != mach_list:
            raise ValueError(
                "ERROR: The mach Numbers do not match across the various sectional radi polar c81 files. All the sectional radi need to have the same mach Numbers across all c81 polar files"
            )
        if "alphas" in bet_disk.keys() and bet_disk["alphas"] != alpha_list:
            raise ValueError(
                "ERROR: The alphas do not match across the various sectional radi polar c81 files. All the sectional radi need to have the same alphas across all c81 polar files"
            )

        bet_disk["mach_numbers"] = mach_list
        bet_disk["alphas"] = alpha_list

        # since the order of brackets is Mach#, Rey#, Values then we need to return:
        # [[[array for MAch #1]],[[array for MAch #2]],[[array for MAch #3]],[[array for MAch #4]],......]

        secpol = {}
        secpol["lift_coeffs"] = []
        secpol["drag_coeffs"] = []
        for mach in bet_disk["mach_numbers"]:
            secpol["lift_coeffs"].append([cl_list[mach]])
            secpol["drag_coeffs"].append([cd_list[mach]])
        bet_disk["sectional_polars"].append(secpol)

    return bet_disk


########################################################################################################################
def generate_xfoil_bet_json(
    geometry_file_name,
    rotation_direction_rule,
    initial_blade_direction,
    blade_line_chord,
    omega,
    chord_ref,
    n_loading_nodes,
    cylinder,
    number_of_blades,
    angle_unit,
    length_unit,
):
    """
    This function takes in a geometry input files along with the remaining required information and creates a flow360 BET input dictionary
    This geometry input file contains the list of C81 files required to get the polars along with the geometry twist and chord definition
    Attributes
    ----------
    geometry_file_name: string, filepath to the geometry files we want to translate into a BET disk
    bet_disk: dictionary of the required betdisk data that we can't get form the geometry file.
    return: dictionary that we should append to the Flow360.json file we want to run with.
    """

    bet_disk = {}

    twist_vec, chord_vec, sectional_radiuses, xfoil_polar_file_list = parse_geometry_file(
        geometry_file_name, length_unit=length_unit, angle_unit=angle_unit
    )
    bet_disk["entities"] = cylinder
    bet_disk["omega"] = omega
    bet_disk["chord_ref"] = chord_ref
    bet_disk["n_loading_nodes"] = n_loading_nodes
    bet_disk["rotation_direction_rule"] = rotation_direction_rule
    bet_disk["initial_blade_direction"] = initial_blade_direction
    bet_disk["blade_line_chord"] = blade_line_chord
    bet_disk["number_of_blades"] = number_of_blades
    bet_disk["radius"] = sectional_radiuses[-1]
    bet_disk["sectional_radiuses"] = sectional_radiuses
    bet_disk["twists"] = twist_vec
    bet_disk["chords"] = chord_vec
    bet_disk = read_in_xfoil_data(
        bet_disk, xfoil_polar_file_list
    )  # add the mach values along with the polars from the xfoil files.
    bet_disk["reynolds_numbers"] = generate_reynolds()
    bet_disk["alphas"] *= angle_unit
    bet_disk["sectional_radiuses"] *= length_unit
    bet_disk.pop("radius", None)

    return bet_disk


########################################################################################################################
def parse_geometry_file(geometry_file_name, length_unit, angle_unit):
    """
    This function reads in the geometry file. This file is a csv containing the filenames of the polar definition files along with the twist and chord definitions.
    it assumes the following format:

    #Radial station Sectional Radius (grid Units), polar definition file.
    If it is a C81 polar format, all the mach numbers are in the same file, hence 1 file per section.
    If it is a Xfoil polar format, we need multiple file per section if we want to cover multiple machs
    number,filenameM1.csv,filenameM2.csv...
    number2,filename2M1.csv,filename2M2.csv,...
    number3,filename3M1.csv,filename3M2.csv,...
    number4,filename4M1.csv,filename4M2.csv,...
    .....
    #Radial Station (grid units),Chord(meshUnits),"twist(deg)from rotation plane (0 is parallel to rotation plane,  i.e. perpendicular to thrust)"
    number,number,number
    number,number,number
    number,number,number
    .....

    Parameters
    ----------
    geometry_file_name - path to the geometryFile. This file is a csv containing the filenames of the polar definition files along with the twist and chord definitions.

    Returns
    -------
    4 lists: twist_vec, chord_vec, sectional_radiuses, c81_polar_files
    """
    #    read in the geometry file name and return its values
    fid = open(geometry_file_name)
    line = fid.readline()
    if "#" not in line:
        raise ValueError(
            f"ERROR: first character of first line of geometry file {geometry_file_name} should be the # character to denote a header line"
        )

    geometry_file_path = os.path.dirname(os.path.realpath(geometry_file_name))
    sectional_radiuses = []
    polar_files = []
    radius_station = []
    chord = []
    twist = []
    line = fid.readline().strip("\n")
    while True:
        if (
            "#" in line
        ):  # If we have reached the end of sectional radiuses then move on to the twists and chords
            break
        try:
            split_line = line.split(",")
            sectional_radiuses.append(float(split_line[0]))
            polar_files.append(
                [os.path.join(geometry_file_path, file.strip()) for file in split_line[1:]]
            )
            # polarFiles = [x.strip(' ') for x in polarFiles] # remove spaces in file names
            line = fid.readline().strip("\n")  # read next line.
        except Exception as e:
            raise ValueError(
                f"ERROR: exception thrown when parsing line {line} from geometry file {geometry_file_name}"
            )

    while True:
        try:
            line = fid.readline().strip(
                "\n"
            )  # read in the first line of twist and chord definition
            if not line:  # if we have reached the end of the file.
                break
            radius_station.append(float(line.split(",")[0]))
            chord.append(float(line.split(",")[1]))
            twist.append(float(line.split(",")[2]))
        except:
            raise ValueError(
                f"ERROR: exception thrown when parsing line {line} from geometry file {geometry_file_name}"
            )

    # intialize chord and twist with 0,0 value at centerline
    chord_vec = [{"radius": 0.0 * length_unit, "chord": 0.0 * length_unit}]
    twist_vec = [{"radius": 0.0 * length_unit, "twist": 0.0 * angle_unit}]
    for i in range(len(radius_station)):
        twist_vec.append(
            {"radius": radius_station[i] * length_unit, "twist": twist[i] * angle_unit}
        )
        chord_vec.append(
            {"radius": radius_station[i] * length_unit, "chord": chord[i] * length_unit}
        )

    fid.close()

    return twist_vec, chord_vec, sectional_radiuses, polar_files


################################################################################################################
def generate_c81_bet_json(
    geometry_file_name,
    rotation_direction_rule,
    initial_blade_direction,
    blade_line_chord,
    omega,
    chord_ref,
    n_loading_nodes,
    cylinder,
    angle_unit,
    length_unit,
    number_of_blades,
):
    """
    This function takes in a geometry input files along with the remaining required information and creates a flow360 BET input dictionary
    This geometry input file contains the list of C81 files required to get the polars along with the geometry twist and chord definition
    Attributes
    ----------
    geometry_file_name: string, filepath to the geometry files we want to translate into a BET disk
    bet_disk: dictionary of the required betdisk data that we can't get form the geometry file.
    return: dictionary that we should append to the Flow360.json file we want to run with.
    """

    # if betDisk["rotationDirectionRule"] not in ["rightHand", "leftHand"]:
    #     raise ValueError(
    #         f'Exiting. Invalid rotationDirectionRule: {betDisk["rotationDirectionRule"]}'
    #     )
    # if len(betDisk["axisOfRotation"]) != 3:
    #     raise ValueError(f"axisOfRotation must be a list of size 3. Exiting.")
    # if len(betDisk["centerOfRotation"]) != 3:
    #     raise ValueError("centerOfRotation must be a list of size 3. Exiting")

    twist_vec, chord_vec, sectional_radiuses, c81_polar_file_list = parse_geometry_file(
        geometry_file_name=geometry_file_name, length_unit=length_unit, angle_unit=angle_unit
    )

    bet_disk = {}
    bet_disk["entities"] = cylinder
    bet_disk["omega"] = omega
    bet_disk["chord_ref"] = chord_ref
    bet_disk["n_loading_nodes"] = n_loading_nodes
    bet_disk["rotation_direction_rule"] = rotation_direction_rule
    bet_disk["initial_blade_direction"] = initial_blade_direction
    bet_disk["blade_line_chord"] = blade_line_chord
    bet_disk["number_of_blades"] = number_of_blades
    bet_disk["radius"] = sectional_radiuses[-1]
    bet_disk["sectional_radiuses"] = sectional_radiuses
    bet_disk["twists"] = twist_vec
    bet_disk["chords"] = chord_vec
    bet_disk = read_in_c81_polars(
        bet_disk, c81_polar_file_list
    )  # add the mach values along with the polars from the c81 files.
    bet_disk["reynolds_numbers"] = generate_reynolds()
    bet_disk["alphas"] *= angle_unit
    bet_disk["sectional_radiuses"] *= length_unit
    bet_disk.pop("radius", None)

    return bet_disk


########################################################################################################################
def check_comment(comment_line, line_num, numelts):
    """
    This function is used when reading an XROTOR input file to make sure that what should be comments, really are.

    Attributes
    ----------
    comment_line: string
    numelts: int
    """
    if not comment_line:  # if the comment_line is empty
        return

    # otherwise make sure that we are on a comment line
    if not comment_line[0] == "!" and not (len(comment_line) == numelts):
        raise ValueError(f"wrong format for line #%i: {comment_line}" % (line_num))


########################################################################################################################
def check_num_values(values_list, line_num, numelts):
    """
    This function is used to make sure we have the expected number of inputs in a given line

    Attributes
    ----------
    values: list
    numelts:  int
    return: None, it raises an exception if the error condition is met.
    """
    # make sure that we have the expected number of values
    if not (len(values_list) == numelts):
        raise ValueError(
            f"wrong number of items for line #%i: {values_list}. We were expecting %i numbers and got %i"
            % (line_num, len(values_list), numelts)
        )


########################################################################################################################
def read_dfdc_file(dfdc_file_name):
    """
    This functions read in the dfdc filename provided.
    it does rudimentary checks to make sure the file is truly in the dfdc format.


    Attributes
    ----------
    dfdc_file_name: string
    return: a dictionary with all the required values. That dictionary will be used to create BETDisks section of the
            Flow360 input JSON file.



    Description of the DFDC input File
    ----------------------------------
    The dfdc input file has the following format:
    Case run definition:
    rho :air density (dimensional: kg/m3)
    vso aka cinf : speed of sound ( dimensional: m/s)
        !   RMU         Fluid dynamic viscosity  (dimensioned)
        !   VSO         Fluid speed of sound     (dimensioned)
        !   VEL         Flight speed             (dimensioned)
    rmu: Fluid dynamic viscosity  (dimensional (kg/ms) standard air:1.789E-05)
    Alt: Altitude for fluid properties (km),  999.0 if not defined
    Vel: flight speed dimensional (m/s)
    xi0:Blade root radial coordinate value (dimensional (m))
    xiw: hub wake displacement radius (unused)
    nAeroSections aka naero: number of AERO sections the blade is defined by, NOT TO BE CONFUSED WITH nGeomStations (AKA II) WHICH DEFINE THE BLADE GEOMETRY
    dfdcInputDict stores all the blade sectional information as lists of nsection elements
    rRsection: r/R location of this blade section
    Aerodynamic definition of the blade section at xiaero
        a0deg: angle of zero lift in degrees
        dclda: Incompressible 2-d lift curve slope in radians
        clmax: Max cl after which we use the post stall dc/dalfa (aka dclda_stall)
        clmin: Min cl before which we use the negative alpha stall dc/dalfa (aka dclda_stall)

        dclda_stall: 2-d lift curve slope at stall
        dcl_stall: cl increment, onset to full stall
        cmconst: constant Incompressible 2-d pitching moment
        mcrit: critical Mach #
        cdmin: Minimum drag coefficient value
        cldmin: Lift at minimum drag coefficient value
        dcddcl2: Parabolic drag param d(Cd)/dcl^2
        reyref: reference Reynold's number
        reyexp: Reynold's number exponent Cd~Re^rexp

    n_geom_stations: number of geometric stations where the blade geometry is defined at
    n_blades: number of blades on the propeller
    Each geometry station will have the following parameters:
      r: station r in meters
      c: local chord in meters
      beta0deg: Twist relative to disk plane. ie symmetric 2D section at beta0Deg would create 0 thrust, more beta0deg means more local angle of attack for the blade
      Ubody: (unused) Nacelle perturbation axial  velocity


    """
    with open(dfdc_file_name, "r") as fid:

        # read in lines 5->8 which contains the run case information
        dfdc_input_dict = {}
        line_num = 0  # counter needed to report which line the error is on.
        for i in range(4):
            fid.readline()  # we have 4 blank lines
            line_num += 1
        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 4)
        values = fid.readline().split()
        line_num += 1
        # we don't want ot apply the check to this line b/c we could have a varying number of RPMs

        dfdc_input_dict["vel"] = float(values[1])
        dfdc_input_dict["RPM"] = float(values[2])

        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 5)
        values = fid.readline().split()
        line_num += 1
        check_num_values(values, line_num, 4)
        dfdc_input_dict["rho"] = float(values[0])

        for i in range(7):
            fid.readline()  # skip next 8 lines.
            line_num += 1

        comment_line = fid.readline().upper().split()  # convert all to upper case
        line_num += 1
        check_comment(comment_line, line_num, 2)  # 2 because line should have 2 components
        values = fid.readline().split()
        line_num += 1
        check_num_values(values, line_num, 1)  # we should have 1 value.
        dfdc_input_dict["nAeroSections"] = int(values[0])
        # define the lists with the right number of elements
        dfdc_input_dict["rRstations"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["a0deg"] = [0] * dfdc_input_dict["nAeroSections"]  # WARNING, ao is in deg
        dfdc_input_dict["dclda"] = [0] * dfdc_input_dict[
            "nAeroSections"
        ]  # but dclda is in cl per radians
        dfdc_input_dict["clmax"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["clmin"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["dcldastall"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["dclstall"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["mcrit"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["cdmin"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["clcdmin"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["dcddcl2"] = [0] * dfdc_input_dict["nAeroSections"]

        comment_line = fid.readline().upper().split()  # convert all to upper case
        line_num += 1
        check_comment(comment_line, line_num, 2)  # 2 because line should have 2 components
        for i in range(dfdc_input_dict["nAeroSections"]):  # iterate over all the sections

            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 1)  # we should have 1 value.
            dfdc_input_dict["rRstations"][i] = float(values[0])  # aka xisection

            comment_line = fid.readline().upper().split()  # convert all to upper case
            line_num += 1
            check_comment(comment_line, line_num, 5)  # 5 because line should have 5 components
            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 4)  # we should have 4 value.
            dfdc_input_dict["a0deg"][i] = float(values[0])  # WARNING, ao is in deg
            dfdc_input_dict["dclda"][i] = float(values[1])  # but dclda is in cl per radians
            dfdc_input_dict["clmax"][i] = float(values[2])
            dfdc_input_dict["clmin"][i] = float(values[3])

            comment_line = fid.readline().upper().split()  # convert all to upper case
            line_num += 1
            check_comment(comment_line, line_num, 5)  # 5 because line should have 5 components
            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 4)  # we should have 4 value.
            dfdc_input_dict["dcldastall"][i] = float(values[0])
            dfdc_input_dict["dclstall"][i] = float(values[1])
            dfdc_input_dict["mcrit"][i] = float(values[3])

            comment_line = fid.readline().upper().split()  # convert all to upper case
            line_num += 1
            check_comment(comment_line, line_num, 4)  # 4 because line should have 4 components
            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 3)  # we should have 3 value.
            dfdc_input_dict["cdmin"][i] = float(values[0])
            dfdc_input_dict["clcdmin"][i] = float(values[1])
            dfdc_input_dict["dcddcl2"][i] = float(values[2])

            for i in range(2):
                fid.readline()  # skip next 3 lines.
                line_num += 1

        for i in range(3):
            fid.readline()  # skip next 3 lines.
            line_num += 1
        # Now we are done with the various aero sections and we start looking at blade geometry definitions
        comment_line = fid.readline().upper().split()  # convert all to upper case
        line_num += 1
        check_comment(comment_line, line_num, 3)  # 3 because line should have 3 components
        values = fid.readline().split()
        line_num += 1
        check_num_values(values, line_num, 3)  # we should have 3 values.
        dfdc_input_dict["nBlades"] = int(values[1])
        comment_line = fid.readline().upper().split()  # convert all to upper case
        line_num += 1
        check_comment(comment_line, line_num, 2)
        values = fid.readline().split()
        line_num += 1
        check_num_values(values, line_num, 1)
        dfdc_input_dict["nGeomStations"] = int(values[0])
        # 2nd value on that  line is the number of blades
        dfdc_input_dict["rRGeom"] = [0] * dfdc_input_dict["nGeomStations"]
        dfdc_input_dict["cRGeom"] = [0] * dfdc_input_dict["nGeomStations"]
        dfdc_input_dict["beta0Deg"] = [0] * dfdc_input_dict["nGeomStations"]
        comment_line = fid.readline().upper().split()  # convert all to upper case
        line_num += 1
        check_comment(comment_line, line_num, 4)  # 4 because line should have 4 components
        for i in range(dfdc_input_dict["nGeomStations"]):  # iterate over all the geometry stations
            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 3)  # we should have 3 values.
            dfdc_input_dict["rRGeom"][i] = float(
                values[0]
            )  # key string is not quite true b/c it is the dimensional radius  but I need a place to store the r locations that matches the Xrotor format
            dfdc_input_dict["cRGeom"][i] = float(
                values[1]
            )  # key string is not quite true b/c it is the dimensional chord but I need a place to store the chord dimensions that matches the Xrotor format
            dfdc_input_dict["beta0Deg"][i] = float(values[2])  # twist values
        if dfdc_input_dict["rRGeom"][0] != 0:  # As per discussion in
            # https://enreal.slack.com/archives/C01PFAJ76FL/p1643652853237749?thread_ts=1643413462.002919&cid=C01PFAJ76FL
            # i need to ensure that the blade coordinates go all the way to r/R=0 and have a 0 chord  90deg twist at r/R=0
            dfdc_input_dict["rRGeom"].insert(0, 0.0)
            dfdc_input_dict["cRGeom"].insert(0, 0.0)
            dfdc_input_dict["beta0Deg"].insert(0, 90.0)
            dfdc_input_dict["nGeomStations"] += 1  # we have added one station.

    # for i in range(dfdcInputDict['nGeomStations']):  # iterate over all the geometry stations to nondimensionalize by radius.
    #     dfdcInputDict['rRGeom'][i] = dfdcInputDict['rRGeom'][i] * dfdcInputDict['rad']  # aka r/R location
    #     dfdcInputDict['cRGeom'][i] = dfdcInputDict['cRGeom'][i] * dfdcInputDict['rad']  # aka r/R location

    dfdc_input_dict["rad"] = dfdc_input_dict["rRGeom"][
        -1
    ]  # radius in m is the last value in the r list
    # calculate Extra values and add them  to the dict
    dfdc_input_dict["omegaDim"] = dfdc_input_dict["RPM"] * pi / 30
    dfdc_input_dict["inputType"] = (
        "dfdc"  # we need to store which file format we are using to handle the r vs r/R situation correctly.
    )
    # Now we are done, we have all the data we need.
    return dfdc_input_dict


########################################################################################################################


########################################################################################################################
def read_xrotor_file(xrotor_file_name):
    """
    This functions read in the Xrotor filename provided.
    it does rudimentary checks to make sure the file is truly in the Xrotor format.


    Attributes
    ----------
    input: xrotor_file_name: string
    returns: a dictionary with all the required values. That dictionary will be used to create BETdisks section of the
            Flow360 input JSON file.


    Xrotor file description
    -----------------------
    The xrotor Input file has the following definitions:
    Case run definition:
    rho :air density (dimensional: kg/m3)
    vso aka cinf : speed of sound ( dimensional: m/s)
        !   RMU         Fluid dynamic viscosity  (dimensioned)
        !   VSO         Fluid speed of sound     (dimensioned)
        !   VEL         Flight speed             (dimensioned)
        !   RAD         Rotor tip radius         (dimensioned)
    rmu: Fluid dynamic viscosity  (dimensional (kg/ms) standard air:1.789E-05)
    Alt: Altitude for fluid properties (km),  999.0 if not defined
    Rad: rotor Tip radius dimensional (m)
    Vel: flight speed dimensional (m/s)
    Adv: Advance ratio (Vel/Vtip) where Vtip = propeller tip speed
    Rake: unused- Winglet/droop type tips. We assume a planar propeller.
    xi0:Blade root radial coordinate value (dimensional (m))
    xiw: hub wake displacement radius (unused)
    nAeroSections aka naero: number of AERO sections the blade is defined by, NOT TO BE CONFUSED WITH nGeomStations (AKA II) WHICH DEFINE THE BLADE GEOMETRY
    xrotorInputDict stores all the blade sectional information as lists of nsection elements
    rRsection: r/R location of this blade section
    Aerodynamic definition of the blade section at xiaero
        a0deg: angle of zero lift in degrees
        dclda: Incompressible 2-d lift curve slope in radians
        clmax: Max cl after which we use the post stall dc/dalfa (aka dclda_stall)
        clmin: Min cl before which we use the negative alpha stall dc/dalfa (aka dclda_stall)

        dclda_stall: 2-d lift curve slope at stall
        dcl_stall: cl increment, onset to full stall
        cmconst: constant Incompressible 2-d pitching moment
        mcrit: critical Mach #
        cdmin: Minimum drag coefficient value
        cldmin: Lift at minimum drag coefficient value
        dcddcl2: Parabolic drag param d(Cd)/dcl^2
        reyref: reference Reynold's number
        reyexp: Reynold's number exponent Cd~Re^rexp

    nGeomStations: number of geometric stations where the blade geometry is defined at
    nBlades: number of blades on the propeller
    Each geometry station will have the following parameters:
      r/R: station r/R
      c/R: local chord divided by radius
      beta0deg: Twist relative to disk plane. ie symmetric 2D section at beta0Deg would create 0 thrust, more beta0deg means more local angle of attack for the blade
      Ubody: (unused) Nacelle perturbation axial  velocity

    """

    try:
        fid = open(xrotor_file_name, "r")
        line_num = 0  # counter needed to know which line we are on for error reporting
        # Top line in the file should start with the XROTOR keywords.
        top_line = fid.readline()
        line_num += 1
        if top_line.find("DFDC") == 0:  # If we are actually doing a DFDC file instead of Xrotor
            fid.close()  # close the file b/c we will reopen it in read_dfdc_file
            return read_dfdc_file(xrotor_file_name)

        elif top_line.find("XROTOR") == -1:
            raise ValueError("This input Xrotor file does not seem to be a valid Xrotor input file")

        # read in lines 2->8 which contains the run case information
        xrotor_input_dict = {}

        fid.readline()
        line_num += 1
        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 5)

        values = fid.readline().split()
        line_num += 1
        check_num_values(values, line_num, 4)

        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 5)
        values = fid.readline().split()
        line_num += 1
        check_num_values(values, line_num, 4)
        xrotor_input_dict["rad"] = float(values[0])
        xrotor_input_dict["vel"] = float(values[1])
        xrotor_input_dict["adv"] = float(values[2])

        fid.readline()
        line_num += 1
        fid.readline()
        line_num += 1
        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 2)
        values = fid.readline().split()
        line_num += 1
        check_num_values(values, line_num, 1)

        n_aero_sections = int(values[0])
        # Initialize the dictionary with all the information to re-create the polars at each defining aero section.
        xrotor_input_dict["nAeroSections"] = n_aero_sections
        xrotor_input_dict["rRstations"] = [0] * n_aero_sections
        xrotor_input_dict["a0deg"] = [0] * n_aero_sections
        xrotor_input_dict["dclda"] = [0] * n_aero_sections
        xrotor_input_dict["clmax"] = [0] * n_aero_sections
        xrotor_input_dict["clmin"] = [0] * n_aero_sections
        xrotor_input_dict["dcldastall"] = [0] * n_aero_sections
        xrotor_input_dict["dclstall"] = [0] * n_aero_sections
        xrotor_input_dict["mcrit"] = [0] * n_aero_sections
        xrotor_input_dict["cdmin"] = [0] * n_aero_sections
        xrotor_input_dict["clcdmin"] = [0] * n_aero_sections
        xrotor_input_dict["dcddcl2"] = [0] * n_aero_sections

        for i in range(
            n_aero_sections
        ):  # loop ever each aero section and populate the required variables.
            comment_line = fid.readline().upper().split()
            line_num += 1
            check_comment(comment_line, line_num, 2)
            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 1)
            xrotor_input_dict["rRstations"][i] = float(values[0])

            comment_line = fid.readline().upper().split()
            line_num += 1
            check_comment(comment_line, line_num, 5)
            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 4)
            xrotor_input_dict["a0deg"][i] = float(values[0])
            xrotor_input_dict["dclda"][i] = float(values[1])
            xrotor_input_dict["clmax"][i] = float(values[2])
            xrotor_input_dict["clmin"][i] = float(values[3])

            comment_line = fid.readline().upper().split()
            line_num += 1
            check_comment(comment_line, line_num, 5)
            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 4)
            xrotor_input_dict["dcldastall"][i] = float(values[0])
            xrotor_input_dict["dclstall"][i] = float(values[1])
            xrotor_input_dict["mcrit"][i] = float(values[3])

            comment_line = fid.readline().upper().split()
            line_num += 1
            check_comment(comment_line, line_num, 4)
            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 3)
            xrotor_input_dict["cdmin"][i] = float(values[0])
            xrotor_input_dict["clcdmin"][i] = float(values[1])
            xrotor_input_dict["dcddcl2"][i] = float(values[2])

            comment_line = fid.readline().upper().split()
            line_num += 1
            check_comment(comment_line, line_num, 3)
            values = fid.readline().split()
            line_num += 1
        # skip the duct information
        fid.readline()
        line_num += 1
        fid.readline()
        line_num += 1

        # Now we are done with the various aero sections and we start
        # looking at blade geometry definitions
        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 3)
        values = fid.readline().split()
        line_num += 1
        check_num_values(values, line_num, 2)

        n_geom_stations = int(values[0])
        xrotor_input_dict["nGeomStations"] = n_geom_stations
        xrotor_input_dict["nBlades"] = int(values[1])
        xrotor_input_dict["rRGeom"] = [0] * n_geom_stations
        xrotor_input_dict["cRGeom"] = [0] * n_geom_stations
        xrotor_input_dict["beta0Deg"] = [0] * n_geom_stations

        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 5)

        # iterate over all the geometry stations
        for i in range(n_geom_stations):

            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 4)
            xrotor_input_dict["rRGeom"][i] = float(values[0])
            xrotor_input_dict["cRGeom"][i] = float(values[1])
            xrotor_input_dict["beta0Deg"][i] = float(values[2])

    finally:  # We are done reading
        fid.close()

    # Set the twist at the root to be 90 so that it is continuous on
    # either side of the origin. I.e Across blades' root. Also set
    # the chord to be 0 at the root
    if xrotor_input_dict["rRGeom"][0] != 0:
        xrotor_input_dict["rRGeom"].insert(0, 0.0)
        xrotor_input_dict["cRGeom"].insert(0, 0.0)
        xrotor_input_dict["beta0Deg"].insert(0, 90.0)
        xrotor_input_dict["nGeomStations"] += 1

    # AdvanceRatio = Vinf/Vtip => Vinf/OmegaR
    xrotor_input_dict["omegaDim"] = xrotor_input_dict["vel"] / (
        xrotor_input_dict["adv"] * xrotor_input_dict["rad"]
    )
    xrotor_input_dict["RPM"] = xrotor_input_dict["omegaDim"] * 30 / pi
    xrotor_input_dict["inputType"] = (
        "xrotor"  # we need to store which file format we are using to handle the r vs r/R situation correctly.
    )
    return xrotor_input_dict


def float_range(start, stop, step=1):
    return [float(a) for a in range(start, stop, step)]


########################################################################################################################
def generate_twists(xrotor_dict, mesh_unit, length_unit, angle_unit):
    """
    Transform the Xrotor format blade twists distribution into the Flow360 standard.

    Attributes
    ----------
    xrotor_dict: dictionary of Xrotor data as read in by def readXROTORFile(xrotorFileName):
    mesh_unit: float,  Grid unit length in the mesh.
    return:  list of dictionaries containing the radius ( in grid units) and twist in degrees.
    """
    # generate the twists vector required from the BET input
    twist_vec = []
    if xrotor_dict["inputType"] == "xrotor":
        multiplier = xrotor_dict[
            "rad"
        ]  # X rotor uses r/R we need to convert that to r in mesh units
    elif xrotor_dict["inputType"] == "dfdc":
        multiplier = 1.0  # dfdc is already in meters so only need to convert it ot mesh units.

    for i in range(xrotor_dict["nGeomStations"]):
        # dimensional radius we are at in grid unit
        r = xrotor_dict["rRGeom"][i] * multiplier * u.m / mesh_unit
        twist = xrotor_dict["beta0Deg"][i]
        twist_vec.append({"radius": r * length_unit, "twist": twist * angle_unit})

    return twist_vec


########################################################################################################################
def generate_chords(xrotor_dict, mesh_unit, length_unit):
    """
    Transform the Xrotor format blade chords distribution into the Flow360 standard.

    Attributes
    ----------
    xrotor_dict: dictionary of Xrotor data as read in by def read_xrotor_file(xrotor_file_name):
    mesh_unit: float,  Grid unit length per meter in the mesh. if your grid is in mm then mesh_unit = 0.001 meter per mm;
    If your grid is in inches then mesh_nit = 0.0254 meter per in etc...
    return:  list of dictionaries containing the radius ( in grid units) and chords in grid units.
    """
    # generate the dimensional chord vector required from the BET input
    chord_vec = []
    if xrotor_dict["inputType"] == "xrotor":
        multiplier = xrotor_dict[
            "rad"
        ]  # X rotor uses r/R we need to convert that to r in mesh units
    elif xrotor_dict["inputType"] == "dfdc":
        multiplier = 1.0  # dfdc is already in meters so only need to convert it ot mesh units.
    for i in range(xrotor_dict["nGeomStations"]):
        r = xrotor_dict["rRGeom"][i] * multiplier * u.m / mesh_unit
        chord = xrotor_dict["cRGeom"][i] * multiplier * u.m / mesh_unit
        chord_vec.append({"radius": r * length_unit, "chord": chord * length_unit})

    return chord_vec


########################################################################################################################
def generate_machs():
    """
    The Flow360 BET input file expects a set of Mach numbers to interpolate
    between using the Mach number the blade sees.
    To that end we will generate 4 different tables at 4 different Mach #s
    equivalent to M^2=0, 1/3, 2/3, 0.9


    Attributes
    ----------
    return: list of floats
    """

    mach_vec = [0, sqrt(1 / 3), sqrt(2 / 3), sqrt(0.9)]
    return mach_vec


########################################################################################################################
def generate_reynolds():
    """
    Flow360 has the functionality to interpolate across Reynolds numbers but we are not using that functionality
    just make it a constant 1

    """
    return [1]


########################################################################################################################
def generate_alphas():
    """
    Generate the list of Alphas that the BET 2d section polar is for in 1 degree steps from -180 to 180
    return: list of floats
    """
    # generate the list of Alphas that the 2d section polar is for:

    # option 1:
    # 10 deg steps from -180 ->-30 and from 30 to 180. 1 deg steps from -29 to 29
    # negAng = list(arange(-30, -5, 1).astype(float))
    # posAng = list(arange(-5, 10, 1).astype(float))
    # posAng2 = list(arange(10, 29, 1).astype(float))
    # return list(arange(-180, -30, 10).astype(float)) + negAng + posAng + posAng2 + list(arange(30, 190, 10).astype(float))  # json doesn't like the numpy default int64 type so I make it a float

    # option 2: return every degree with a refinement of every 1/2 degree between -10 and 10
    neg_ang = float_range(-180, -9)
    pos_ang = [
        -9,
        -8.5,
        -8,
        -7.5,
        -7,
        -6.5,
        -6,
        -5.5,
        -5,
        -4.5,
        -4,
        -3.5,
        -3,
        -2.5,
        -2,
        -1.5,
        -1,
        -0.75,
        -0.5,
        -0.25,
        0,
        0.25,
        0.5,
        0.75,
        1,
        1.25,
        1.5,
        1.75,
        2,
        2.25,
        2.5,
        2.75,
        3,
        3.5,
        4,
        4.5,
        5,
        5.5,
        6,
        6.5,
        7,
        7.5,
        8,
        8.5,
        9,
    ]
    pos_ang_2 = float_range(10, 181)
    return neg_ang + pos_ang + pos_ang_2
    # return floatRange(-180, 181)


########################################################################################################################
def find_cl_min_max_alphas(c_lift, cl_min, cl_max):
    """
    Find the index in the c_lift list where we are just below the cl_min
    value and the one where we are just above the cl_max value. Use the fact that CL should be continually increasing
    from -pi -> Pi radians.
    The goal of this function is to separate the linear CL regime (i.e. from cl_min to cl_max) and extract its indices
    We Traverse the list from the beginning until we hit cl_min


    Attributes
    ----------

    c_lift: list of floats
    cl_min: float
    cl_max: float
    return: 2 ints as indices
    """

    cl_min_idx = 0  # initialize as the first index
    cl_max_idx = len(c_lift)  # initialize as the last index
    for i in range(len(c_lift)):
        if c_lift[i] < cl_min:
            cl_min_idx = i
        if c_lift[i] > cl_max:
            cl_max_idx = i
            break
    return (
        cl_min_idx - 1,
        cl_max_idx + 1,
    )  # return the two indices right before and after the two found values.


########################################################################################################################
def blend_func_value(blend_window, alpha, alpha_min_max, alpha_range):
    """
    This functions is used to blend the flat plate CL and CD polar to the given Cl and CD polars.
    The returned blend value is 1 when we use the given CL and CD values and 0 when we use the Flat plate values.
    Within the blend_window range of alphas it returns a COS^2 based smooth blend.

    Attributes
    ----------

        blend_window: float size of the window we want to blend from the given 2D polar
        alpha: float alpha we are at in radians
        alpha_min_max: float,   alpha min  or alpha max for that 2D polar in radians. Outside of those values we use
    the Flat plate coefficients
        alpha_range: string, used to figure out whether we are doing before CLmin or beyond CLmax
        return: float (blend value for that alpha
    """

    if "above_cl_max" in alpha_range:
        # we are on the cl_max side:
        if alpha < alpha_min_max:
            return 1
        if alpha > alpha_min_max + blend_window:
            return 0
        return cos((alpha - alpha_min_max) / blend_window * pi / 2) ** 2
    if "below_cl_min" in alpha_range:
        # we are on the cl_min side:
        if alpha > alpha_min_max:
            return 1
        if alpha < alpha_min_max - blend_window:
            return 0
        return cos((alpha - alpha_min_max) / blend_window * pi / 2) ** 2
    else:
        raise ValueError(
            f"alpha_range must be either above_cl_max or below_cl_min, it is: {alpha_range}"
        )


########################################################################################################################
def xrotor_blend_to_flat_plate(c_lift, c_drag, alphas, alpha_min_idx, alpha_max_idx):
    """
     Blend the c_lift and c_drag values outside of the normal working range of alphas to the flat plate CL and CD values.

    Attributes
    ----------
    c_lift: float
    c_drag: float
    alphas: list of floats
    alpha_min_idx: int, index within the above list of alphas
    alpha_max_idx: int, index within the above list of alphas

    return: 2 Floats representing the blended CL and CD at that alpha
    """

    blend_window = 0.5  # 0.5 radians
    alpha_min = alphas[alpha_min_idx] * pi / 180
    alpha_max = alphas[alpha_max_idx] * pi / 180

    for i in range(alpha_min_idx):  # from -pi to alpha_min in the c_lift array
        a = alphas[i] * pi / 180  # alpha in radians

        blend_val = blend_func_value(
            blend_window, a, alpha_min, "below_cl_min"
        )  # we are on the alpha_cl_min side going up in CL
        # this follows the flat plate lift and drag equations times the blend val coefficient
        c_lift[i] = c_lift[i] * blend_val + (1 - blend_val) * cos(a) * 2 * pi * sin(a) / sqrt(
            1 + (2 * pi * sin(a)) ** 2
        )
        c_drag[i] = (
            c_drag[i] * blend_val
            + (1 - blend_val) * sin(a) * (2 * pi * sin(a)) ** 3 / sqrt(1 + (2 * pi * sin(a)) ** 6)
            + 0.05
        )

    for j in range(alpha_max_idx, len(alphas)):  # from alphaMax to Pi in the c_lift array
        a = alphas[j] * pi / 180  # alpha in radians
        blend_val = blend_func_value(
            blend_window, a, alpha_max, "above_cl_max"
        )  # we are on the alpha_cl_max side of things going up in CL
        # this follows the flat plate lift and drag equations times the blend val coefficient
        c_lift[j] = c_lift[j] * blend_val + (1 - blend_val) * cos(a) * 2 * pi * sin(a) / sqrt(
            1 + (2 * pi * sin(a)) ** 2
        )
        c_drag[j] = (
            c_drag[j] * blend_val
            + (1 - blend_val) * sin(a) * (2 * pi * sin(a)) ** 3 / sqrt(1 + (2 * pi * sin(a)) ** 6)
            + 0.05
        )
    return c_lift, c_drag


########################################################################################################################
def calc_cl_cd(xrotor_dict, alphas, mach_num, nrR_station):
    """

    This function is transcribed from the Xrotor source code. https://web.mit.edu/drela/Public/web/xrotor/
    Use the 2D polar parameters from the Xrotor input file to get the Cl and Cd at the various Alphas and given MachNum

    Calculate compressibility factor taken from xaero.f in xrotor source code
    Factors for compressibility drag model, HHY 10/23/00
    Mcrit is set by user ( ie read in from Xrotor file )
    Effective Mcrit is Mcrit_eff = Mcrit - CLMFACTOR*(CL-CLDmin) - DMDD
    DMDD is the delta Mach to get CD=CDMDD (usually 0.0020)
    Compressible drag is CDC = CDMFACTOR*(Mach-Mcrit_eff)^MEXP
    CDMstall is the drag at which compressible stall begins

    Attributes
    ----------
    xrotor_dict: dictionary of Xrotor data as read in by def readXROTORFile(xrotorFileName):
    alphas: list of ints, alphas we have for the polar.
    mach_num: float, mach number we do this polar at.
    nrR_station: int, which r/R station we have to define this polar for.
    return: 2 list of floats representing the CL and CD for  that polar
    """

    CDMFACTOR = 10.0
    CLMFACTOR = 0.25
    MEXP = 3.0
    CDMDD = 0.0020
    CDMSTALL = 0.1000

    # Prandtl-Glauert compressibility factor
    MSQ = mach_num**2

    if MSQ > 1.0:
        print("CLFUNC: Local Mach^2 number limited to 0.99, was ", MSQ)
        MSQ = 0.99

    PG = 1.0 / sqrt(1.0 - MSQ)
    MACH = mach_num

    # Generate CL from dCL/dAlpha and Prandtl-Glauert scaling
    A_zero = xrotor_dict["a0deg"][nrR_station] * pi / 180
    DCLDA = xrotor_dict["dclda"][nrR_station]

    CLA = [0] * len(alphas)
    for i, a in enumerate(alphas):
        CLA[i] = DCLDA * PG * ((a * pi / 180) - A_zero)
    CLA = array(CLA)

    # Reduce CLmax to match the CL of onset of serious compressible drag
    CLMAX = xrotor_dict["clmax"][nrR_station]
    CLMIN = xrotor_dict["clmin"][nrR_station]
    CLDMIN = xrotor_dict["clcdmin"][nrR_station]
    MCRIT = xrotor_dict["mcrit"][nrR_station]

    DMSTALL = (CDMSTALL / CDMFACTOR) ** (1.0 / MEXP)
    CLMAXM = max(0.0, (MCRIT + DMSTALL - MACH) / CLMFACTOR) + CLDMIN
    CLMAX = min(CLMAX, CLMAXM)
    CLMINM = min(0.0, -(MCRIT + DMSTALL - MACH) / CLMFACTOR) + CLDMIN
    CLMIN = max(CLMIN, CLMINM)

    # CL limiter function (turns on after +-stall)
    DCL_STALL = xrotor_dict["dclstall"][nrR_station]
    ECMAX = expList(clip((CLA - CLMAX) / DCL_STALL, -inf, 200))
    ECMIN = expList(clip((CLA * (-1) + CLMIN) / DCL_STALL, -inf, 200))
    CLLIM = logList((ECMAX + 1.0) / (ECMIN + 1.0)) * DCL_STALL

    # Subtract off a (nearly unity) fraction of the limited CL function
    # This sets the dCL/dAlpha in the stalled regions to 1-FSTALL of that
    # in the linear lift range
    DCLDA_STALL = xrotor_dict["dcldastall"][nrR_station]
    FSTALL = DCLDA_STALL / DCLDA
    CLIFT = CLA - CLLIM * (1.0 - FSTALL)

    # In the basic linear lift range drag is a quadratic function of lift
    # CD = CD0 (constant) + quadratic with CL)
    CDMIN = xrotor_dict["cdmin"][nrR_station]
    DCDCL2 = xrotor_dict["dcddcl2"][nrR_station]

    # Don't do any reynolds number corrections b/c we know it is minimal
    RCORR = 1
    CDRAG = (((CLIFT - CLDMIN) ** 2) * DCDCL2 + CDMIN) * RCORR

    # Post-stall drag added
    FSTALL = DCLDA_STALL / DCLDA
    DCDX = CLLIM * (1.0 - FSTALL) / (PG * DCLDA)
    DCD = (DCDX**2) * 2.0

    # Compressibility drag (accounts for drag rise above Mcrit with CL effects
    # CDC is a function of a scaling factor*(M-Mcrit(CL))**MEXP
    # DMDD is the Mach difference corresponding to CD rise of CDMDD at MCRIT
    DMDD = (CDMDD / CDMFACTOR) ** (1.0 / MEXP)
    CRITMACH = absList(CLIFT - CLDMIN) * CLMFACTOR * (-1) + MCRIT - DMDD
    CDC = array([0 for i in range(len(CRITMACH))])
    for crit_mach_idx in range(len(CRITMACH)):
        if MACH < CRITMACH[crit_mach_idx]:
            continue
        else:
            CDC[crit_mach_idx] = CDMFACTOR * (MACH - CRITMACH[crit_mach_idx]) ** MEXP

    # you could use something like this to add increase drag by Prandtl-Glauert
    # (or any function you choose)
    FAC = 1.0
    # --- Total drag terms
    CDRAG = CDRAG * FAC + DCD + CDC

    # Now we modify the Clift and CDrag outside of the large alpha range to smooth out
    # the Cl and CD outside of the expected operating range

    # Find the Alpha for ClMax and CLMin
    alpha_min_idx, alpha_max_idx = find_cl_min_max_alphas(CLIFT, CLMIN, CLMAX)
    # Blend the CLIFt and CDRAG values from above with the flat plate formulation to
    # be used outside of the alphaCLmin to alphaCLMax window
    CLIFT, CDRAG = xrotor_blend_to_flat_plate(CLIFT, CDRAG, alphas, alpha_min_idx, alpha_max_idx)

    return list(CLIFT), list(CDRAG)


########################################################################################################################
def get_polar(xrotor_dict, alphas, machs, rR_station):
    """
    Return the 2D Cl and CD polar expected by the Flow360 BET model.
    b/c we have 4 Mach Values * 1 Reynolds value we need 4 different arrays per sectional polar as in:
    since the order of brackets is Mach#, Rey#, Values then we need to return:
    [[[array for MAch #1]],[[array for MAch #2]],[[array for MAch #3]],[[array for MAch #4]]]


    Attributes
    ----------
    xrotor_dict: dictionary of Xrotor data as read in by def readXROTORFile(xrotorFileName):
    alphas: list of floats
    machs: list of float
    rR_station: station index.
    return: list of dictionaries
    """

    secpol = {}
    secpol["lift_coeffs"] = []
    secpol["drag_coeffs"] = []
    for mach_num in machs:
        cl, cd = calc_cl_cd(xrotor_dict, alphas, mach_num, rR_station)
        secpol["lift_coeffs"].append([cl])
        secpol["drag_coeffs"].append([cd])
    return secpol


########################################################################################################################
def generate_xrotor_bet_json(
    xrotor_file_name,
    rotation_direction_rule,
    initial_blade_direction,
    blade_line_chord,
    omega,
    chord_ref,
    n_loading_nodes,
    cylinder,
    angle_unit,
    length_unit,
    mesh_unit,
):
    """

    This file takes in an Xrotor or DFDC input file and translates it into a flow360 BET input dictionary

    DFDC and Xrotor come from the same family of CFD codes. They are both written by Mark Drela over at MIT.
    we can use the same translator for both DFDC and Xrotor.

    Attributes
    ----------
    xrotor_file_name: string, filepath to the Xrotor/DFDC file we want to translate into a BETDisk
    bet_disk: This is a dict that already contains some bet_disk definition information. We will add to that same dict
    before returning it.
    mesh_unit is in grid units per meter, if your grid is in mm then mesh_unit = 0.001 grid unit per meter.
     If your grid is in inches then meshUnit = 0.0254 grid Unit per meter etc...
        It should contain the following key value pairs:
        ['axisOfRotation']: [a,b,c],
        ['centerOfRotation']: [x,y,z],
        ['rotationDirectionRule']: "rightHand" or "leftHand",
        ['thickness']: value,
        ['meshUnit']:value,
        ['chordRef']:value,
        ['nLoadingNodes']

    Returns
    -------
    returns: Dictionary that we should append to the Flow360.json file we want to run with.
    """

    # if betDisk["rotationDirectionRule"] not in ["rightHand", "leftHand"]:
    #     raise ValueError(
    #         "Invalid rotationDirectionRule of {}. Exiting.".format(betDisk["rotationDirectionRule"])
    #     )
    # if len(betDisk["axisOfRotation"]) != 3:
    #     raise ValueError(f"axisOfRotation must be a list of size 3. Exiting.")
    # if len(betDisk["centerOfRotation"]) != 3:
    #     raise ValueError("centerOfRotation must be a list of size 3. Exiting")

    xrotor_dict = read_xrotor_file(xrotor_file_name)

    bet_disk = {}
    bet_disk["entities"] = cylinder
    bet_disk["omega"] = omega
    bet_disk["chord_ref"] = chord_ref
    bet_disk["n_loading_nodes"] = n_loading_nodes
    bet_disk["rotation_direction_rule"] = rotation_direction_rule
    bet_disk["initial_blade_direction"] = initial_blade_direction
    bet_disk["blade_line_chord"] = blade_line_chord
    bet_disk["number_of_blades"] = xrotor_dict["nBlades"]
    bet_disk["radius"] = xrotor_dict["rad"] * u.m / mesh_unit
    bet_disk["twists"] = generate_twists(
        xrotor_dict, mesh_unit=mesh_unit, length_unit=length_unit, angle_unit=angle_unit
    )
    bet_disk["chords"] = generate_chords(xrotor_dict, mesh_unit=mesh_unit, length_unit=length_unit)
    bet_disk["mach_numbers"] = generate_machs()
    bet_disk["alphas"] = generate_alphas()
    bet_disk["reynolds_numbers"] = generate_reynolds()
    bet_disk["sectional_radiuses"] = [
        bet_disk["radius"] * r for r in xrotor_dict["rRstations"]
    ] * length_unit
    bet_disk["sectional_polars"] = []

    for secId in range(0, xrotor_dict["nAeroSections"]):
        polar = get_polar(xrotor_dict, bet_disk["alphas"], bet_disk["mach_numbers"], secId)
        bet_disk["sectional_polars"].append(polar)

    bet_disk["alphas"] *= angle_unit
    bet_disk.pop(
        "radius", None
    )  # radius is only needed to get sectional_radiuses but not by the solver.

    # with open("bet_translator_dict.json", "w") as file1:
    #     json.dump(betDisk, file1, indent=4)

    return bet_disk


########################################################################################################################
def test_translator():
    """
    run the translator with a representative set of inputs
    dumps betDisk JSON file that can be added to a Flow360 JSON file.

    meshUnit is in gridUnits per meters, if your grid is in mm then meshUnit = 0.001 grid Unit per meter.
     If your grid is in inches then meshUnit = 0.0254 grid Unit per meter etc...
    """
    bet_disk_dict = {
        "diskThickness": 0.05,
        "meshUnit": 1,
        "chordRef": 1,
        "nLoadingNodes": 20,
        "tipGap": "inf",
        "bladeLineChord": 1,
        "axisOfRotation": [0, 0, 1],
        "centerOfRotation": [0, 0, 0],
        "rotationDirectionRule": "rightHand",
    }

    # initialBladeDirection =  [1, 0, 0]  # Used for time accurate Blade Line simulations
    xrotor_file_name = "examples/xrotorTranslator/ecruzer.prop"

    xrotor_input_dict = generate_xrotor_bet_json(xrotor_file_name, bet_disk_dict)
    bet_disk_json = {
        "BETDisks": [xrotor_input_dict]
    }  # make all that data a subset of BETDisks dictionary, notice the [] b/c
    # the BETDisks dictionary accepts a list of bet disks
    # dump the sample dictionary to a json file
    json.dump(bet_disk_json, open("sampleBETJSON.json", "w"), indent=4)


########################################################################################################################
if __name__ == "__main__":
    # if run on its own, then just run the test_translator() function
    test_translator()
