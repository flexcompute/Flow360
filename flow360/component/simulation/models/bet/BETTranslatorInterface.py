"""Translator for C81, DFDC, Xfoil and XROTOR BET input files."""

import os
from math import *
from os import path

import numpy as np
from scipy.interpolate import interp1d

import flow360.component.simulation.units as u

from .utils import *


def read_in_xfoil_polar(polar_file):
    """
    Read in the xfoil format polar file.

    Parameters
    ----------
    polar_file: path to the xfoil polar file

    Attributes
    ----------
    return: alpha_list, mach_list, cl_list, cd_list
    """
    cl_alphas = []
    cl_values = {}
    cd_values = {}

    xfoil_fid = open(polar_file, "r")
    xfoil_fid.readline()
    for i in range(8):
        line = xfoil_fid.readline()

    mach_num = line.strip().split(" ")[4]
    cl_values[mach_num] = []
    cd_values[mach_num] = []
    for i in range(4):
        line = xfoil_fid.readline()
    while True:
        linecontents = line.strip().split(" ")

        c = linecontents.count(
            ""
        )
        for i in range(c):
            linecontents.remove("")

        cl_alphas.append(float(linecontents[0]))
        cl_values[mach_num].append(float(linecontents[1]))
        cd_values[mach_num].append(float(linecontents[2]))
        line = xfoil_fid.readline()
        if len(line) == 0:
            break

    cl_alphas, cl_mach_nums, cl_values, cd_values = blend_polars_to_flat_plate(
        cl_alphas, [mach_num], cl_values, cd_values
    )

    deg_increment_ang = list(np.arange(-30, 30, 1).astype(float))

    alphas = (
        list(np.arange(-180, -30, 10).astype(float))
        + deg_increment_ang
        + list(np.arange(30, 190, 10).astype(float))
    )

    cl_interp = interp1d(
        cl_alphas, cl_values[cl_mach_nums[0]], kind="linear"
    )
    cd_interp = interp1d(
        cl_alphas, cd_values[cl_mach_nums[0]], kind="linear"
    )
    cls = [0 for i in range(len(alphas))]
    cds = [0 for i in range(len(alphas))]
    for i, alpha in enumerate(alphas):
        cls[i] = float(cl_interp(alpha))
        cds[i] = float(cd_interp(alpha))

    xfoil_fid.close()

    return alphas, cl_mach_nums[0], cls, cds


def blend_polars_to_flat_plate(cl_alphas, cl_mach_nums, cl_values, cd_values):
    """
    Blend a given arbitrary set of CL and CD polars that are missing values to cover the whole -180 to 180
    range of angles. The resulting polars will have the missing values be replaced by the flat plate CL and CD.

    Parameters
    ----------
    cl_alphas: list of alpha angles
    cl_mach_nums: list of mach numbers
    cl_values: dict with dimensions nMach*nAlphas
    cd_values: dict with dimensions nMach*nAlphas

    Attributes
    ----------
    return: cl_alphas, cl_mach_nums, cl_values, cd_values with polars completed to +/- 180
    """

    polar_alpha_step_blend = 10

    alpha_min = cl_alphas[0]
    alpha_max = cl_alphas[-1]
    if alpha_min < -180:
        raise ValueError(f"ERROR: alpha_min is smaller than -180: {alpha_min}")
    if alpha_max > 180:
        raise ValueError(f"ERROR: alpha_max is greater than 180: {alpha_max}")

    blend_window = 0.5

    num_missing_alphas_min = round((alpha_min + 180) / polar_alpha_step_blend)
    num_missing_alphas_max = round((180 - alpha_max) / polar_alpha_step_blend)

    for i in range(num_missing_alphas_min - 1):
        cl_alphas.insert(0, cl_alphas[0] - polar_alpha_step_blend)
        a = cl_alphas[0] * pi / 180
        for i, mach in enumerate(cl_mach_nums):
            blend_val = blend_func_value(
                blend_window, a, alpha_min * pi / 180, "below_cl_min"
            )

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
            cl_values[mach].insert(0, cl)
            cd_values[mach].insert(0, cd)

    for i in range(num_missing_alphas_max - 1):
        cl_alphas.append(cl_alphas[-1] + polar_alpha_step_blend)
        a = cl_alphas[-1] * pi / 180
        for i, mach in enumerate(cl_mach_nums):
            blend_val = blend_func_value(
                blend_window, a, alpha_max * pi / 180, "above_cl_max"
            )

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
            cl_values[mach].append(cl)
            cd_values[mach].append(cd)

    cl_alphas.insert(0, -180)
    cl_alphas.append(180)
    for i, mach in enumerate(cl_mach_nums):
        cl_values[mach].insert(0, 0)
        cd_values[mach].insert(0, 0.05)
        cl_values[mach].append(0)
        cd_values[mach].append(0.05)

    return cl_alphas, cl_mach_nums, cl_values, cd_values


def read_in_c81_polar_c81_format(polar_file):
    """
    Read in the c81 format polar file.
    This function checks that the list of Alphas is consistent across CL and CD and that the number of Machs is also consistent across Cl and CD.

    Parameters
    ----------
    polar_file: path to the c81 polar file

    Attributes
    ----------
    return: tuple of lists
    """
    cl_alphas = []
    cd_alphas = []
    cl_values = {}
    cd_values = {}

    c81_fid = open(polar_file, "r")
    c81_fid.readline()
    line = c81_fid.readline()
    cl_mach_nums = line.strip().split(" ")
    cl_mach_nums = [
        float(i) for i in cl_mach_nums if i
    ]
    for mach in cl_mach_nums:
        cl_values[mach] = []
    line = c81_fid.readline()
    while True:
        if (
            line[:7] == "       "
        ):
            break
        cl_alphas.append(float(line[:7]))

        for i, mach in enumerate(cl_mach_nums):
            index_beg = i * 7 + 7
            index_end = (i + 1) * 7 + 7
            cl_values[mach].append(float(line[index_beg:index_end]))
        line = c81_fid.readline()

    cd_mach_nums = line.strip().split(" ")
    cd_mach_nums = [
        float(i) for i in cd_mach_nums if i
    ]
    if cl_mach_nums != cd_mach_nums:
        raise ValueError(
            f"ERROR: in file {polar_file}, The machs in the Cl polar do not match the machs in the CD polar, we have {cl_mach_nums} Cl mach values and {cd_mach_nums} CD mach values:"
        )

    for mach in cd_mach_nums:
        cd_values[mach] = []
    line = c81_fid.readline()
    while True:
        if (
            line[:7] == "       "
        ):
            break
        cd_alphas.append(float(line[:7]))

        for i, mach in enumerate(cd_mach_nums):
            index_beg = i * 7 + 7
            index_end = (i + 1) * 7 + 7
            cd_values[mach].append(float(line[index_beg:index_end]))
        line = c81_fid.readline()

    if cl_alphas != cd_alphas:
        raise ValueError(
            f"ERROR: in file {polar_file}, The alphas in the Cl polar do not match the alphas in the CD polar. We have {cl_alphas} Cls and {cd_alphas} Cds"
        )


    return cl_alphas, cl_mach_nums, cl_values, cd_values


def read_in_c81_polar_csv(polar_file):
    """
    Read in the c81 format polar file as a csv file.
    Check whether list of Alphas is consistent across CL and CD and that the number of Machs is also consistent across Cl and CD.

    Parameters
    ----------
    polar_file: path to the c81 csv polar file

    Attributes
    ----------
    return: tuple of lists
    """

    cl_alphas = []
    cd_alphas = []
    cl_values = {}
    cd_values = {}

    c81_fid = open(polar_file, "r")
    c81_fid.readline()
    line = c81_fid.readline()
    cl_mach_nums = line.split(",")
    cl_mach_nums = [
        float(i.strip()) for i in cl_mach_nums if i
    ]
    for mach in cl_mach_nums:
        cl_values[mach] = []
    line = c81_fid.readline()
    while True:
        values = line.split(",")
        if values[0] == "":
            break
        cl_alphas.append(float(values[0]))
        for i, mach in enumerate(cl_mach_nums):
            cl_values[mach].append(float(values[i + 1].strip()))
        line = c81_fid.readline()

    cd_mach_nums = line.split(
        ","
    )
    cd_mach_nums = [
        float(i.strip()) for i in cd_mach_nums if i
    ]
    if cl_mach_nums != cd_mach_nums:
        raise ValueError(
            f"ERROR: in file {polar_file}, The machs in the Cl polar do not match the machs in the CD polar, we have {cl_mach_nums} Cl mach values and {cd_mach_nums} CD mach values:"
        )

    for mach in cd_mach_nums:
        cd_values[mach] = []
    line = c81_fid.readline()
    while True:
        values = line.split(",")
        if values[0] == "":
            break
        cd_alphas.append(float(values[0]))
        for i, mach in enumerate(cd_mach_nums):
            cd_values[mach].append(float(values[i + 1].strip()))
        line = c81_fid.readline()

    if cl_alphas != cd_alphas:
        raise ValueError(
            f"ERROR: in file {polar_file}, The alphas in the Cl polar do not match the alphas in the CD polar. We have {len(cl_alphas)} Cls and {len(cd_alphas)} Cds"
        )

    if (
        cl_alphas[0] != -180 and cl_alphas[-1] != 180
    ):
        blend_polars_to_flat_plate(cl_alphas, cl_mach_nums, cl_values, cd_values)
    c81_fid.close()
    return cl_alphas, cl_mach_nums, cl_values, cd_values


def read_in_xfoil_data(bet_disk, xfoil_polar_files):
    """
    Read in the Xfoil polars and assigns the resulting values correctly into the BETDisk dictionary.

    Parameters
    ----------
    bet_disk: dictionary, contains required betdisk data
    xfoil_polar_files: list of xfoil polar files

    Attributes
    ----------
    return: dictionary
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
        )
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
            )
            mach_numbers_for_section.append(float(mach_num))
            secpol["lift_coeffs"].append([cl_values])
            secpol["drag_coeffs"].append([cd_values])
        mach_numbers.append(mach_numbers_for_section)
        bet_disk["sectional_polars"].append(secpol)
    for i in range(
        len(mach_numbers) - 1
    ):
        if mach_numbers[i] != mach_numbers[i + 1]:
            raise ValueError(
                f'ERROR: the mach numbers from the Xfoil polars need to be the same set for each cross section. Here sections {i} \
                    and {i+1} have the following sets of mach numbers:{secpol["mach_numbers"][i]} and {secpol["mach_numbers"][i+1]}'
            )
    bet_disk["alphas"] = alpha_list
    bet_disk["mach_numbers"] = mach_numbers[
        0
    ]

    return bet_disk


def read_in_c81_polars(bet_disk, c81_polar_files):
    """
    Read in the C81 polars and assigns the resulting values correctly into the BETDisk dictionary.

    Parameters
    ----------
    bet_disk: dictionary, contains required betdisk data
    c81_polar_files: list of C81 polar files

    Attributes
    ----------
    return: dictionary
    """
    if len(c81_polar_files) != len(bet_disk["sectional_radiuses"]):
        raise ValueError(
            f'Error: There is an error in the number of polar files ({len(c81_polar_files)}) vs the number of sectional Radiuses ({len(bet_disk["sectionalRadiuses"])})'
        )

    bet_disk["sectional_polars"] = []
    for sec_idx, section in enumerate(bet_disk["sectional_radiuses"]):
        polar_file = c81_polar_files[sec_idx][0]
        print(f"doing sectional_radius {section} with polar file {polar_file}")
        if not path.isfile(polar_file):
            raise ValueError(f"Error: c81 format polar file {polar_file} does not exist.")

        if "csv" in polar_file:
            alpha_list, mach_list, cl_list, cd_list = read_in_c81_polar_csv(polar_file)

        else:
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

        secpol = {}
        secpol["lift_coeffs"] = []
        secpol["drag_coeffs"] = []
        for mach in bet_disk["mach_numbers"]:
            secpol["lift_coeffs"].append([cl_list[mach]])
            secpol["drag_coeffs"].append([cd_list[mach]])
        bet_disk["sectional_polars"].append(secpol)

    return bet_disk


def generate_xfoil_bet_json(
    geometry_file_name,
    rotation_direction_rule,
    initial_blade_direction,
    blade_line_chord,
    omega,
    chord_ref,
    n_loading_nodes,
    entities,
    number_of_blades,
    angle_unit,
    length_unit,
):
    """
    Take in a geometry input file along with the remaining required information and creates a flow360 BET input dictionary.
    This geometry input file contains the list of C81 files required to get the polars along with the geometry twist and chord definition.

    Attributes
    ----------
    geometry_file_name: string, string, path to the geometry file
    bet_disk: dictionary, contains required betdisk data
    return: dictionary with BETDisk parameters
    """

    bet_disk = {}

    twist_vec, chord_vec, sectional_radiuses, xfoil_polar_file_list = parse_geometry_file(
        geometry_file_name, length_unit=length_unit, angle_unit=angle_unit
    )
    bet_disk["entities"] = entities.stored_entities
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
    )
    bet_disk["reynolds_numbers"] = generate_reynolds()
    bet_disk["alphas"] *= angle_unit
    bet_disk["sectional_radiuses"] *= length_unit
    bet_disk.pop("radius", None)

    return bet_disk


def parse_geometry_file(geometry_file_name, length_unit, angle_unit):
    """
    Read in the geometry file. This file is a csv containing the filenames of the polar definition files along with the twist and chord definitions.
    Assumes the following format:
    If it is a C81 polar format, all the mach numbers are in the same file, hence 1 file per section.
    If it is a Xfoil polar format, we need multiple file per section if we want to cover multiple machs
    number,filenameM1.csv,filenameM2.csv...
    number2,filename2M1.csv,filename2M2.csv,...
    number3,filename3M1.csv,filename3M2.csv,...
    number4,filename4M1.csv,filename4M2.csv,...
    .....
    number,number,number
    number,number,number
    number,number,number
    .....

    Parameters
    ----------
    geometry_file_name: string, path to the geometry file

    Attributes
    ----------
    return: tuple of lists
    """

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
        ):
            break
        try:
            split_line = line.split(",")
            sectional_radiuses.append(float(split_line[0]))
            polar_files.append(
                [os.path.join(geometry_file_path, file.strip()) for file in split_line[1:]]
            )
            line = fid.readline().strip("\n")
        except Exception as e:
            raise ValueError(
                f"ERROR: exception thrown when parsing line {line} from geometry file {geometry_file_name}"
            )

    while True:
        try:
            line = fid.readline().strip(
                "\n"
            )
            if not line:
                break
            radius_station.append(float(line.split(",")[0]))
            chord.append(float(line.split(",")[1]))
            twist.append(float(line.split(",")[2]))
        except:
            raise ValueError(
                f"ERROR: exception thrown when parsing line {line} from geometry file {geometry_file_name}"
            )

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


def generate_c81_bet_json(
    geometry_file_name,
    rotation_direction_rule,
    initial_blade_direction,
    blade_line_chord,
    omega,
    chord_ref,
    n_loading_nodes,
    entities,
    angle_unit,
    length_unit,
    number_of_blades,
):
    """
    Take in a geometry input file along with the remaining required information and creates a flow360 BET input dictionary.
    This geometry input file contains the list of C81 files required to get the polars along with the geometry twist and chord definition.

    Attributes
    ----------
    geometry_file_name: string, path to the geometry file
    bet_disk: dictionary, contains required BETDisk data
    return: dictionary with BETDisk parameters
    """

    twist_vec, chord_vec, sectional_radiuses, c81_polar_file_list = parse_geometry_file(
        geometry_file_name=geometry_file_name, length_unit=length_unit, angle_unit=angle_unit
    )

    bet_disk = {}
    bet_disk["entities"] = entities.stored_entities
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
    )
    bet_disk["reynolds_numbers"] = generate_reynolds()
    bet_disk["alphas"] *= angle_unit
    bet_disk["sectional_radiuses"] *= length_unit
    bet_disk.pop("radius", None)

    return bet_disk


def check_comment(comment_line, line_num, numelts):
    """
    Used when reading an XROTOR input file to make sure that what should be comments, really are.

    Attributes
    ----------
    comment_line: string
    numelts: int
    """
    if not comment_line:
        return

    if not comment_line[0] == "!" and not (len(comment_line) == numelts):
        raise ValueError(f"wrong format for line #%i: {comment_line}" % (line_num))


def check_num_values(values_list, line_num, numelts):
    """
    Used to make sure we have the expected number of inputs in a given line.

    Attributes
    ----------
    values: list
    numelts:  int
    return: None, raises an exception if the error condition is met
    """

    if not (len(values_list) == numelts):
        raise ValueError(
            f"wrong number of items for line #%i: {values_list}. We were expecting %i numbers and got %i"
            % (line_num, len(values_list), numelts)
        )


def read_dfdc_file(dfdc_file_name):
    """
    Read in the provided dfdc file.
    Does rudimentary checks to make sure the file is truly in the dfdc format.

    Attributes
    ----------
    dfdc_file_name: string
    return: dictionary

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

        dfdc_input_dict = {}
        line_num = 0
        for i in range(4):
            fid.readline()
            line_num += 1
        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 4)
        values = fid.readline().split()
        line_num += 1

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
            fid.readline()
            line_num += 1

        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 2)
        values = fid.readline().split()
        line_num += 1
        check_num_values(values, line_num, 1)
        dfdc_input_dict["nAeroSections"] = int(values[0])
        dfdc_input_dict["rRstations"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["a0deg"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["dclda"] = [0] * dfdc_input_dict[
            "nAeroSections"
        ]
        dfdc_input_dict["clmax"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["clmin"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["dcldastall"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["dclstall"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["mcrit"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["cdmin"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["clcdmin"] = [0] * dfdc_input_dict["nAeroSections"]
        dfdc_input_dict["dcddcl2"] = [0] * dfdc_input_dict["nAeroSections"]

        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 2)
        for i in range(dfdc_input_dict["nAeroSections"]):

            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 1)
            dfdc_input_dict["rRstations"][i] = float(values[0])

            comment_line = fid.readline().upper().split()
            line_num += 1
            check_comment(comment_line, line_num, 5)
            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 4)
            dfdc_input_dict["a0deg"][i] = float(values[0])
            dfdc_input_dict["dclda"][i] = float(values[1])
            dfdc_input_dict["clmax"][i] = float(values[2])
            dfdc_input_dict["clmin"][i] = float(values[3])

            comment_line = fid.readline().upper().split()
            line_num += 1
            check_comment(comment_line, line_num, 5)
            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 4)
            dfdc_input_dict["dcldastall"][i] = float(values[0])
            dfdc_input_dict["dclstall"][i] = float(values[1])
            dfdc_input_dict["mcrit"][i] = float(values[3])

            comment_line = fid.readline().upper().split()
            line_num += 1
            check_comment(comment_line, line_num, 4)
            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 3)
            dfdc_input_dict["cdmin"][i] = float(values[0])
            dfdc_input_dict["clcdmin"][i] = float(values[1])
            dfdc_input_dict["dcddcl2"][i] = float(values[2])

            for i in range(2):
                fid.readline()
                line_num += 1

        for i in range(3):
            fid.readline()
            line_num += 1

        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 3)
        values = fid.readline().split()
        line_num += 1
        check_num_values(values, line_num, 3)
        dfdc_input_dict["nBlades"] = int(values[1])
        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 2)
        values = fid.readline().split()
        line_num += 1
        check_num_values(values, line_num, 1)
        dfdc_input_dict["nGeomStations"] = int(values[0])
        dfdc_input_dict["rRGeom"] = [0] * dfdc_input_dict["nGeomStations"]
        dfdc_input_dict["cRGeom"] = [0] * dfdc_input_dict["nGeomStations"]
        dfdc_input_dict["beta0Deg"] = [0] * dfdc_input_dict["nGeomStations"]
        comment_line = fid.readline().upper().split()
        line_num += 1
        check_comment(comment_line, line_num, 4)
        for i in range(dfdc_input_dict["nGeomStations"]):
            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 3)
            dfdc_input_dict["rRGeom"][i] = float(
                values[0]
            )
            dfdc_input_dict["cRGeom"][i] = float(
                values[1]
            )
            dfdc_input_dict["beta0Deg"][i] = float(values[2])
        if dfdc_input_dict["rRGeom"][0] != 0:
            dfdc_input_dict["rRGeom"].insert(0, 0.0)
            dfdc_input_dict["cRGeom"].insert(0, 0.0)
            dfdc_input_dict["beta0Deg"].insert(0, 90.0)
            dfdc_input_dict["nGeomStations"] += 1

    dfdc_input_dict["rad"] = dfdc_input_dict["rRGeom"][
        -1
    ]
    dfdc_input_dict["omegaDim"] = dfdc_input_dict["RPM"] * pi / 30
    dfdc_input_dict["inputType"] = (
        "dfdc"
    )
    return dfdc_input_dict


def read_xrotor_file(xrotor_file_name):
    """
    Read in the provided Xrotor file.
    Does rudimentary checks to make sure the file is truly in the Xrotor format.

    Attributes
    ----------
    input: xrotor_file_name: string
    returns: dictionary

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
        line_num = 0
        top_line = fid.readline()
        line_num += 1
        if top_line.find("DFDC") == 0:
            fid.close()
            return read_dfdc_file(xrotor_file_name)

        elif top_line.find("XROTOR") == -1:
            raise ValueError("This input Xrotor file does not seem to be a valid Xrotor input file")

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
        ):
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

        fid.readline()
        line_num += 1
        fid.readline()
        line_num += 1

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

        for i in range(n_geom_stations):

            values = fid.readline().split()
            line_num += 1
            check_num_values(values, line_num, 4)
            xrotor_input_dict["rRGeom"][i] = float(values[0])
            xrotor_input_dict["cRGeom"][i] = float(values[1])
            xrotor_input_dict["beta0Deg"][i] = float(values[2])

    finally:
        fid.close()

    if xrotor_input_dict["rRGeom"][0] != 0:
        xrotor_input_dict["rRGeom"].insert(0, 0.0)
        xrotor_input_dict["cRGeom"].insert(0, 0.0)
        xrotor_input_dict["beta0Deg"].insert(0, 90.0)
        xrotor_input_dict["nGeomStations"] += 1

    xrotor_input_dict["omegaDim"] = xrotor_input_dict["vel"] / (
        xrotor_input_dict["adv"] * xrotor_input_dict["rad"]
    )
    xrotor_input_dict["RPM"] = xrotor_input_dict["omegaDim"] * 30 / pi
    xrotor_input_dict["inputType"] = (
        "xrotor"
    )
    return xrotor_input_dict


def float_range(start, stop, step=1):
    return [float(a) for a in range(start, stop, step)]


def generate_twists(xrotor_dict, mesh_unit, length_unit, angle_unit):
    """
    Transform the XROTOR format blade twists distribution into the Flow360 standard.

    Attributes
    ----------
    xrotor_dict: dictionary, contains XROTOR data
    mesh_unit: float, grid unit length with units
    return: list of dictionaries
    """

    twist_vec = []
    if xrotor_dict["inputType"] == "xrotor":
        multiplier = xrotor_dict[
            "rad"
        ]
    elif xrotor_dict["inputType"] == "dfdc":
        multiplier = 1.0

    for i in range(xrotor_dict["nGeomStations"]):
        r = xrotor_dict["rRGeom"][i] * multiplier * u.m / mesh_unit
        twist = xrotor_dict["beta0Deg"][i]
        twist_vec.append({"radius": r * length_unit, "twist": twist * angle_unit})

    return twist_vec


def generate_chords(xrotor_dict, mesh_unit, length_unit):
    """
    Transform the XROTOR format blade chords distribution into the Flow360 standard.

    Attributes
    ----------
    xrotor_dict: dictionary, contains Xrotor data
    mesh_unit: float, grid unit length with units
    return: list of dictionaries
    """

    chord_vec = []
    if xrotor_dict["inputType"] == "xrotor":
        multiplier = xrotor_dict[
            "rad"
        ]
    elif xrotor_dict["inputType"] == "dfdc":
        multiplier = 1.0
    for i in range(xrotor_dict["nGeomStations"]):
        r = xrotor_dict["rRGeom"][i] * multiplier * u.m / mesh_unit
        chord = xrotor_dict["cRGeom"][i] * multiplier * u.m / mesh_unit
        chord_vec.append({"radius": r * length_unit, "chord": chord * length_unit})

    return chord_vec


def generate_machs():
    """
    Generate 4 different tables at 4 different Mach.

    Attributes
    ----------
    return: list of floats
    """

    mach_vec = [0, sqrt(1 / 3), sqrt(2 / 3), sqrt(0.9)]
    return mach_vec


def generate_reynolds():
    """
    Assigns a constant.

    Attributes
    ----------
    return: 1.
    """
    return [1]


def generate_alphas():
    """
    Generate the list of Alphas that the BET 2d section polar is for in 1 degree steps from -180 to 180.

    Attributes
    ----------
    return: list of floats
    """

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


def find_cl_min_max_alphas(c_lift, cl_min, cl_max):
    """
    Separate the linear CL regime (i.e. from cl_min to cl_max) and extract its indices.

    Attributes
    ----------
    c_lift: list of floats
    cl_min: float
    cl_max: float
    return: tuple of ints
    """

    cl_min_idx = 0
    cl_max_idx = len(c_lift)
    for i in range(len(c_lift)):
        if c_lift[i] < cl_min:
            cl_min_idx = i
        if c_lift[i] > cl_max:
            cl_max_idx = i
            break
    return (
        cl_min_idx - 1,
        cl_max_idx + 1,
    )


def blend_func_value(blend_window, alpha, alpha_min_max, alpha_range):
    """
    Blend the flat plate CL and CD polar to the given Cl and CD polars.
    The returned blend value is 1 when we use the given CL and CD values and 0 when we use the Flat plate values.
    Within the blend_window range of alphas it returns a COS^2 based smooth blend.

    Attributes
    ----------
    blend_window: float, size of the window we want to blend from the given 2D polar
    alpha: float, current alpha in radians
    alpha_min_max: float, alpha min or alpha max for that 2D polar in radians
    alpha_range: string, used to figure out whether we are doing before CLmin or beyond CLmax
    return: float, blend value for current alpha
    """

    if "above_cl_max" in alpha_range:
        if alpha < alpha_min_max:
            return 1
        if alpha > alpha_min_max + blend_window:
            return 0
        return cos((alpha - alpha_min_max) / blend_window * pi / 2) ** 2
    if "below_cl_min" in alpha_range:
        if alpha > alpha_min_max:
            return 1
        if alpha < alpha_min_max - blend_window:
            return 0
        return cos((alpha - alpha_min_max) / blend_window * pi / 2) ** 2
    else:
        raise ValueError(
            f"alpha_range must be either above_cl_max or below_cl_min, it is: {alpha_range}"
        )


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
    return: tuple of floats, represent the blended CL and CD at current alpha
    """

    blend_window = 0.5
    alpha_min = alphas[alpha_min_idx] * pi / 180
    alpha_max = alphas[alpha_max_idx] * pi / 180

    for i in range(alpha_min_idx):
        a = alphas[i] * pi / 180

        blend_val = blend_func_value(
            blend_window, a, alpha_min, "below_cl_min"
        )
        c_lift[i] = c_lift[i] * blend_val + (1 - blend_val) * cos(a) * 2 * pi * sin(a) / sqrt(
            1 + (2 * pi * sin(a)) ** 2
        )
        c_drag[i] = (
            c_drag[i] * blend_val
            + (1 - blend_val) * sin(a) * (2 * pi * sin(a)) ** 3 / sqrt(1 + (2 * pi * sin(a)) ** 6)
            + 0.05
        )

    for j in range(alpha_max_idx, len(alphas)):
        a = alphas[j] * pi / 180
        blend_val = blend_func_value(
            blend_window, a, alpha_max, "above_cl_max"
        )
        c_lift[j] = c_lift[j] * blend_val + (1 - blend_val) * cos(a) * 2 * pi * sin(a) / sqrt(
            1 + (2 * pi * sin(a)) ** 2
        )
        c_drag[j] = (
            c_drag[j] * blend_val
            + (1 - blend_val) * sin(a) * (2 * pi * sin(a)) ** 3 / sqrt(1 + (2 * pi * sin(a)) ** 6)
            + 0.05
        )
    return c_lift, c_drag


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
    xrotor_dict: dictionary, contains XROTOR data
    alphas: list of ints, alphas we have for the polar
    mach_num: float, mach number we do this polar at
    nrR_station: int, which r/R station we have to define this polar for
    return: tuple of lists, represent the CL and CD for that polar
    """

    CDMFACTOR = 10.0
    CLMFACTOR = 0.25
    MEXP = 3.0
    CDMDD = 0.0020
    CDMSTALL = 0.1000

    MSQ = mach_num**2

    if MSQ > 1.0:
        print("CLFUNC: Local Mach^2 number limited to 0.99, was ", MSQ)
        MSQ = 0.99

    PG = 1.0 / sqrt(1.0 - MSQ)
    MACH = mach_num

    A_zero = xrotor_dict["a0deg"][nrR_station] * pi / 180
    DCLDA = xrotor_dict["dclda"][nrR_station]

    CLA = [0] * len(alphas)
    for i, a in enumerate(alphas):
        CLA[i] = DCLDA * PG * ((a * pi / 180) - A_zero)
    CLA = array(CLA)

    CLMAX = xrotor_dict["clmax"][nrR_station]
    CLMIN = xrotor_dict["clmin"][nrR_station]
    CLDMIN = xrotor_dict["clcdmin"][nrR_station]
    MCRIT = xrotor_dict["mcrit"][nrR_station]

    DMSTALL = (CDMSTALL / CDMFACTOR) ** (1.0 / MEXP)
    CLMAXM = max(0.0, (MCRIT + DMSTALL - MACH) / CLMFACTOR) + CLDMIN
    CLMAX = min(CLMAX, CLMAXM)
    CLMINM = min(0.0, -(MCRIT + DMSTALL - MACH) / CLMFACTOR) + CLDMIN
    CLMIN = max(CLMIN, CLMINM)

    DCL_STALL = xrotor_dict["dclstall"][nrR_station]
    ECMAX = expList(clip((CLA - CLMAX) / DCL_STALL, -inf, 200))
    ECMIN = expList(clip((CLA * (-1) + CLMIN) / DCL_STALL, -inf, 200))
    CLLIM = logList((ECMAX + 1.0) / (ECMIN + 1.0)) * DCL_STALL

    DCLDA_STALL = xrotor_dict["dcldastall"][nrR_station]
    FSTALL = DCLDA_STALL / DCLDA
    CLIFT = CLA - CLLIM * (1.0 - FSTALL)

    CDMIN = xrotor_dict["cdmin"][nrR_station]
    DCDCL2 = xrotor_dict["dcddcl2"][nrR_station]

    RCORR = 1
    CDRAG = (((CLIFT - CLDMIN) ** 2) * DCDCL2 + CDMIN) * RCORR

    FSTALL = DCLDA_STALL / DCLDA
    DCDX = CLLIM * (1.0 - FSTALL) / (PG * DCLDA)
    DCD = (DCDX**2) * 2.0

    DMDD = (CDMDD / CDMFACTOR) ** (1.0 / MEXP)
    CRITMACH = absList(CLIFT - CLDMIN) * CLMFACTOR * (-1) + MCRIT - DMDD
    CDC = array([0 for i in range(len(CRITMACH))])
    for crit_mach_idx in range(len(CRITMACH)):
        if MACH < CRITMACH[crit_mach_idx]:
            continue
        else:
            CDC[crit_mach_idx] = CDMFACTOR * (MACH - CRITMACH[crit_mach_idx]) ** MEXP

    FAC = 1.0

    CDRAG = CDRAG * FAC + DCD + CDC


    alpha_min_idx, alpha_max_idx = find_cl_min_max_alphas(CLIFT, CLMIN, CLMAX)

    CLIFT, CDRAG = xrotor_blend_to_flat_plate(CLIFT, CDRAG, alphas, alpha_min_idx, alpha_max_idx)

    return list(CLIFT), list(CDRAG)


def get_polar(xrotor_dict, alphas, machs, rR_station):
    """
    Return the 2D Cl and CD polar expected by the Flow360 BET model.
    b/c we have 4 Mach Values * 1 Reynolds value we need 4 different arrays per sectional polar as in:
    since the order of brackets is Mach#, Rey#, Values then we need to return:
    [[[array for MAch #1]],[[array for MAch #2]],[[array for MAch #3]],[[array for MAch #4]]]


    Attributes
    ----------
    xrotor_dict: dictionary, contains XROTOR data
    alphas: list of floats
    machs: list of floats
    rR_station: int, station index
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


def generate_xrotor_bet_json(
    xrotor_file_name,
    rotation_direction_rule,
    initial_blade_direction,
    blade_line_chord,
    omega,
    chord_ref,
    n_loading_nodes,
    entities,
    angle_unit,
    length_unit,
    mesh_unit,
):
    """
    Takes in an XROTOR or DFDC input file and translates it into a flow360 BET input dictionary.

    DFDC and XROTOR come from the same family of CFD codes. They are both written by Mark Drela over at MIT.

    Attributes
    ----------
    geometry_file_name: string, path to the XROTOR file
    bet_disk: dictionary, contains required BETDisk data
    mesh_unit: float, grid unit length with units
    return: dictionary with BETDisk parameters
    """

    xrotor_dict = read_xrotor_file(xrotor_file_name)

    bet_disk = {}

    bet_disk["entities"] = entities.stored_entities
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
    )

    return bet_disk


def generate_dfdc_bet_json(
    dfdc_file_name,
    rotation_direction_rule,
    initial_blade_direction,
    blade_line_chord,
    omega,
    chord_ref,
    n_loading_nodes,
    entities,
    angle_unit,
    length_unit,
    mesh_unit,
):
    return generate_xrotor_bet_json(
        xrotor_file_name=dfdc_file_name,
        rotation_direction_rule=rotation_direction_rule,
        initial_blade_direction=initial_blade_direction,
        blade_line_chord=blade_line_chord,
        omega=omega,
        chord_ref=chord_ref,
        n_loading_nodes=n_loading_nodes,
        entities=entities,
        angle_unit=angle_unit,
        length_unit=length_unit,
        mesh_unit=mesh_unit,
    )