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
def readInXfoilPolar(polarFile):
    """
    Parameters
    ----------
    polarFile: path to the xfoil polar file.

    Returns
    -------
    alphaList, machList, clList, cdList
    """
    clAlphas = []
    clValues = {}  # dictionary of list with the machs as keys
    cdValues = {}  # dictionary of list with the machs as keys

    xfoilFid = open(polarFile, "r")
    xfoilFid.readline()  # skip the header
    for i in range(8):  # skip the first 9 lines
        line = xfoilFid.readline()

    machNum = line.strip().split(" ")[4]
    clValues[machNum] = []
    cdValues[machNum] = []
    for i in range(4):  # skip the next 4 lines
        line = xfoilFid.readline()
    while True:
        linecontents = line.strip().split(" ")

        c = linecontents.count(
            ""
        )  # remove all instances of '' because that number varies form file to file.
        for i in range(c):
            linecontents.remove("")

        clAlphas.append(float(linecontents[0]))
        clValues[machNum].append(float(linecontents[1]))
        cdValues[machNum].append(float(linecontents[2]))
        line = xfoilFid.readline()
        if len(line) == 0:  # If we did all the alphas and we are done
            break
    # extrapolate alphas to +-180 deg and Use the flat plate Cl and CD outside of where we have values from Xfoil
    clAlphas, clMachNums, clValues, cdValues = blendPolarstoFlatplate(
        clAlphas, [machNum], clValues, cdValues
    )

    # Now we interpolate the polar data to a constant set of alphas to make sure we have all the smae alphas across all mach and section
    # 10 deg steps from -180 ->-30 and from 30 to 180. 1 deg steps from -29 to 29
    degIncrementAng = list(np.arange(-30, 30, 1).astype(float))

    alphas = (
        list(np.arange(-180, -30, 10).astype(float))
        + degIncrementAng
        + list(np.arange(30, 190, 10).astype(float))
    )  # json doesn't like the numpy default int64 type so I make it a float

    clInterp = interp1d(
        clAlphas, clValues[clMachNums[0]], kind="linear"
    )  # method should be linear to make sure we still have 0 at the +- 180 values
    cdInterp = interp1d(
        clAlphas, cdValues[clMachNums[0]], kind="linear"
    )  # method should be linear to make sure we still have 0 at the +- 180 values
    cls = [0 for i in range(len(alphas))]
    cds = [0 for i in range(len(alphas))]
    for i, alpha in enumerate(alphas):  # interpolate the cl and cd over the new set of alphas
        cls[i] = float(clInterp(alpha))
        cds[i] = float(cdInterp(alpha))

    xfoilFid.close()

    return alphas, clMachNums[0], cls, cds


########################################################################################################################
def blendPolarstoFlatplate(clAlphas, clMachNums, clValues, cdValues):
    """
    This function blends a given arbitrary set of CL and CD polars that are missing values to cover the whole -180 to 180
    range of angles. The resulting polars will have the missing values be replaced by the flat plate CL and CD.
    Parameters
    ----------
    clAlphas: list of alpha angles
    clMachNums: list of mach numbers
    clValues: dict with dimensions nMach*nAlphas
    cdValues: dict with dimensions nMach*nAlphas

    Returns
    -------
    clAlphas, clMachNums, clValues, cdValues with polars completed to +- 180
    """

    polarAlphaStepBlend = 10  # add a polar point every N alpha

    alphaMin = clAlphas[0]
    alphaMax = clAlphas[-1]
    if alphaMin < -180:
        raise ValueError(f"ERROR: alphaMin is smaller then -180: {alphaMin}")
    if alphaMax > 180:
        raise ValueError(f"ERROR: alphaMax is greater then 180: {alphaMin}")

    blendWindow = 0.5  # 0.5 radians

    # create a point every 10 deg, how many points do we need.
    numMissingAlphasMin = round((alphaMin + 180) / polarAlphaStepBlend)
    numMissingAlphasMax = round((180 - alphaMax) / polarAlphaStepBlend)

    for i in range(numMissingAlphasMin - 1):  # add alphas at beginning of clAlphas list
        clAlphas.insert(0, clAlphas[0] - polarAlphaStepBlend)
        a = clAlphas[0] * pi / 180  # smallest alpha in radians
        for i, mach in enumerate(clMachNums):
            blendVal = blendFuncValue(
                blendWindow, a, alphaMin * pi / 180, "belowCLmin"
            )  # we are on the alphaCLmin side going up in CL
            # this follows the flat plate lift and drag equations times the blend val coefficient

            cLift = clValues[mach][0] * blendVal + (1 - blendVal) * cos(a) * 2 * pi * sin(a) / sqrt(
                1 + (2 * pi * sin(a)) ** 2
            )
            cd = (
                cdValues[mach][0] * blendVal
                + (1 - blendVal)
                * sin(a)
                * (2 * pi * sin(a)) ** 3
                / sqrt(1 + (2 * pi * sin(a)) ** 6)
                + 0.05
            )
            mach = str(mach)
            clValues[mach].insert(0, cLift)  # add a new cl value at the beginning
            cdValues[mach].insert(0, cd)  # add a new cl value at the beginning

    for i in range(numMissingAlphasMax - 1):  # add alphas at end of clAlphas list
        clAlphas.append(clAlphas[-1] + polarAlphaStepBlend)
        a = clAlphas[-1] * pi / 180  # smallest alpha in radians
        for i, mach in enumerate(clMachNums):
            blendVal = blendFuncValue(
                blendWindow, a, alphaMax * pi / 180, "aboveCLmax"
            )  # we are on the alphaCLmin side going up in CL
            # this follows the flat plate lift and drag equations times the blend val coefficient

            cLift = clValues[mach][-1] * blendVal + (1 - blendVal) * cos(a) * 2 * pi * sin(
                a
            ) / sqrt(1 + (2 * pi * sin(a)) ** 2)
            cd = (
                cdValues[mach][-1] * blendVal
                + (1 - blendVal)
                * sin(a)
                * (2 * pi * sin(a)) ** 3
                / sqrt(1 + (2 * pi * sin(a)) ** 6)
                + 0.05
            )
            mach = str(mach)
            clValues[mach].append(cLift)  # add a new cl value at the beginning
            cdValues[mach].append(cd)  # add a new cl value at the beginning

    clAlphas.insert(0, -180)  # make sure that the last value in the list is 180
    clAlphas.append(180)  # make sure that the last value in the list is 180
    for i, mach in enumerate(clMachNums):
        clValues[mach].insert(0, 0)  # make sure Cl=0 at alpha -180
        cdValues[mach].insert(0, 0.05)  # Cd=0.05 is flat plate Cd at 180
        clValues[mach].append(0)  # make sure Cl=0 at alpha -180
        cdValues[mach].append(0.05)  # Cd=0.05 is flat plate Cd at 180

    return clAlphas, clMachNums, clValues, cdValues


###############################################################################################################
def readInC81Polarc81Format(polarFile):
    """
    Read in the c81 format polar file
    This function checks that the list of Alphas is consistent across CL and CD and that the number of Machs is also consistent across Cl and CD.
    Parameters
    ----------
    polarFile

    Returns
    -------
    4 lists of floats: clAlphas, clMachNums, clValues, cdValues
    """
    clAlphas = []
    cdAlphas = []
    clValues = {}  # dictionary of list with the machs as keys
    cdValues = {}  # dictionary of list with the machs as keys

    c81fid = open(polarFile, "r")
    c81fid.readline()  # skip the header
    line = c81fid.readline()
    clMachNums = line.strip().split(" ")
    clMachNums = [
        float(i) for i in clMachNums if i
    ]  # remove empty items and trailing \n in clMachNums
    for mach in clMachNums:
        clValues[mach] = []
    line = c81fid.readline()
    while True:
        # c81 format is as per this document https://cibinjoseph.github.io/C81-Interface/page/index.html
        # first 7 chars in string is A0A then 7 chars per cl value
        if (
            line[:7] == "       "
        ):  # If we did all the alphas and now that line starts with a bunch of spaces.
            break
        clAlphas.append(float(line[:7]))

        for i, mach in enumerate(clMachNums):
            indexBeg = i * 7 + 7
            indexEnd = (i + 1) * 7 + 7
            clValues[mach].append(float(line[indexBeg:indexEnd]))
        line = c81fid.readline()

    # Now do the CDs
    cdMachNums = line.strip().split(" ")
    # we already read the mach numbers line in the while loop above. so just split it.
    cdMachNums = [
        float(i) for i in cdMachNums if i
    ]  # remove empty items and trailing \n in clMachNums
    if clMachNums != cdMachNums:  # if we have different lists of  machs
        raise ValueError(
            f"ERROR: in file {polarFile}, The machs in the Cl polar do not match the machs in the CD polar, we have {clMachNums} Cl mach values and {cdMachNums} CD mach values:"
        )

    for mach in cdMachNums:
        cdValues[mach] = []
    line = c81fid.readline()
    while True:
        # c81 format is as per this document https://cibinjoseph.github.io/C81-Interface/page/index.html
        # first 7 chars in string is A0A then 7 chars per cl value
        if (
            line[:7] == "       "
        ):  # If we did all the alphas and now that line starts with a bunch of spaces.
            break
        cdAlphas.append(float(line[:7]))

        for i, mach in enumerate(cdMachNums):
            indexBeg = i * 7 + 7
            indexEnd = (i + 1) * 7 + 7
            cdValues[mach].append(float(line[indexBeg:indexEnd]))
        line = c81fid.readline()

    if clAlphas != cdAlphas:  # if we have different  lists of alphas
        raise ValueError(
            f"ERROR: in file {polarFile}, The alphas in the Cl polar do not match the alphas in the CD polar. We have {clAlphas} Cls and {cdAlphas} Cds"
        )

    # We also have the moment informatiomn in a c81 file but we ignore that for our purposes.

    return clAlphas, clMachNums, clValues, cdValues


###############################################################################################################
def readInC81Polarcsv(polarFile):
    """
    # read in the c81 format polar file as a csv file
    # the script checks that the list of Alphas is consistent across CL and CD and that the number of Machs is also consistent across Cl and CD.
    Parameters
    ----------
    polarFile

    Returns
    -------
     4 lists of floats: clAlphas, clMachNums, clValues, cdValues
    """

    clAlphas = []
    cdAlphas = []
    clValues = {}  # dictionary of list with the machs as keys
    cdValues = {}  # dictionary of list with the machs as keys

    c81fid = open(polarFile, "r")
    c81fid.readline()  # skip the header
    line = c81fid.readline()
    clMachNums = line.split(",")
    clMachNums = [
        float(i.strip()) for i in clMachNums if i
    ]  # remove empty items and trailing \n in clMachNums
    # numClMachs=len(clMachNums) #number of machs we have
    for mach in clMachNums:
        clValues[mach] = []
    line = c81fid.readline()
    while True:
        values = line.split(",")
        if values[0] == "":  # If we did all the alphas
            break
        clAlphas.append(float(values[0]))
        for i, mach in enumerate(clMachNums):
            clValues[mach].append(float(values[i + 1].strip()))
        line = c81fid.readline()

    # Now do the CDs
    cdMachNums = line.split(
        ","
    )  # we already read the mach numbers line in the while loop above. so just split it.
    cdMachNums = [
        float(i.strip()) for i in cdMachNums if i
    ]  # remove empty items and trailing \n in clMachNums
    if clMachNums != cdMachNums:  # if we have different lists of  machs
        raise ValueError(
            f"ERROR: in file {polarFile}, The machs in the Cl polar do not match the machs in the CD polar, we have {clMachNums} Cl mach values and {cdMachNums} CD mach values:"
        )

    for mach in cdMachNums:
        cdValues[mach] = []
    line = c81fid.readline()
    while True:
        values = line.split(",")
        if values[0] == "":  # If we did all the alphas
            break
        cdAlphas.append(float(values[0]))
        for i, mach in enumerate(cdMachNums):
            cdValues[mach].append(float(values[i + 1].strip()))
        line = c81fid.readline()

    if clAlphas != cdAlphas:  # if we have different  lists of alphas
        raise ValueError(
            f"ERROR: in file {polarFile}, The alphas in the Cl polar do not match the alphas in the CD polar. We have {len(clAlphas)} Cls and {len(cdAlphas)} Cds"
        )

    # We also have the moment information in a c81 file but we ignore that for our purposes.
    if (
        clAlphas[0] != -180 and clAlphas[-1] != 180
    ):  # if we don't have polars for the full circle of alpha angles.
        blendPolarstoFlatplate(clAlphas, clMachNums, clValues, cdValues)
    c81fid.close()
    return clAlphas, clMachNums, clValues, cdValues


###############################################################################################################
def readInXfoilData(betDisk, xfoilPolarfiles):
    """
    This function reads in the Xfoil polars and assigns the resulting values correctly into the BET disk dictionary
    Parameters
    ----------
    betDisk - Dictionary of values needed for the BET disk implementation
    xfoilPolarfiles - list of xfoil polar files

    Returns
    -------
    betDisk - same dictionary as was passed to function but with all the polar information added.
    """
    if len(xfoilPolarfiles) != len(betDisk["sectionalRadiuses"]):
        raise ValueError(
            f'Error: There is an error in the number of polar files ({len(xfoilPolarfiles)}) vs the number of sectional Radiuses ({len(betDisk["sectionalRadiuses"])})'
        )

    betDisk["sectionalPolars"] = []
    betDisk["MachNumbers"] = []

    machNumbers = []

    for secIdx, section in enumerate(betDisk["sectionalRadiuses"]):
        secpol = (
            {}
        )  # temporary dict to store all the section polars before assigning it to the right location.
        secpol["liftCoeffs"] = []
        secpol["dragCoeffs"] = []

        polarFiles = xfoilPolarfiles[secIdx]
        machNumbersforsection = []
        for polarFile in polarFiles:
            print(f"doing sectionalRadius {section} with polar file {polarFile}")
            if not path.isfile(polarFile):
                raise ValueError(f"Error: xfoil format polar file {polarFile} does not exist.")
            alphaList, machNum, clValues, cdValues = readInXfoilPolar(
                polarFile
            )  # read in xfoil data and use flat plate values outside of given polar range
            machNumbersforsection.append(float(machNum))
            secpol["liftCoeffs"].append([clValues])
            secpol["dragCoeffs"].append([cdValues])
        machNumbers.append(machNumbersforsection)
        betDisk["sectionalPolars"].append(secpol)
    for i in range(
        len(machNumbers) - 1
    ):  # check to make sure all N cross sections have the same list of mach numbers
        if machNumbers[i] != machNumbers[i + 1]:
            raise ValueError(
                f'ERROR: the mach numbers from the Xfoil polars need to be the same set for each cross section. Here sections {i} \
                    and {i+1} have the following sets of mach numbers:{secpol["machNumbers"][i]} and {secpol["machNumbers"][i+1]}'
            )
    betDisk["alphas"] = alphaList
    betDisk["MachNumbers"] = machNumbers[
        0
    ]  # they should all be the same set so just pick the first one.
    #    betDisk['sectionalPolars'].append(secpol)

    return betDisk


###############################################################################################################
def readInC81Polars(betDisk, c81Polarfiles):
    """
    This function reads in the C81 polars and assigns the resulting values correctly into the BET disk dictionary
    Parameters
    ----------
    betDisk - Dictionary of values needed for the BET disk implementation
    c81Polarfiles - list of C81 polar files

    Returns
    -------
    betDisk - same dictionary as was passed to function but with all the polar information added.
    """
    if len(c81Polarfiles) != len(betDisk["sectional_radiuses"]):
        raise ValueError(
            f'Error: There is an error in the number of polar files ({len(c81Polarfiles)}) vs the number of sectional Radiuses ({len(betDisk["sectionalRadiuses"])})'
        )

    betDisk["sectional_polars"] = []
    for secIdx, section in enumerate(betDisk["sectional_radiuses"]):
        polarFile = c81Polarfiles[secIdx][0]  # Take the first element of that list.
        print(f"doing sectional_radius {section} with polar file {polarFile}")
        if not path.isfile(polarFile):
            raise ValueError(f"Error: c81 format polar file {polarFile} does not exist.")

        if "csv" in polarFile:  # if we are dealing with a csv file
            alphaList, machList, clList, cdList = readInC81Polarcsv(polarFile)

        else:
            # we are dealing with a genuine c81 file, then I need to handle it by splitting the list into certain sizes
            alphaList, machList, clList, cdList = readInC81Polarc81Format(polarFile)
        if "mach_numbers" in betDisk.keys() and betDisk["mach_numbers"] != machList:
            raise ValueError(
                "ERROR: The mach Numbers do not match across the various sectional radi polar c81 files. All the sectional radi need to have the same mach Numbers across all c81 polar files"
            )
        if "alphas" in betDisk.keys() and betDisk["alphas"] != alphaList:
            raise ValueError(
                "ERROR: The alphas do not match across the various sectional radi polar c81 files. All the sectional radi need to have the same alphas across all c81 polar files"
            )

        betDisk["mach_numbers"] = machList
        betDisk["alphas"] = alphaList

        # since the order of brackets is Mach#, Rey#, Values then we need to return:
        # [[[array for MAch #1]],[[array for MAch #2]],[[array for MAch #3]],[[array for MAch #4]],......]

        secpol = {}
        secpol["lift_coeffs"] = []
        secpol["drag_coeffs"] = []
        for mach in betDisk["mach_numbers"]:
            secpol["lift_coeffs"].append([clList[mach]])
            secpol["drag_coeffs"].append([cdList[mach]])
        betDisk["sectional_polars"].append(secpol)

    return betDisk


########################################################################################################################
def generateXfoilBETJSON(geometryFileName, betDisk):
    """
    This function takes in a geometry input files along with the remaining required information and creates a flow360 BET input dictionary
    This geometry input file contains the list of C81 files required to get the polars along with the geometry twist and chord definition
    Attributes
    ----------
    geometryFileName: string, filepath to the geometry files we want to translate into a BET disk
    betDisk: dictionary of the required betdisk data that we can't get form the geometry file.
    return: dictionary that we should append to the Flow360.json file we want to run with.
    """

    if betDisk["rotationDirectionRule"] not in ["rightHand", "leftHand"]:
        raise ValueError(
            f'Exiting. Invalid rotationDirectionRule: {betDisk["rotationDirectionRule"]}'
        )
    if len(betDisk["axisOfRotation"]) != 3:
        raise ValueError(f"axisOfRotation must be a list of size 3. Exiting.")
    if len(betDisk["centerOfRotation"]) != 3:
        raise ValueError("centerOfRotation must be a list of size 3. Exiting")

    twistVec, chordVec, sectionalRadiuses, xfoilPolarfileList = parseGeometryfile(geometryFileName)
    betDisk["radius"] = sectionalRadiuses[-1]
    betDisk["sectionalRadiuses"] = sectionalRadiuses
    betDisk["twists"] = twistVec
    betDisk["chords"] = chordVec
    betDisk = readInXfoilData(
        betDisk, xfoilPolarfileList
    )  # add the mach values along with the polars from the xfoil files.
    betDisk["ReynoldsNumbers"] = generateReys()

    return betDisk


########################################################################################################################
def parseGeometryfile(geometryFileName, length_unit, angle_unit):
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
    geometryFileName - path to the geometryFile. This file is a csv containing the filenames of the polar definition files along with the twist and chord definitions.

    Returns
    -------
    4 lists: twistVec, chordVec, sectionalRadiuses, c81Polarfiles
    """
    #    read in the geometry file name and return its values
    fid = open(geometryFileName)
    line = fid.readline()
    if "#" not in line:
        raise ValueError(
            f"ERROR: first character of first line of geometry file {geometryFileName} should be the # character to denote a header line"
        )

    geometryFilePath = os.path.dirname(os.path.realpath(geometryFileName))
    sectionalRadiuses = []
    polarFiles = []
    radiusStation = []
    chord = []
    twist = []
    line = fid.readline().strip("\n")
    while True:
        if (
            "#" in line
        ):  # If we have reached the end of sectional radiuses then move on to the twists and chords
            break
        try:
            splitLine = line.split(",")
            sectionalRadiuses.append(float(splitLine[0]))
            polarFiles.append(
                [os.path.join(geometryFilePath, file.strip()) for file in splitLine[1:]]
            )
            # polarFiles = [x.strip(' ') for x in polarFiles] # remove spaces in file names
            line = fid.readline().strip("\n")  # read next line.
        except Exception as e:
            raise ValueError(
                f"ERROR: exception thrown when parsing line {line} from geometry file {geometryFileName}"
            )

    while True:
        try:
            line = fid.readline().strip(
                "\n"
            )  # read in the first line of twist and chord definition
            if not line:  # if we have reached the end of the file.
                break
            radiusStation.append(float(line.split(",")[0]))
            chord.append(float(line.split(",")[1]))
            twist.append(float(line.split(",")[2]))
        except:
            raise ValueError(
                f"ERROR: exception thrown when parsing line {line} from geometry file {geometryFileName}"
            )

    # intialize chord and twist with 0,0 value at centerline
    chordVec = [{"radius": 0.0 * length_unit, "chord": 0.0 * length_unit}]
    twistVec = [{"radius": 0.0 * length_unit, "twist": 0.0 * angle_unit}]
    for i in range(len(radiusStation)):
        twistVec.append({"radius": radiusStation[i] * length_unit, "twist": twist[i] * angle_unit})
        chordVec.append({"radius": radiusStation[i] * length_unit, "chord": chord[i] * length_unit})

    fid.close()

    return twistVec, chordVec, sectionalRadiuses, polarFiles


################################################################################################################
def generateC81BETJSON(
    geometryFileName,
    rotation_direction_rule,
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
    geometryFileName: string, filepath to the geometry files we want to translate into a BET disk
    betDisk: dictionary of the required betdisk data that we can't get form the geometry file.
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

    twistVec, chordVec, sectionalRadiuses, c81PolarfileList = parseGeometryfile(
        geometryFileName=geometryFileName, length_unit=length_unit, angle_unit=angle_unit
    )

    betDisk = {}
    betDisk["entities"] = cylinder
    betDisk["omega"] = omega
    betDisk["chord_ref"] = chord_ref
    betDisk["n_loading_nodes"] = n_loading_nodes
    betDisk["rotation_direction_rule"] = rotation_direction_rule
    betDisk["number_of_blades"] = number_of_blades
    betDisk["radius"] = sectionalRadiuses[-1]
    betDisk["sectional_radiuses"] = sectionalRadiuses
    betDisk["twists"] = twistVec
    betDisk["chords"] = chordVec
    betDisk = readInC81Polars(
        betDisk, c81PolarfileList
    )  # add the mach values along with the polars from the c81 files.
    betDisk["reynolds_numbers"] = generateReys()
    betDisk["alphas"] *= angle_unit
    betDisk["sectional_radiuses"] *= length_unit
    betDisk.pop("radius", None)

    return betDisk


########################################################################################################################
def check_comment(comment_line, linenum, numelts):
    """
    This function is used when reading an XROTOR input file to make sure that what should be comments really are

    Attributes
    ----------
    comment_line: string
    numelts: int
    """
    if not comment_line:  # if the comment_line is empty.
        return

    # otherwise make sure that we are on a comment line
    if not comment_line[0] == "!" and not (len(comment_line) == numelts):
        raise ValueError(f"wrong format for line #%i: {comment_line}" % (linenum))


########################################################################################################################
def check_num_values(values_list, linenum, numelts):
    """
    This function is used to make sure we have the expected number of inputs in a given line

    Attributes
    ----------
    values: list
    numelts:  int
    return: None, it raises an exception if the error condition is met.
    """
    # make sure that we have the expected number of values.
    if not (len(values_list) == numelts):
        raise ValueError(
            f"wrong number of items for line #%i: {values_list}. We were expecting %i numbers and got %i"
            % (linenum, len(values_list), numelts)
        )


########################################################################################################################
def readDFDCFile(dfdcFileName):
    """
    This functions read in the dfdc filename provided.
    it does rudimentary checks to make sure the file is truly in the dfdc format.


    Attributes
    ----------
    dfdcFileName: string
    return: a dictionary with all the required values. That dictionary will be used to create BETdisks section of the
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

    nGeomStations: number of geometric stations where the blade geometry is defined at
    nBlades: number of blades on the propeller
    Each geometry station will have the following parameters:
      r: station r in meters
      c: local chord in meters
      beta0deg: Twist relative to disk plane. ie symmetric 2D section at beta0Deg would create 0 thrust, more beta0deg means more local angle of attack for the blade
      Ubody: (unused) Nacelle perturbation axial  velocity


    """
    with open(dfdcFileName, "r") as fid:

        # read in lines 5->8 which contains the run case information
        dfdcInputDict = {}
        linenum = 0  # counter needed to report which line the error is on.
        for i in range(4):
            fid.readline()  # we have 4 blank lines
            linenum += 1
        comment_line = fid.readline().upper().split()
        linenum += 1
        check_comment(comment_line, linenum, 4)
        values = fid.readline().split()
        linenum += 1
        # we don't want ot apply the check to this line b/c we could have a varying number of RPMs

        dfdcInputDict["vel"] = float(values[1])
        dfdcInputDict["RPM"] = float(values[2])

        comment_line = fid.readline().upper().split()
        linenum += 1
        check_comment(comment_line, linenum, 5)
        values = fid.readline().split()
        linenum += 1
        check_num_values(values, linenum, 4)
        dfdcInputDict["rho"] = float(values[0])

        for i in range(7):
            fid.readline()  # skip next 8 lines.
            linenum += 1

        comment_line = fid.readline().upper().split()  # convert all to upper case
        linenum += 1
        check_comment(comment_line, linenum, 2)  # 2 because line should have 2 components
        values = fid.readline().split()
        linenum += 1
        check_num_values(values, linenum, 1)  # we should have 1 value.
        dfdcInputDict["nAeroSections"] = int(values[0])
        # define the lists with the right number of elements
        dfdcInputDict["rRstations"] = [0] * dfdcInputDict["nAeroSections"]
        dfdcInputDict["a0deg"] = [0] * dfdcInputDict["nAeroSections"]  # WARNING, ao is in deg
        dfdcInputDict["dclda"] = [0] * dfdcInputDict[
            "nAeroSections"
        ]  # but dclda is in cl per radians
        dfdcInputDict["clmax"] = [0] * dfdcInputDict["nAeroSections"]
        dfdcInputDict["clmin"] = [0] * dfdcInputDict["nAeroSections"]
        dfdcInputDict["dcldastall"] = [0] * dfdcInputDict["nAeroSections"]
        dfdcInputDict["dclstall"] = [0] * dfdcInputDict["nAeroSections"]
        dfdcInputDict["mcrit"] = [0] * dfdcInputDict["nAeroSections"]
        dfdcInputDict["cdmin"] = [0] * dfdcInputDict["nAeroSections"]
        dfdcInputDict["clcdmin"] = [0] * dfdcInputDict["nAeroSections"]
        dfdcInputDict["dcddcl2"] = [0] * dfdcInputDict["nAeroSections"]

        comment_line = fid.readline().upper().split()  # convert all to upper case
        linenum += 1
        check_comment(comment_line, linenum, 2)  # 2 because line should have 2 components
        for i in range(dfdcInputDict["nAeroSections"]):  # iterate over all the sections

            values = fid.readline().split()
            linenum += 1
            check_num_values(values, linenum, 1)  # we should have 1 value.
            dfdcInputDict["rRstations"][i] = float(values[0])  # aka xisection

            comment_line = fid.readline().upper().split()  # convert all to upper case
            linenum += 1
            check_comment(comment_line, linenum, 5)  # 5 because line should have 5 components
            values = fid.readline().split()
            linenum += 1
            check_num_values(values, linenum, 4)  # we should have 4 value.
            dfdcInputDict["a0deg"][i] = float(values[0])  # WARNING, ao is in deg
            dfdcInputDict["dclda"][i] = float(values[1])  # but dclda is in cl per radians
            dfdcInputDict["clmax"][i] = float(values[2])
            dfdcInputDict["clmin"][i] = float(values[3])

            comment_line = fid.readline().upper().split()  # convert all to upper case
            linenum += 1
            check_comment(comment_line, linenum, 5)  # 5 because line should have 5 components
            values = fid.readline().split()
            linenum += 1
            check_num_values(values, linenum, 4)  # we should have 4 value.
            dfdcInputDict["dcldastall"][i] = float(values[0])
            dfdcInputDict["dclstall"][i] = float(values[1])
            dfdcInputDict["mcrit"][i] = float(values[3])

            comment_line = fid.readline().upper().split()  # convert all to upper case
            linenum += 1
            check_comment(comment_line, linenum, 4)  # 4 because line should have 4 components
            values = fid.readline().split()
            linenum += 1
            check_num_values(values, linenum, 3)  # we should have 3 value.
            dfdcInputDict["cdmin"][i] = float(values[0])
            dfdcInputDict["clcdmin"][i] = float(values[1])
            dfdcInputDict["dcddcl2"][i] = float(values[2])

            for i in range(2):
                fid.readline()  # skip next 3 lines.
                linenum += 1

        for i in range(3):
            fid.readline()  # skip next 3 lines.
            linenum += 1
        # Now we are done with the various aero sections and we start looking at blade geometry definitions
        comment_line = fid.readline().upper().split()  # convert all to upper case
        linenum += 1
        check_comment(comment_line, linenum, 3)  # 3 because line should have 3 components
        values = fid.readline().split()
        linenum += 1
        check_num_values(values, linenum, 3)  # we should have 3 values.
        dfdcInputDict["nBlades"] = int(values[1])
        comment_line = fid.readline().upper().split()  # convert all to upper case
        linenum += 1
        check_comment(comment_line, linenum, 2)
        values = fid.readline().split()
        linenum += 1
        check_num_values(values, linenum, 1)
        dfdcInputDict["nGeomStations"] = int(values[0])
        # 2nd value on that  line is the number of blades
        dfdcInputDict["rRGeom"] = [0] * dfdcInputDict["nGeomStations"]
        dfdcInputDict["cRGeom"] = [0] * dfdcInputDict["nGeomStations"]
        dfdcInputDict["beta0Deg"] = [0] * dfdcInputDict["nGeomStations"]
        comment_line = fid.readline().upper().split()  # convert all to upper case
        linenum += 1
        check_comment(comment_line, linenum, 4)  # 4 because line should have 4 components
        for i in range(dfdcInputDict["nGeomStations"]):  # iterate over all the geometry stations
            values = fid.readline().split()
            linenum += 1
            check_num_values(values, linenum, 3)  # we should have 3 values.
            dfdcInputDict["rRGeom"][i] = float(
                values[0]
            )  # key string is not quite true b/c it is the dimensional radius  but I need a place to store the r locations that matches the Xrotor format
            dfdcInputDict["cRGeom"][i] = float(
                values[1]
            )  # key string is not quite true b/c it is the dimensional chord but I need a place to store the chord dimensions that matches the Xrotor format
            dfdcInputDict["beta0Deg"][i] = float(values[2])  # twist values
        if dfdcInputDict["rRGeom"][0] != 0:  # As per discussion in
            # https://enreal.slack.com/archives/C01PFAJ76FL/p1643652853237749?thread_ts=1643413462.002919&cid=C01PFAJ76FL
            # i need to ensure that the blade coordinates go all the way to r/R=0 and have a 0 chord  90deg twist at r/R=0
            dfdcInputDict["rRGeom"].insert(0, 0.0)
            dfdcInputDict["cRGeom"].insert(0, 0.0)
            dfdcInputDict["beta0Deg"].insert(0, 90.0)
            dfdcInputDict["nGeomStations"] += 1  # we have added one station.

    # for i in range(dfdcInputDict['nGeomStations']):  # iterate over all the geometry stations to nondimensionalize by radius.
    #     dfdcInputDict['rRGeom'][i] = dfdcInputDict['rRGeom'][i] * dfdcInputDict['rad']  # aka r/R location
    #     dfdcInputDict['cRGeom'][i] = dfdcInputDict['cRGeom'][i] * dfdcInputDict['rad']  # aka r/R location

    dfdcInputDict["rad"] = dfdcInputDict["rRGeom"][
        -1
    ]  # radius in m is the last value in the r list
    # calculate Extra values and add them  to the dict
    dfdcInputDict["omegaDim"] = dfdcInputDict["RPM"] * pi / 30
    dfdcInputDict["inputType"] = (
        "dfdc"  # we need to store which file format we are using to handle the r vs r/R situation correctly.
    )
    # Now we are done, we have all the data we need.
    return dfdcInputDict


########################################################################################################################


########################################################################################################################
def readXROTORFile(xrotorFileName):
    """
    This functions read in the Xrotor filename provided.
    it does rudimentary checks to make sure the file is truly in the Xrotor format.


    Attributes
    ----------
    input: xrotorFileName: string
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
        fid = open(xrotorFileName, "r")
        linenum = 0  # counter needed to know which line we are on for error reporting
        # Top line in the file should start with the XROTOR keywords.
        topLine = fid.readline()
        linenum += 1
        if topLine.find("DFDC") == 0:  # If we are actually doing a DFDC file instead of Xrotor
            fid.close()  # close the file b/c we will reopen it in readDFDCFile
            return readDFDCFile(xrotorFileName)

        elif topLine.find("XROTOR") == -1:
            raise ValueError("This input Xrotor file does not seem to be a valid Xrotor input file")

        # read in lines 2->8 which contains the run case information
        xrotorInputDict = {}

        fid.readline()
        linenum += 1
        comment_line = fid.readline().upper().split()
        linenum += 1
        check_comment(comment_line, linenum, 5)

        values = fid.readline().split()
        linenum += 1
        check_num_values(values, linenum, 4)

        comment_line = fid.readline().upper().split()
        linenum += 1
        check_comment(comment_line, linenum, 5)
        values = fid.readline().split()
        linenum += 1
        check_num_values(values, linenum, 4)
        xrotorInputDict["rad"] = float(values[0])
        xrotorInputDict["vel"] = float(values[1])
        xrotorInputDict["adv"] = float(values[2])

        fid.readline()
        linenum += 1
        fid.readline()
        linenum += 1
        comment_line = fid.readline().upper().split()
        linenum += 1
        check_comment(comment_line, linenum, 2)
        values = fid.readline().split()
        linenum += 1
        check_num_values(values, linenum, 1)

        nAeroSections = int(values[0])
        # Initialize the dictionary with all the information to re-create the polars at each defining aero section.
        xrotorInputDict["nAeroSections"] = nAeroSections
        xrotorInputDict["rRstations"] = [0] * nAeroSections
        xrotorInputDict["a0deg"] = [0] * nAeroSections
        xrotorInputDict["dclda"] = [0] * nAeroSections
        xrotorInputDict["clmax"] = [0] * nAeroSections
        xrotorInputDict["clmin"] = [0] * nAeroSections
        xrotorInputDict["dcldastall"] = [0] * nAeroSections
        xrotorInputDict["dclstall"] = [0] * nAeroSections
        xrotorInputDict["mcrit"] = [0] * nAeroSections
        xrotorInputDict["cdmin"] = [0] * nAeroSections
        xrotorInputDict["clcdmin"] = [0] * nAeroSections
        xrotorInputDict["dcddcl2"] = [0] * nAeroSections

        for i in range(
            nAeroSections
        ):  # loop ever each aero section and populate the required variables.
            comment_line = fid.readline().upper().split()
            linenum += 1
            check_comment(comment_line, linenum, 2)
            values = fid.readline().split()
            linenum += 1
            check_num_values(values, linenum, 1)
            xrotorInputDict["rRstations"][i] = float(values[0])

            comment_line = fid.readline().upper().split()
            linenum += 1
            check_comment(comment_line, linenum, 5)
            values = fid.readline().split()
            linenum += 1
            check_num_values(values, linenum, 4)
            xrotorInputDict["a0deg"][i] = float(values[0])
            xrotorInputDict["dclda"][i] = float(values[1])
            xrotorInputDict["clmax"][i] = float(values[2])
            xrotorInputDict["clmin"][i] = float(values[3])

            comment_line = fid.readline().upper().split()
            linenum += 1
            check_comment(comment_line, linenum, 5)
            values = fid.readline().split()
            linenum += 1
            check_num_values(values, linenum, 4)
            xrotorInputDict["dcldastall"][i] = float(values[0])
            xrotorInputDict["dclstall"][i] = float(values[1])
            xrotorInputDict["mcrit"][i] = float(values[3])

            comment_line = fid.readline().upper().split()
            linenum += 1
            check_comment(comment_line, linenum, 4)
            values = fid.readline().split()
            linenum += 1
            check_num_values(values, linenum, 3)
            xrotorInputDict["cdmin"][i] = float(values[0])
            xrotorInputDict["clcdmin"][i] = float(values[1])
            xrotorInputDict["dcddcl2"][i] = float(values[2])

            comment_line = fid.readline().upper().split()
            linenum += 1
            check_comment(comment_line, linenum, 3)
            values = fid.readline().split()
            linenum += 1
        # skip the duct information
        fid.readline()
        linenum += 1
        fid.readline()
        linenum += 1

        # Now we are done with the various aero sections and we start
        # looking at blade geometry definitions
        comment_line = fid.readline().upper().split()
        linenum += 1
        check_comment(comment_line, linenum, 3)
        values = fid.readline().split()
        linenum += 1
        check_num_values(values, linenum, 2)

        nGeomStations = int(values[0])
        xrotorInputDict["nGeomStations"] = nGeomStations
        xrotorInputDict["nBlades"] = int(values[1])
        xrotorInputDict["rRGeom"] = [0] * nGeomStations
        xrotorInputDict["cRGeom"] = [0] * nGeomStations
        xrotorInputDict["beta0Deg"] = [0] * nGeomStations

        comment_line = fid.readline().upper().split()
        linenum += 1
        check_comment(comment_line, linenum, 5)

        # iterate over all the geometry stations
        for i in range(nGeomStations):

            values = fid.readline().split()
            linenum += 1
            check_num_values(values, linenum, 4)
            xrotorInputDict["rRGeom"][i] = float(values[0])
            xrotorInputDict["cRGeom"][i] = float(values[1])
            xrotorInputDict["beta0Deg"][i] = float(values[2])

    finally:  # We are done reading
        fid.close()

    # Set the twist at the root to be 90 so that it is continuous on
    # either side of the origin. I.e Across blades' root. Also set
    # the chord to be 0 at the root
    if xrotorInputDict["rRGeom"][0] != 0:
        xrotorInputDict["rRGeom"].insert(0, 0.0)
        xrotorInputDict["cRGeom"].insert(0, 0.0)
        xrotorInputDict["beta0Deg"].insert(0, 90.0)
        xrotorInputDict["nGeomStations"] += 1

    # AdvanceRatio = Vinf/Vtip => Vinf/OmegaR
    xrotorInputDict["omegaDim"] = xrotorInputDict["vel"] / (
        xrotorInputDict["adv"] * xrotorInputDict["rad"]
    )
    xrotorInputDict["RPM"] = xrotorInputDict["omegaDim"] * 30 / pi
    xrotorInputDict["inputType"] = (
        "xrotor"  # we need to store which file format we are using to handle the r vs r/R situation correctly.
    )
    return xrotorInputDict


def floatRange(start, stop, step=1):
    return [float(a) for a in range(start, stop, step)]


########################################################################################################################
def generateTwists(xrotorDict, mesh_unit, length_unit, angle_unit):
    """
    Transform the Xrotor format blade twists distribution into the Flow360 standard.

    Attributes
    ----------
    xrotorDict: dictionary of Xrotor data as read in by def readXROTORFile(xrotorFileName):
    meshUnit: float,  Grid unit length in the mesh.
    return:  list of dictionaries containing the radius ( in grid units) and twist in degrees.
    """
    # generate the twists vector required from the BET input
    twistVec = []
    if xrotorDict["inputType"] == "xrotor":
        multiplier = xrotorDict[
            "rad"
        ]  # X rotor uses r/R we need to convert that to r in mesh units
    elif xrotorDict["inputType"] == "dfdc":
        multiplier = 1.0  # dfdc is already in meters so only need to convert it ot mesh units.

    for i in range(xrotorDict["nGeomStations"]):
        # dimensional radius we are at in grid unit
        r = xrotorDict["rRGeom"][i] * multiplier * u.m / mesh_unit
        twist = xrotorDict["beta0Deg"][i]
        twistVec.append({"radius": r * length_unit, "twist": twist * angle_unit})

    return twistVec


########################################################################################################################
def generateChords(xrotorDict, mesh_unit, length_unit):
    """
    Transform the Xrotor format blade chords distribution into the Flow360 standard.

    Attributes
    ----------
    xrotorDict: dictionary of Xrotor data as read in by def readXROTORFile(xrotorFileName):
    meshUnit: float,  Grid unit length per meter in the mesh. if your grid is in mm then meshUnit = 0.001 meter per mm;
    If your grid is in inches then meshUnit = 0.0254 meter per in etc...
    return:  list of dictionaries containing the radius ( in grid units) and chords in grid units.
    """
    # generate the dimensional chord vector required from the BET input
    chordVec = []
    if xrotorDict["inputType"] == "xrotor":
        multiplier = xrotorDict[
            "rad"
        ]  # X rotor uses r/R we need to convert that to r in mesh units
    elif xrotorDict["inputType"] == "dfdc":
        multiplier = 1.0  # dfdc is already in meters so only need to convert it ot mesh units.
    for i in range(xrotorDict["nGeomStations"]):
        r = xrotorDict["rRGeom"][i] * multiplier * u.m / mesh_unit
        chord = xrotorDict["cRGeom"][i] * multiplier * u.m / mesh_unit
        chordVec.append({"radius": r * length_unit, "chord": chord * length_unit})

    return chordVec


########################################################################################################################
def generateMachs():
    """
    The Flow360 BET input file expects a set of Mach numbers to interpolate
    between using the Mach number the blade sees.
    To that end we will generate 4 different tables at 4 different Mach #s
    equivalent to M^2=0, 1/3, 2/3, 0.9


    Attributes
    ----------
    return: list of floats
    """

    machVec = [0, sqrt(1 / 3), sqrt(2 / 3), sqrt(0.9)]
    return machVec


########################################################################################################################
def generateReys():
    """
    Flow360 has the functionality to interpolate across Reynolds numbers but we are not using that functionality
    just make it a constant 1

    """
    return [1]


########################################################################################################################
def generateAlphas():
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
    negAng = floatRange(-180, -9)
    posAng = [
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
    posAng2 = floatRange(10, 181)
    return negAng + posAng + posAng2
    # return floatRange(-180, 181)


########################################################################################################################
def findClMinMaxAlphas(CLIFT, CLMIN, CLMAX):
    """
    Find the index in the CLIFT list where we are just below the CLMin
    value and the one where we are just above the CLmax value. Use the fact that CL should be continually increasing
    from -pi -> Pi radians.
    The goal of this function is to separate the linear CL regime (i.e. from CLmin to CLmax) and extract its indices
    We Traverse the list from the beginning until we hit CLMIN


    Attributes
    ----------

    CLIFT: list of floats
    CLMIN: float
    CLMAX: float
    return: 2 ints as indices
    """

    clMinIdx = 0  # initialize as the first index
    clMaxIdx = len(CLIFT)  # initialize as the last index
    for i in range(len(CLIFT)):
        if CLIFT[i] < CLMIN:
            clMinIdx = i
        if CLIFT[i] > CLMAX:
            clMaxIdx = i
            break
    return (
        clMinIdx - 1,
        clMaxIdx + 1,
    )  # return the two indices right before and after the two found values.


########################################################################################################################
def blendFuncValue(blendWindow, alpha, alphaMinMax, alphaRange):
    """
    This functions is used to blend the flat plate CL and CD polar to the given Cl and CD polars.
    The returned blend value is 1 when we use the given CL and CD values and 0 when we use the Flat plate values.
    Within the blendWindow range of alphas it returns a COS^2 based smooth blend.

    Attributes
    ----------

        blendWindow: float size of the window we want to blend from the given 2D polar
        alpha: float alpha we are at in radians
        alphaMinMax: float,   alpha min  or alpha max for that 2D polar in radians. Outside of those values we use
    the Flat plate coefficients
        alphaRange: string, used to figure out whether we are doing before CLmin or beyond CLmax
        return: float (blend value for that alpha
    """

    if "aboveCLmax" in alphaRange:
        # we are on the CLMAX side:
        if alpha < alphaMinMax:
            return 1
        if alpha > alphaMinMax + blendWindow:
            return 0
        return cos((alpha - alphaMinMax) / blendWindow * pi / 2) ** 2
    if "belowCLmin" in alphaRange:
        # we are on the CLMIN side:
        if alpha > alphaMinMax:
            return 1
        if alpha < alphaMinMax - blendWindow:
            return 0
        return cos((alpha - alphaMinMax) / blendWindow * pi / 2) ** 2
    else:
        raise ValueError(f"alphaRange must be either aboveCLmax or belowCLmin, it is: {alphaRange}")


########################################################################################################################
def xrotorBlend2flatPlate(CLIFT, CDRAG, alphas, alphaMinIdx, alphaMaxIdx):
    """
     Blend the Clift and Cdrag values outside of the normal working range of alphas to the flat plate CL and CD values.

    Attributes
    ----------
    CLIFT: float
    CDRAG: float
    alphas: list of floats
    alphaMinIdx: int, index within the above list of alphas
    alphaMaxIdx: int, index within the above list of alphas

    return: 2 Floats representing the blended CL and CD at that alpha
    """

    blendWindow = 0.5  # 0.5 radians
    alphaMin = alphas[alphaMinIdx] * pi / 180
    alphaMax = alphas[alphaMaxIdx] * pi / 180

    for i in range(alphaMinIdx):  # from -pi to alphaMin in the CLIFT array
        a = alphas[i] * pi / 180  # alpha in radians

        blendVal = blendFuncValue(
            blendWindow, a, alphaMin, "belowCLmin"
        )  # we are on the alphaCLmin side going up in CL
        # this follows the flat plate lift and drag equations times the blend val coefficient
        CLIFT[i] = CLIFT[i] * blendVal + (1 - blendVal) * cos(a) * 2 * pi * sin(a) / sqrt(
            1 + (2 * pi * sin(a)) ** 2
        )
        CDRAG[i] = (
            CDRAG[i] * blendVal
            + (1 - blendVal) * sin(a) * (2 * pi * sin(a)) ** 3 / sqrt(1 + (2 * pi * sin(a)) ** 6)
            + 0.05
        )

    for j in range(alphaMaxIdx, len(alphas)):  # from alphaMax to Pi in the CLIFT array
        a = alphas[j] * pi / 180  # alpha in radians
        blendVal = blendFuncValue(
            blendWindow, a, alphaMax, "aboveCLmax"
        )  # we are on the alphaCLmax side of things going up in CL
        # this follows the flat plate lift and drag equations times the blend val coefficient
        CLIFT[j] = CLIFT[j] * blendVal + (1 - blendVal) * cos(a) * 2 * pi * sin(a) / sqrt(
            1 + (2 * pi * sin(a)) ** 2
        )
        CDRAG[j] = (
            CDRAG[j] * blendVal
            + (1 - blendVal) * sin(a) * (2 * pi * sin(a)) ** 3 / sqrt(1 + (2 * pi * sin(a)) ** 6)
            + 0.05
        )
    return CLIFT, CDRAG


########################################################################################################################
def calcClCd(xrotorDict, alphas, machNum, nrRstation):
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
    xrotorDict: dictionary of Xrotor data as read in by def readXROTORFile(xrotorFileName):
    alphas: list of ints, alphas we have for the polar.
    machNum: float, mach number we do this polar at.
    nrRstation: int, which r/R station we have to define this polar for.
    return: 2 list of floats representing the CL and CD for  that polar
    """

    CDMFACTOR = 10.0
    CLMFACTOR = 0.25
    MEXP = 3.0
    CDMDD = 0.0020
    CDMSTALL = 0.1000

    # Prandtl-Glauert compressibility factor
    MSQ = machNum**2

    if MSQ > 1.0:
        print("CLFUNC: Local Mach^2 number limited to 0.99, was ", MSQ)
        MSQ = 0.99

    PG = 1.0 / sqrt(1.0 - MSQ)
    MACH = machNum

    # Generate CL from dCL/dAlpha and Prandtl-Glauert scaling
    A_zero = xrotorDict["a0deg"][nrRstation] * pi / 180
    DCLDA = xrotorDict["dclda"][nrRstation]

    CLA = [0] * len(alphas)
    for i, a in enumerate(alphas):
        CLA[i] = DCLDA * PG * ((a * pi / 180) - A_zero)
    CLA = array(CLA)

    # Reduce CLmax to match the CL of onset of serious compressible drag
    CLMAX = xrotorDict["clmax"][nrRstation]
    CLMIN = xrotorDict["clmin"][nrRstation]
    CLDMIN = xrotorDict["clcdmin"][nrRstation]
    MCRIT = xrotorDict["mcrit"][nrRstation]

    DMSTALL = (CDMSTALL / CDMFACTOR) ** (1.0 / MEXP)
    CLMAXM = max(0.0, (MCRIT + DMSTALL - MACH) / CLMFACTOR) + CLDMIN
    CLMAX = min(CLMAX, CLMAXM)
    CLMINM = min(0.0, -(MCRIT + DMSTALL - MACH) / CLMFACTOR) + CLDMIN
    CLMIN = max(CLMIN, CLMINM)

    # CL limiter function (turns on after +-stall)
    DCL_STALL = xrotorDict["dclstall"][nrRstation]
    ECMAX = expList(clip((CLA - CLMAX) / DCL_STALL, -inf, 200))
    ECMIN = expList(clip((CLA * (-1) + CLMIN) / DCL_STALL, -inf, 200))
    CLLIM = logList((ECMAX + 1.0) / (ECMIN + 1.0)) * DCL_STALL

    # Subtract off a (nearly unity) fraction of the limited CL function
    # This sets the dCL/dAlpha in the stalled regions to 1-FSTALL of that
    # in the linear lift range
    DCLDA_STALL = xrotorDict["dcldastall"][nrRstation]
    FSTALL = DCLDA_STALL / DCLDA
    CLIFT = CLA - CLLIM * (1.0 - FSTALL)

    # In the basic linear lift range drag is a quadratic function of lift
    # CD = CD0 (constant) + quadratic with CL)
    CDMIN = xrotorDict["cdmin"][nrRstation]
    DCDCL2 = xrotorDict["dcddcl2"][nrRstation]

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
    for critMachIdx in range(len(CRITMACH)):
        if MACH < CRITMACH[critMachIdx]:
            continue
        else:
            CDC[critMachIdx] = CDMFACTOR * (MACH - CRITMACH[critMachIdx]) ** MEXP

    # you could use something like this to add increase drag by Prandtl-Glauert
    # (or any function you choose)
    FAC = 1.0
    # --- Total drag terms
    CDRAG = CDRAG * FAC + DCD + CDC

    # Now we modify the Clift and CDrag outside of the large alpha range to smooth out
    # the Cl and CD outside of the expected operating range

    # Find the Alpha for ClMax and CLMin
    alphaMinIdx, alphaMaxIdx = findClMinMaxAlphas(CLIFT, CLMIN, CLMAX)
    # Blend the CLIFt and CDRAG values from above with the flat plate formulation to
    # be used outside of the alphaCLmin to alphaCLMax window
    CLIFT, CDRAG = xrotorBlend2flatPlate(CLIFT, CDRAG, alphas, alphaMinIdx, alphaMaxIdx)

    return list(CLIFT), list(CDRAG)


########################################################################################################################
def getPolar(xrotorDict, alphas, machs, rRstation):
    """
    Return the 2D Cl and CD polar expected by the Flow360 BET model.
    b/c we have 4 Mach Values * 1 Reynolds value we need 4 different arrays per sectional polar as in:
    since the order of brackets is Mach#, Rey#, Values then we need to return:
    [[[array for MAch #1]],[[array for MAch #2]],[[array for MAch #3]],[[array for MAch #4]]]


    Attributes
    ----------
    xrotorDict: dictionary of Xrotor data as read in by def readXROTORFile(xrotorFileName):
    alphas: list of floats
    machs: list of float
    rRstation: station index.
    return: list of dictionaries
    """

    secpol = {}
    secpol["lift_coeffs"] = []
    secpol["drag_coeffs"] = []
    for machNum in machs:
        cl, cd = calcClCd(xrotorDict, alphas, machNum, rRstation)
        secpol["lift_coeffs"].append([cl])
        secpol["drag_coeffs"].append([cd])
    return secpol


########################################################################################################################
def generateXrotorBETJSON(
    xrotorFileName,
    rotation_direction_rule,
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
    xrotorFileName: string, filepath to the Xrotor/DFDC file we want to translate into a BET disk
    betDisk: This is a dict that already contains some betDisk definition information. We will add to that same dict
    before returning it.
    meshUnit is in gridUnits per meters, if your grid is in mm then meshUnit = 0.001 grid Unit per meter.
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

    xrotorDict = readXROTORFile(xrotorFileName)

    betDisk = {}
    betDisk["entities"] = cylinder
    betDisk["omega"] = omega
    betDisk["chord_ref"] = chord_ref
    betDisk["n_loading_nodes"] = n_loading_nodes
    betDisk["rotation_direction_rule"] = rotation_direction_rule
    betDisk["number_of_blades"] = xrotorDict["nBlades"]
    betDisk["radius"] = xrotorDict["rad"] * u.m / mesh_unit
    betDisk["twists"] = generateTwists(
        xrotorDict, mesh_unit=mesh_unit, length_unit=length_unit, angle_unit=angle_unit
    )
    betDisk["chords"] = generateChords(xrotorDict, mesh_unit=mesh_unit, length_unit=length_unit)
    betDisk["mach_numbers"] = generateMachs()
    betDisk["alphas"] = generateAlphas()
    betDisk["reynolds_numbers"] = generateReys()
    betDisk["sectional_radiuses"] = [
        betDisk["radius"] * r for r in xrotorDict["rRstations"]
    ] * length_unit
    betDisk["sectional_polars"] = []

    for secId in range(0, xrotorDict["nAeroSections"]):
        polar = getPolar(xrotorDict, betDisk["alphas"], betDisk["mach_numbers"], secId)
        betDisk["sectional_polars"].append(polar)

    betDisk["alphas"] *= angle_unit
    betDisk.pop(
        "radius", None
    )  # radius is only needed to get sectional_radiuses but not by the solver.

    # with open("bet_translator_dict.json", "w") as file1:
    #     json.dump(betDisk, file1, indent=4)

    return betDisk


########################################################################################################################
def test_translator():
    """
    run the translator with a representative set of inputs
    dumps betDisk JSON file that can be added to a Flow360 JSON file.

    meshUnit is in gridUnits per meters, if your grid is in mm then meshUnit = 0.001 grid Unit per meter.
     If your grid is in inches then meshUnit = 0.0254 grid Unit per meter etc...
    """
    betDiskDict = {
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
    xrotorFileName = "examples/xrotorTranslator/ecruzer.prop"

    xrotorInputDict = generateXrotorBETJSON(xrotorFileName, betDiskDict)
    betDiskJson = {
        "BETDisks": [xrotorInputDict]
    }  # make all that data a subset of BETDisks dictionary, notice the [] b/c
    # the BETDisks dictionary accepts a list of bet disks
    # dump the sample dictionary to a json file
    json.dump(betDiskJson, open("sampleBETJSON.json", "w"), indent=4)


########################################################################################################################
if __name__ == "__main__":
    # if run on its own, then just run the test_translator() function
    test_translator()
