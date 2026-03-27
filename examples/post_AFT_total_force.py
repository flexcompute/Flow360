"""
Post-processing script for AFT (Aerodynamic Force and Torque) total forces.

Reads total_forces_v2.csv outputs from multiple Flow360 cases, aggregates
aerodynamic coefficients across angle-of-attack (AOA) sweeps, and produces
comparison plots between solver releases. Supports both steady and unsteady
(time-averaged) workflows.

Usage:
    python post_AFT_total_force_v2.py

Configuration is read from ./config_files/config.json. Case IDs for each
configuration are read from text files under <rootfolder>/caseIDfiles/.
"""

import json
import os
import re
import shutil
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas

import flow360
import flow360 as fl
from flow360.cloud.http_util import api_key_auth, http
from flow360.log import log, set_logging_level
from flow360.user_config import UserConfig

# Solver version environment variables (used for labeling/comparison workflows)
SOLVER_VERSION = os.environ.get("solverVersion", "")
SOLVER_VERSION_REF = os.environ.get("solverVersionRef", "")
# Local path where runtime support logs are downloaded
RESULTS_PATH = "./resultsPath"

UserConfig.set_profile("rui.cheng@flexcompute.com")
set_logging_level("INFO")

flow360.Env.prod.active()
# Color map for up to 8 cases: first 4 use solid lines, next 4 use dashed lines
cmap = matplotlib.colormaps['tab10']
plt.rcParams['font.size'] = 16


def read_case_config(config_file):
    """Load the JSON configuration file that defines cases and plot settings."""
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def read_caseID(file_name):
    """Read a list of Flow360 case IDs from a plain-text file (one ID per line)."""
    data_array = []
    try:
        with open(file_name, 'r') as file:
            for line in file:
                data_array.append(line.strip())
        return data_array
    except FileNotFoundError:
        print(f"The file at {file_name} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def fetch_data(case_id):
    """Download total_forces_v2.csv for a case and rename it with the case ID prefix."""
    case = flow360.Case(case_id)
    case.results.download(total_forces=True, surface_forces=False, bet_forces=False, cfl=False)
    forcename = case_id + '_total_forces_v2.csv'
    os.rename('total_forces_v2.csv', forcename)
    return forcename


def get_data_at_last_pseudo_step(filename):
    """
    Extract force data at the last pseudo-step of each physical step.

    For each physical step, only the final pseudo-step row is retained, which
    represents the converged value for that step.

    Returns a dict mapping column names to lists of values (one per physical step).
    """
    dataframe = pandas.read_csv(filename, skipinitialspace=True)
    # Drop any unnamed index columns that pandas may add
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
    data_raw = dataframe.to_dict("list")
    data = {key: [] for key in data_raw.keys()}

    n = len(data_raw['physical_step'])
    j = 0
    # Detect physical step boundaries by looking for step number changes
    for i in range(0, n - 1):
        if data_raw['physical_step'][i] != data_raw['physical_step'][i + 1]:
            j += 1
            for key in data_raw.keys():
                data[key].append(data_raw[key][i])
    print("total_physical_steps=", j)
    # Append the final row (last pseudo-step of the last physical step)
    for key in data_raw.keys():
        data[key].append(data_raw[key][n - 1])
    return data


def get_data_last_aver_npseudo_step(filename, npseduo):
    """
    Average the last 10% of pseudo-steps within each physical step.

    The averaging window is computed per physical step as:
        max(1, int(num_pseudo_steps_in_step * 0.1))
    This automatically adapts to the actual pseudo-step count rather than
    relying on the `npseduo` input (which is retained for signature compatibility).

    Used for steady cases where final pseudo-step values may not be fully
    converged; averaging smooths residual oscillations.

    Returns a dict mapping column names to lists of averaged values (one per
    physical step).
    """
    dataframe = pandas.read_csv(filename, skipinitialspace=True)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
    data_raw = dataframe.to_dict("list")
    data = {key: [] for key in data_raw.keys()}

    n = len(data_raw['physical_step'])
    j = 0
    step_start = 0  # index of the first row of the current physical step
    for i in range(0, n):
        # Treat end-of-file as an implicit step boundary for the last physical step
        if i == n - 1 or data_raw['physical_step'][i] != data_raw['physical_step'][i + 1]:
            j += 1
            num_pseudo = i - step_start + 1
            # Use last 10% of pseudo-steps, at least 1
            npseduo_auto = max(1, int(num_pseudo * 0.1))
            for key in data_raw.keys():
                # Average the window of pseudo-steps ending at index i
                meanvalue = sum(data_raw[key][i - npseduo_auto + 1:i + 1]) / npseduo_auto
                data[key].append(meanvalue)
            step_start = i + 1  # next physical step starts at the following row
    print("total_physical_steps=", j)
    return data


def plot_line(i, j, coll, forces, forcename, label, axs, case, colori=0):
    """
    Plot a single force coefficient curve onto subplot axs[i, j].

    Cases 0-3 use solid lines; cases 4-7 use dashed lines with recycled colors,
    allowing up to 8 distinguishable curves.
    """
    if colori < 4:
        axs[i, j].plot(coll, forces[forcename], '-', label=case, color=cmap.colors[colori], linewidth=3)
    else:
        axs[i, j].plot(coll, forces[forcename], '--', label=case, color=cmap.colors[colori - 4], linewidth=3)
    axs[i, j].set_ylabel(label)


def plot_forces(folder, case, forces, coll, forcestoplot, figure_extname):
    """
    Plot a 2x3 grid of force coefficients for a single case and save to disk.

    The figure is saved under <folder>/figures/<figure_extname>/total_force/.
    """
    labels = forcestoplot
    jrange = [0, 1, 2]
    irange = [0, 1]
    ni = len(irange)
    nj = len(jrange)
    fig, axs = plt.subplots(ni, nj, figsize=(28, 14), layout='constrained')
    for i in irange:
        for j in jrange:
            index = i * nj + j
            plot_line(i, j, coll, forces, forcestoplot[index], labels[index], axs, case)
            axs[i, j].set_xlabel('alpha [deg]')
            axs[i, j].grid()
            axs[i, j].tick_params(axis='both', labelsize=14)
    figurename = os.path.join(folder, "figures", figure_extname, "total_force", case + "_forces_coeff.png")
    print(figurename)
    plt.savefig(figurename, dpi=500, bbox_inches='tight')


def plot_forces_diff(folder, case, forces, coll, forcestoplot, figure_extname):
    """
    Plot a 2x3 grid of force coefficient differences (delta values) and save to disk.

    Used to visualize the deviation between two solver versions or configurations.
    The figure is saved under <folder>/figures/<figure_extname>/total_force/.
    """
    labels = [f"delta_{v}" for v in forcestoplot]
    jrange = [0, 1, 2]
    irange = [0, 1]
    ni = len(irange)
    nj = len(jrange)
    fig, axs = plt.subplots(ni, nj, figsize=(28, 14), layout='constrained')
    for i in irange:
        for j in jrange:
            index = i * nj + j
            plot_line(i, j, coll, forces, forcestoplot[index], labels[index], axs, case)
            axs[i, j].set_xlabel('alpha [deg]')
            axs[i, j].grid()
    figurename = os.path.join(folder, "figures", figure_extname, "total_force", case + "_delta_forces_coeff.png")
    print(figurename)
    plt.savefig(figurename, dpi=500, bbox_inches='tight')


def readtestdata():
    """
    Return HLPW4 wind-tunnel reference data (flap deflections 40/37 deg).

    Used as experimental baseline for CL, CD, and CMy comparisons.
    """
    # flap 40/37 for HLPW4
    testAOA = [-3.803630114, 0.630306005, 2.781749964, 4.937990189, 6.033500195, 7.045030117,
               8.113100052, 9.171830177, 10.24489975, 11.2947998, 12.35690022, 13.4066,
               14.46520042, 15.53349972, 16.01869965, 16.53619957, 17.04520035, 18.05450058,
               18.56760025, 19.07299995, 19.57209969, 20.54969978, 21.46619987]
    testCD = [0.10334, 0.116306998, 0.134581, 0.159256995, 0.173684999, 0.186706007,
              0.201723993, 0.218577996, 0.235310003, 0.249883994, 0.265890002, 0.282851994,
              0.298635006, 0.313288003, 0.320418, 0.326321006, 0.332423002, 0.343849987,
              0.349783003, 0.356994003, 0.361550003, 0.367354006, 0.436477989]
    testCL = [0.586673021, 1.156720042, 1.362380028, 1.584380031, 1.688420057, 1.778620005,
              1.877210021, 1.977859974, 2.0697999, 2.149060011, 2.228889942, 2.312809944,
              2.380460024, 2.438509941, 2.462029934, 2.478869915, 2.495490074, 2.51967001,
              2.524410009, 2.523799896, 2.514909983, 2.463219881, 2.318239927]
    testCMy = [-0.336261004, -0.399899989, -0.38962701, -0.381300986, -0.375851989, -0.370599985,
               -0.363555998, -0.357874006, -0.353008986, -0.344168007, -0.338178992, -0.334307998,
               -0.323486, -0.310099006, -0.303438008, -0.294616997, -0.287313014, -0.270090997,
               -0.261483014, -0.25413999, -0.244862005, -0.232053995, -0.303606004]
    return testAOA, testCL, testCD, testCMy


def readvolantdata():
    """Return Volant rotor reference data (RPM, CFz thrust, CMz torque)."""
    testAOA = [1500]
    testCFz = [14.947]
    testCMz = [2.0163]
    return testAOA, testCFz, testCMz


def plot_forces_comp(folder, cases, fname, forces, coll, forcestoplot, testdata, xlabel, figure_extname, delta=False, title="Solver release test"):
    """
    Plot overlaid force coefficient curves for multiple cases and save to disk.

    Optionally overlays HLPW4 experimental reference data when testdata=True.
    The figure is saved under <folder>/figures/forces/<fname>.png.

    Args:
        folder: Root folder name (also used as the plot title source).
        cases: List of case label strings.
        fname: Output filename (without path or extension).
        forces: Dict indexed by case number, each value is a dict of force arrays.
        coll: X-axis values (e.g., AOA list).
        forcestoplot: List of 6 force coefficient column names to display.
        testdata: If True, overlay HLPW4 wind-tunnel reference data.
        xlabel: Label string for the x-axis.
    """
    ncase = len(cases)
    labels = [f"delta_{v}" for v in forcestoplot] if delta else forcestoplot
    jrange = [0, 1, 2]
    irange = [0, 1]
    ni = len(irange)
    nj = len(jrange)
    fig, axs = plt.subplots(ni, nj, figsize=(28, 14))
    plt.subplots_adjust(wspace=0.22, hspace=0.2)

    for ii in range(0, ncase):
        for i in irange:
            for j in jrange:
                index = i * nj + j
                plot_line(i, j, coll, forces[ii], forcestoplot[index], labels[index], axs, cases[ii], ii)

    # adding reference test data for comparison
    if testdata:
        testAOA, testCL, testCD, testCMy = readtestdata()
        axs[0, 0].plot(testAOA, testCL, 'o', label='test', color='k')
        axs[0, 1].plot(testAOA, testCD, 'o', label='test', color='k')
        if 'CMy' in forcestoplot:
            idx = forcestoplot.index('CMy')
            axs[idx // nj, idx % nj].plot(testAOA, testCMy, 'o', label='test', color='k')

    axs[0, 1].set_title(title, fontsize=24)
    axs[0, 0].legend(fontsize=18, framealpha=0.8)

    for i in [0, 1]:
        for j in [0, 1, 2]:
            axs[i, j].set_xlabel(xlabel, fontsize=18, fontweight='bold')
            axs[i, j].set_ylabel(axs[i, j].get_ylabel(), fontsize=18, fontweight='bold')
            axs[i, j].tick_params(axis='both', labelsize=16)
            axs[i, j].grid()

    figurename = os.path.join(folder, "figures", figure_extname, "total_force", fname + ".png")
    print("figure name=", figurename)
    plt.savefig(figurename, dpi=500, bbox_inches='tight')


def list_difference(list1, list2):
    """Compute element-wise difference list1[i] - list2[i]."""
    return [list1[i] - list2[i] for i in range(len(list1))]


def _getCaseRuntimeStats(case: flow360.Case):
    """
    Download support logs for a case from the admin API and extract runtime info.

    Downloads a zip archive of solver support logs to RESULTS_PATH/<case.name>/
    and returns the path to that directory.  Skips the download if the zip is
    already present (cached from a previous run).
    """
    logpath = os.path.join(RESULTS_PATH, case.name)
    if not os.path.exists(logpath):
        os.mkdir(logpath)

    zipFile = os.path.join(RESULTS_PATH, case.name, "supportLogs.zip")
    if os.path.exists(zipFile):
        print(f"  [cache] support logs already downloaded: {zipFile}")
    else:
        queryUrl = f"https://admin-api.simulation.cloud/admin/jobs/support/logs/resource/{case.info.user_id}/{case.id}"
        resp = http.session.get(queryUrl, auth=api_key_auth)
        with open(zipFile, "wb") as f:
            f.write(resp.content)
        shutil.unpack_archive(zipFile, logpath)

    print(logpath)
    return logpath


# Mapping from node name patterns to GPU model names (sourced from postprocessing_utils.py)
clusterNameToGPUName = {
    r"cell\d+": "NVIDIA A100-SXM4-80GB",
    r"h200-\d+": "NVIDIA H200-SXM5-141GB",
    r"b200-\d+": "NVIDIA B200-SXM-180GB",
    "dev1": "NVIDIA GeForce RTX 3080-10GB",
    "dev4": "NVIDIA GeForce RTX 3090 Ti-24GB",
    "dev001": "NVIDIA RTX A5000-24GB",
    "dev002": "NVIDIA RTX A5000-24GB",
}


def _getNodesAndMPIRanks(filePath):
    """Parse solver log to extract node names, MPI rank count, and GPU types."""
    nodeNames = set()
    mpiRanks = []
    with open(filePath, "r") as fh:
        for line in fh:
            ret = re.findall(r"\(Rank ([0-9]+)\) Node ([0-9A-Za-z\-]+)", line)
            if len(ret) > 0:
                assert len(ret) == 1
                nodeNames.add(ret[0][1])
                mpiRanks.append(int(ret[0][0]))

    def compareNodeName(e):
        ret = re.findall("([0-9]+)", e)
        return int(ret[0])

    nodeList = sorted(list(nodeNames), key=compareNodeName)
    GPUList = set()
    for nodeName in nodeList:
        for pattern, GPUName in clusterNameToGPUName.items():
            if re.match(pattern, nodeName):
                GPUList.add(GPUName)
    return nodeList, max(mpiRanks) + 1, list(GPUList)


def _getSimulationStats(solverOut):
    """
    Parse solver stdout to extract node list, MPI rank count, GPU list, and wall time.

    Returns a dict with keys: nodeList, numOfRanks, GPU, runTimeInSeconds.
    Falls back to default sentinel values ("N/A" / -1) if parsing fails.
    """
    nodeList = ["N/A"]
    numRanks = -1
    runTime = -1
    GPUList = ["N/A"]
    try:
        nodeList, numRanks, GPUList = _getNodesAndMPIRanks(solverOut)
        runTime = _getTotalRunTime(solverOut)
    except Exception:
        pass
    return {
        "nodeList": nodeList,
        "numOfRanks": numRanks,
        "GPU": GPUList,
        "runTimeInSeconds": runTime,
    }


def _getTotalRunTime(filePath):
    """
    Parse total wall-clock time from a solver log file.

    Searches for the line 'Wall clock time for time marching: X seconds'
    and returns X as a float. Returns -1 if not found.
    """
    runTime = -1
    with open(filePath, "r") as fh:
        for line in fh:
            ret = re.findall(r"Wall clock time for time marching: ([0-9\.]+) seconds", line)
            if len(ret) > 0:
                assert len(ret) == 1
                runTime = float(ret[0])
    return runTime


def _getClusterFromNodeName(nodeList):
    """
    Infer cluster identity from node hostnames.

    Recognizes on-prem cells (cell<N>), Google Cloud workers, a5k nodes,
    H200 nodes, and B200 nodes. Returns a set of cluster identifiers.
    """
    clusters = set()
    for nodeName in nodeList:
        if re.match("cell[0-9]+", nodeName):
            names = re.findall("cell[0-9]+", nodeName)
            assert len(names) == 1
            clusters.add(names[0])
        elif re.match("cloud-google-worker", nodeName):
            clusters.add("gcloud")
        elif re.match("a5k-[0-9]{3}", nodeName):
            clusters.add(nodeName)
        elif re.match(r"h200\-[0-9]{3}", nodeName):
            names = re.findall(r"h200\-[0-9]{3}", nodeName)
            assert len(names) == 1
            clusters.add(names[0])
        elif re.match(r"b200\-[0-9]{3}", nodeName):
            names = re.findall(r"b200\-[0-9]{3}", nodeName)
            assert len(names) == 1
            clusters.add(names[0])
        else:
            log.warning("Unknown node name: {}".format(nodeName))
            clusters.add(nodeName)
    return clusters


def getRuntimeTable(case: fl.Case, caseRef: fl.Case = None):
    """Fetch and print runtime stats for a case; returns the stats dict."""
    stats = [_getCaseRuntimeStats(case)]
    print(stats[0])
    return stats[0]


def main(config_file="./config_files/config.json"):
    """
    Main workflow: read config, fetch/load force data, and generate comparison plots.

    Steps:
    1. Bootstrap force key structure from a reference case.
    2. Read config.json for case names, release tags, AOA lists, and plot settings.
    3. For each case: load CSV data (fetching from Flow360 if needed), aggregate by
       physical step (steady: average last N pseudo-steps; unsteady: average last
       N_period physical steps), and plot per-case forces.
    4. Produce a multi-case overlay comparison figure.
    5. Produce pairwise difference figures between consecutive cases.
    """

    # Note: the following config fields are deprecated but retained for backward compatibility:
    #   - npseduos: replaced by automatically averaging the last 10% of pseudo-steps
    #   - datafileexist: replaced by automatic detection of existing local CSV files

    config = read_case_config(config_file)
    rootfolder = config["rootfolder"]
    casenames = config["casenames"]
    releases = config["releases"]
    subcases = config["subcases"]
    datafileexist = config["datafileexist"]  # 1 = fetch from cloud, 0 = local file exists
    AOAs = config["AOAs"]
    figure_extname = config["figure_extname"]
    testdata = config["testdata"]      # whether to overlay wind-tunnel reference data
    rotorflag = config["rotorflag"]    # True for rotor cases (use CFx/CFy/CFz instead of CL/CD)
    wholeplane = config["wholeplane"]  # True for full-aircraft cases (include skin friction / pressure drag)
    xlabel = config["xlabel"]
    nperiod = config["nperiod"]        # number of physical periods to average (>=2 = unsteady)
    scales = config["scales"]          # normalization scale factor per case
    npseduos = config["npseduos"]      # number of pseudo-steps to average per physical step
    feature_test = config.get("feature_test", False)
    plot_title = "Solver feature test" if feature_test else "Solver release test"

    # Bootstrap: load the first case ID from the first case's ID file to discover
    # all force coefficient column keys without relying on a hardcoded case ID.
    bootstrap_caseID_file = (
        rootfolder + '/caseIDfiles/'
        + casenames[0] + '_' + releases[0] + '_' + subcases[0] + '.txt'
    )
    bootstrap_case_ids = read_caseID(bootstrap_caseID_file)
    bootstrap_case_id = bootstrap_case_ids[0]
    bootstrap_file = (
        os.path.join(rootfolder, "data", casenames[0] + '_' + releases[0])
        + '/' + bootstrap_case_id + '_total_forces_v2.csv'
    )
    if datafileexist[0] == 1:
        bootstrap_file = fetch_data(bootstrap_case_id)
    forces = get_data_at_last_pseudo_step(bootstrap_file)

    ### initalize the data.
    allforces = {}
    diffs = {}
    for key in forces.keys():
        diffs[key] = []
        allforces[key] = []

    # Remove the bootstrap CSV — it was only needed to discover column keys.
    if datafileexist[0] == 1 and os.path.exists(bootstrap_file):
        os.remove(bootstrap_file)

    # Select which 6 force coefficients to display based on simulation type
    if rotorflag:
        forcestoplot = ['CFx', 'CFy', 'CFz', 'CMx', 'CMy', 'CMz']         # rotor: body-axis forces
    elif wholeplane:
        forcestoplot = ['CL', 'CD', 'CDSkinFriction', 'CDPressure', 'CMx', 'CMy']  # full aircraft: include drag breakdown
    else:
        forcestoplot = ['CL', 'CD', 'CFy', 'CMx', 'CMy', 'CMz']           # default: wing/component

    # Create figure output subdirectories
    for subdir in ["total_force", "residual", "forcehistory"]:
        path = os.path.join(rootfolder, "figures", figure_extname, subdir)
        os.makedirs(path, exist_ok=True)

    cases = []
    ncases = len(casenames)
    print("############################################")
    print("   total ", ncases, " cases are compared in these test")

    for i in range(0, ncases):
        scale = scales[i] ## scale is used when the data from different case use diff reference.
        n_physicalstep = nperiod[i]
        npseduo = npseduos[i]

        forcearray = {key: [] for key in forces.keys()}

        krylov_suffix  = "_krylov"  if "krylov"  in subcases[i].lower() else ""
        gravity_suffix = "_gravity" if "gravity" in subcases[i].lower() else ""
        cases.append(casenames[i] + '_' + releases[i] + krylov_suffix + gravity_suffix)
        path = os.path.join(rootfolder, "data", casenames[i] + '_' + releases[i])
        print("##############################################################")
        print("Case:", i, "PATH=", path)
        figurepath = os.path.join(path, "figures")
        for folder in [path, figurepath]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Folder created: {folder}")
            else:
                print(f"Folder already exists: {folder}")

        caseID_file = rootfolder + '/caseIDfiles/' + casenames[i] + '_' + releases[i] + '_' + subcases[i] + '.txt'
        print("read caseID_file=", caseID_file)
        case_ids = read_caseID(caseID_file)

        if case_ids is not None:
            print("Data read from the following case_ids:")
            print(case_ids)

        for case_id in case_ids:
            newname = path + '/' + case_id + '_total_forces_v2.csv'
            if not os.path.exists(newname):
                # local file not found: fetch from Flow360 and move into place
                cvsforces_file = fetch_data(case_id)
                shutil.move(cvsforces_file, newname)
            else:
                print(f"Using existing local file: {newname}")

            # n_physicalstep < 2: steady, average last npseduo pseudo steps
            # n_physicalstep >= 2: unsteady, average last n_physicalstep physical steps
            if n_physicalstep >= 2:
                print("for rotor case, suggest in the config.json, use the rotor rotation period for data average")
                forces = get_data_at_last_pseudo_step(newname)
                for key in forces.keys():
                    # Average over the last n_physicalstep physical steps and apply scale
                    meanvalue = sum(forces[key][-(n_physicalstep + 1):-1]) / n_physicalstep / scale
                    forcearray[key].append(meanvalue)
            else:
                forces = get_data_last_aver_npseudo_step(newname, npseduo)
                for key in forces.keys():
                    forcearray[key].append(forces[key][-1] / scale)
                

        #plot_forces(rootfolder, cases[i], forcearray, AOAs, forcestoplot, figure_extname)
        allforces[i] = forcearray

    print("###########################################################")
    print("print out CL for data verify")
    for i in range(0, ncases):
        print(i, "CL", allforces[i]["CL"])
        print(i, "CD", allforces[i]["CD"])
    print("###########################################################")

    # Multi-case overlay comparison figure
    figurename = rootfolder + "_forces_coeff_compare" + figure_extname
    plot_forces_comp(rootfolder, cases, figurename, allforces, AOAs, forcestoplot, testdata, xlabel, figure_extname, title=plot_title)

    # Pairwise difference figures: each consecutive pair of cases
    diffs = {}
    diffnames = []
    for i in range(0, ncases - 1):
        diff = {key: list_difference(allforces[i][key], allforces[i + 1][key]) for key in forces.keys()}
        diffs[i] = diff
        diffnames.append(cases[i] + '-' + cases[i + 1])

    figurename = rootfolder + "_forces_coeff_diff" + figure_extname
    plot_forces_comp(rootfolder, diffnames, figurename, diffs, AOAs, forcestoplot, False, xlabel, figure_extname, delta=True, title=plot_title)

    # Collect runtime stats for all cases, print summary, and save as a table figure.
    # - Timing/worker fields come from case.get() (raw API response)
    # - GPU type and exact MPI rank count are parsed from the solver log file
    import glob as _glob
    runtime_rows = []
    print("###########################################################")
    print("Runtime summary:")
    for i in range(0, ncases):
        caseID_file = rootfolder + '/caseIDfiles/' + casenames[i] + '_' + releases[i] + '_' + subcases[i] + '.txt'
        ids = read_caseID(caseID_file)
        if not ids:
            continue
        print(f"  [{cases[i]}]")
        path = os.path.join(rootfolder, "data", cases[i])
        for j, case_id in enumerate(ids):
            aoa = AOAs[j] if j < len(AOAs) else 'N/A'
            case = fl.Case(case_id)
            raw = case.get()
            # Compute elapsed time from start/finish timestamps to explicitly exclude queue wait time.
            # elapsedTimeInSeconds from the API equals caseFinishTime - caseStartTime (not submit time),
            # but we recalculate here for clarity.
            start_str = raw.get('caseStartTime')
            finish_str = raw.get('caseFinishTime')
            if start_str and finish_str:
                from datetime import datetime, timezone
                fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
                elapsed_s = (datetime.strptime(finish_str, fmt) - datetime.strptime(start_str, fmt)).total_seconds()
                elapsed_min = f"{elapsed_s / 60:.1f}"
            else:
                elapsed_min = 'N/A'
            ranks = raw.get('numProcessors', 'N/A')
            worker = raw.get('worker', 'N/A')
            # Parse GPU type and exact rank count from solver log
            logdir = _getCaseRuntimeStats(case)
            solverLogs = _glob.glob(os.path.join(logdir, 'casePipeline.Flow360Solver.*.log'))
            if solverLogs:
                stats = _getSimulationStats(solverLogs[0])
                ranks = stats['numOfRanks']
                gpu = ', '.join(stats['GPU']) if stats['GPU'] else 'N/A'
            else:
                gpu = 'N/A'
            # Simplify worker label to GPU family name
            for tag in ['B200', 'H200', 'A100']:
                if tag in gpu:
                    worker = tag
                    break
            # Get total pseudo steps from the force CSV:
            # for each physical step, sum (last pseudo_step - first pseudo_step)
            csv_file = os.path.join(path, case_id + '_total_forces_v2.csv')
            if os.path.exists(csv_file):
                df_tmp = pandas.read_csv(csv_file, skipinitialspace=True)
                df_tmp.columns = df_tmp.columns.str.strip()
                grouped = df_tmp.groupby('physical_step')['pseudo_step']
                total_pseudo_steps = int((grouped.max() - grouped.min()+1).sum())
            else:
                total_pseudo_steps = 'N/A'
            print(f"    {case_id} ({xlabel}={aoa}): elapsedTime={elapsed_min}min  worker={worker}  ranks={ranks}  totalPseudoSteps={total_pseudo_steps}")
            runtime_rows.append([cases[i], case_id[:13], aoa, elapsed_min, worker, ranks, total_pseudo_steps])
    print("###########################################################")

    # Save runtime summary as table figures (max 9 rows per table)
    if runtime_rows:
        col_labels = ['Case', 'Case ID', xlabel, 'ElapsedTime (min)', 'Worker', 'MPI Ranks', 'Total Pseudo Steps']
        row_height = 0.1
        figurepath = os.path.join(rootfolder, "figures", figure_extname, "total_force")
        os.makedirs(figurepath, exist_ok=True)
        max_rows_per_table = 9
        num_tables = (len(runtime_rows) + max_rows_per_table - 1) // max_rows_per_table
        for t in range(num_tables):
            chunk = runtime_rows[t * max_rows_per_table:(t + 1) * max_rows_per_table]
            fig, ax = plt.subplots(figsize=(20, max(2, len(chunk) * 0.55 + 1.5)))
            ax.axis('off')
            tbl = ax.table(cellText=chunk, colLabels=col_labels, loc='center', cellLoc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(12)
            tbl.auto_set_column_width(col=list(range(len(col_labels))))
            for (row, col), cell in tbl.get_celld().items():
                cell.set_height(row_height)
            for col in range(len(col_labels)):
                tbl[0, col].set_facecolor('#d0d0d0')
                tbl[0, col].set_text_props(fontweight='bold')
            title = f'Runtime Summary ({t + 1}/{num_tables})' if num_tables > 1 else 'Runtime Summary'
            ax.set_title(title, fontsize=15, pad=10)
            suffix = f"_part{t + 1}" if num_tables > 1 else ""
            figurename = os.path.join(figurepath, rootfolder + f"_runtime_summary{suffix}" + figure_extname)
            print("Runtime table figure:", figurename)
            plt.savefig(figurename, dpi=200, bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':
    main()
