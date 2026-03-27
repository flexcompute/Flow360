"""
Post-processing script for AFT force/residual convergence history.

Reads total_forces_v2.csv and nonlinear_residual_v2.csv outputs from multiple
Flow360 cases and produces convergence history comparison plots between solver
releases. Supports steady and unsteady workflows.

Usage:
    python post_AFT_forces_history_V5.py

Configuration is read from ./config_files/config.json. Case IDs for each
configuration are read from text files under <rootfolder>/caseIDfiles/.
"""

import json
import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas

import flow360
import flow360 as fl
from flow360.log import set_logging_level
from flow360.user_config import UserConfig

UserConfig.set_profile("rui.cheng@flexcompute.com")
set_logging_level("INFO")

flow360.Env.prod.active()
cmap = matplotlib.colormaps['tab10']
plt.rcParams['font.size'] = 18


# ---------------------------------------------------------------------------
# Config / IO helpers
# ---------------------------------------------------------------------------

def read_case_config(config_file):
    """Load the JSON configuration file."""
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


# ---------------------------------------------------------------------------
# Data fetch / read
# ---------------------------------------------------------------------------

def fetch_totforce(case_id, totforce_flag, residual_flag):
    """Download total forces and/or nonlinear residuals for a case."""
    case = flow360.Case(case_id)
    case.results.download(
        total_forces=totforce_flag,
        nonlinear_residuals=residual_flag,
        surface_forces=False,
        bet_forces=False,
        cfl=False,
    )
    if totforce_flag:
        forcename = case_id + '_total_forces_v2.csv'
        os.rename('total_forces_v2.csv', forcename)
    if residual_flag:
        forcename = case_id + '_nonlinear_residual_v2.csv'
        os.rename('nonlinear_residual_v2.csv', forcename)
    return forcename


def get_convergence_data(filename):
    """Read a CSV file and return all columns as a dict of lists."""
    dataframe = pandas.read_csv(filename, skipinitialspace=True)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
    return dataframe.to_dict("list")


def get_data_at_last_pseudo_step(filename):
    """
    Extract force data at the last pseudo-step of each physical step.

    Returns a dict mapping column names to lists (one entry per physical step).
    """
    dataframe = pandas.read_csv(filename, skipinitialspace=True)
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
    data_raw = dataframe.to_dict("list")
    data = {key: [] for key in data_raw.keys()}

    n = len(data_raw['physical_step'])
    j = 0
    for i in range(0, n - 1):
        if data_raw['physical_step'][i] != data_raw['physical_step'][i + 1]:
            j += 1
            for key in data_raw.keys():
                data[key].append(data_raw[key][i])
    for key in data_raw.keys():
        data[key].append(data_raw[key][n - 1])
    print("total_physical_steps=", j)
    return data


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_line(i, j, coll, forces, forcename, axs, case, colori=0, residual_flag=False):
    """
    Plot a single convergence curve onto subplot axs[i, j].

    Cases 0-3 use solid lines; cases 4-7 use dashed lines with recycled colors.
    Residual quantities are plotted on a log scale.
    """
    color = cmap.colors[colori % 4] if colori < 4 else cmap.colors[colori - 4]
    linestyle = '-' if colori < 4 else '--'
    if residual_flag:
        axs[i, j].semilogy(coll, forces[forcename], linestyle, label=case, color=color, linewidth=2.5)
    else:
        axs[i, j].plot(coll, forces[forcename], linestyle, label=case, color=color, linewidth=2.5)
    axs[i, j].set_ylabel(forcename)


def plot_convergence_comparison(folder, cases, forces, AOA, step, forcestoplot,
                                totforce_flag, residual_flag, figure_extname, xlabel, title="Solver release test"):
    """
    Plot convergence history for all cases at a given AOA/condition.

    Y-axis limits are set with 10% padding (forces) or one decade (residuals).
    """
    jrange = [0, 1, 2]
    irange = [0, 1]
    ni = len(irange)
    nj = len(jrange)
    fig, axs = plt.subplots(ni, nj, figsize=(24, 14))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    print(f"Plotting convergence: {xlabel}={AOA}, step={step}")
    ncases = len(cases)
    maxphstep = max(forces[0]['physical_step'])
    lim = [[None, None] for _ in range(ni * nj)]

    for icase in range(ncases):
        coll = forces[icase]['pseudo_step']
        for i in irange:
            for j in jrange:
                index = i * nj + j
                forcename = forcestoplot[index]
                plot_line(i, j, coll, forces[icase], forcename, axs, cases[icase], icase, residual_flag)
                vmin = min(forces[icase][forcename])
                vmax = max(forces[icase][forcename])
                lim[index][0] = vmin if lim[index][0] is None else min(lim[index][0], vmin)
                lim[index][1] = vmax if lim[index][1] is None else max(lim[index][1], vmax)
                axs[i, j].set_xlabel('pseudo_step' if maxphstep == 0 else 'physical_step',
                                     fontsize=16, fontweight='bold')
                axs[i, j].tick_params(axis='both', labelsize=14)

    # Apply y-axis limits and grid after all cases are plotted to avoid toggle behaviour
    for i in irange:
        for j in jrange:
            index = i * nj + j
            vmin, vmax = lim[index]
            if totforce_flag:
                span = vmax - vmin
                axs[i, j].set_ylim(vmin - 0.1 * span, vmax + 0.1 * span)
            else:
                axs[i, j].set_ylim(vmin / 10, vmax * 10)
            axs[i, j].grid(True)

    plot_type = "Total Force" if totforce_flag else ("Residual" if residual_flag else "")
    axs[0, 1].set_title(f"{title}  {plot_type}  {xlabel}={AOA}", fontsize=20)
    axs[0, 0].legend(fontsize=16, loc='upper center')

    if totforce_flag:
        figurename = os.path.join(folder, "figures", figure_extname, "forcehistory",
                                  f"{figure_extname}force_history_{xlabel}{AOA}_last{step}step.png")
    else:
        figurename = os.path.join(folder, "figures", figure_extname, "residual",
                                  f"{figure_extname}residual_{xlabel}{AOA}_last{step}step.png")
    print("figurename:", figurename)
    plt.savefig(figurename, dpi=500, bbox_inches='tight')
    plt.close()


def plot_convergence_comparison_range(folder, cases, forces, AOA, step, forcestoplot,
                                      totforce_flag, residual_flag, figure_extname, xlabel, title="Solver release test"):
    """
    Plot convergence history with y-axis range determined from the last data point
    of each case, with wider padding (±2x span) to show convergence spread.
    """
    jrange = [0, 1, 2]
    irange = [0, 1]
    ni = len(irange)
    nj = len(jrange)
    fig, axs = plt.subplots(ni, nj, figsize=(24, 14))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    print(f"Plotting convergence range: {xlabel}={AOA}, step={step}")
    ncases = len(cases)
    maxphstep = max(forces[0]['physical_step'])
    lim = np.zeros((len(forcestoplot), 2))

    # Pre-pass: set limits from the last value of each case
    for i in irange:
        for j in jrange:
            index = i * nj + j
            forcename = forcestoplot[index]
            last_values = [forces[icase][forcename][-1] for icase in range(ncases)]
            vmin = min(last_values)
            vmax = max(last_values)
            if totforce_flag:
                span = vmax - vmin
                lim[index][0] = vmin - 2 * span
                lim[index][1] = vmax + 2 * span
            else:
                lim[index][0] = vmin / 10
                lim[index][1] = vmax * 10

    for icase in range(ncases):
        coll = forces[icase]['pseudo_step']
        for i in irange:
            for j in jrange:
                index = i * nj + j
                forcename = forcestoplot[index]
                plot_line(i, j, coll, forces[icase], forcename, axs, cases[icase], icase, residual_flag)
                axs[i, j].set_ylabel(forcename)
                axs[i, j].set_ylim(lim[index][0], lim[index][1])
                axs[i, j].set_xlabel('pseudo_step' if maxphstep == 0 else 'physical_step',
                                     fontsize=16, fontweight='bold')
                axs[i, j].tick_params(axis='both', labelsize=14)

    # Apply grid after all cases are plotted to avoid toggle behaviour
    for i in irange:
        for j in jrange:
            axs[i, j].grid(True)

    plot_type = "Total Force" if totforce_flag else ("Residual" if residual_flag else "")
    axs[0, 1].set_title(f"{title}  {plot_type}  {xlabel}={AOA}", fontsize=20)
    axs[0, 1].legend(fontsize=14, loc='upper center')

    if totforce_flag:
        figurename = os.path.join(folder, "figures", figure_extname, "forcehistory",
                                  f"{figure_extname}range_force_history_{xlabel}{AOA}_last{step}step.png")
    else:
        figurename = os.path.join(folder, "figures", figure_extname, "residual",
                                  f"{figure_extname}range_residual_{xlabel}{AOA}_last{step}step.png")
    print("figurename:", figurename)
    plt.savefig(figurename, dpi=500, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Unsteady data extraction
# ---------------------------------------------------------------------------

def extract_unsteady_convergence(ncases, forcesarray, steppert):
    """
    Extract the last fraction (steppert) of physical steps from each case for plotting.

    steppert=1.0 uses all steps; steppert=0.1 uses only the last 10%.
    The pseudo_step axis is remapped to physical_step + pseudo_step/max_pseudo_step
    for a continuous x-axis across physical steps.
    """
    forces = {}
    temp = {key: [] for key in forcesarray[0].keys()}

    # Find the shortest common physical step span across all cases
    minlength = max(forcesarray[0]['physical_step']) - min(forcesarray[0]['physical_step'])
    for i in range(ncases):
        maxphstep = max(forcesarray[i]['physical_step'])
        minphstep = min(forcesarray[i]['physical_step'])
        minlength = min(minlength, maxphstep - minphstep)
    complength = int(steppert * minlength)

    for i in range(ncases):
        maxphstep = max(forcesarray[i]['physical_step'])
        minphstep = min(forcesarray[i]['physical_step'])
        maxpsstep = max(forcesarray[i]['pseudo_step'])

        beginstep = minphstep if abs(steppert - 1) < 1e-4 else maxphstep - complength
        beginindex = forcesarray[i]['physical_step'].index(beginstep)
        endindex = forcesarray[i]['physical_step'].index(maxphstep) - 1

        for key in forcesarray[i].keys():
            temp[key] = forcesarray[i][key][beginindex:endindex]

        lentemp = len(temp['pseudo_step'])
        if abs(steppert - 1) > 1e-4:
            initial_phstep = temp['physical_step'][0]
            for itemp in range(lentemp):
                temp['physical_step'][itemp] -= initial_phstep

        # Remap pseudo_step to a continuous axis: physical_step + pseudo_step/max_pseudo_step
        for ii in range(lentemp):
            temp['pseudo_step'][ii] = temp['physical_step'][ii] + temp['pseudo_step'][ii] / maxpsstep

        forces[i] = temp
        temp = {key: [] for key in forcesarray[0].keys()}

    return forces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_file="./config_files/config.json"):
    """
    Main workflow: read config, load convergence data, and generate history plots.

    Runs twice — once for total forces and once for nonlinear residuals.
    For each run and each AOA:
      - Steady cases: single convergence plot + range-limited plot
      - Unsteady cases: plots at 1%, 10%, and 100% of the physical step span
    """
    config = read_case_config(config_file)
    rootfolder = config["rootfolder"]
    casenames = config["casenames"]
    releases = config["releases"]
    subcases = config["subcases"]
    AOAs = config["AOAs"]
    figure_extname = config["figure_extname"]
    testdata = config["testdata"]
    rotorflag = config["rotorflag"]
    xlabel = config["xlabel"]
    scales = config["scales"]
    SST_flags = [bool(f) for f in config.get("SSTFlag", [False] * len(casenames))]
    SST_flags += [False] * (len(casenames) - len(SST_flags))
    feature_test = config.get("feature_test", False)
    test_label = "Solver feature test" if feature_test else "Solver release test"

    # Note: datafileexist is deprecated — file presence is detected automatically.

    ncases = len(casenames)
    nAOAs = len(AOAs)

    # Create top-level output folders and figure subdirectories
    for folder in [rootfolder, os.path.join(rootfolder, "figures")]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Folder created: {folder}")
        else:
            print(f"Folder already exists: {folder}")
    for subdir in ["forcehistory", "residual"]:
        path = os.path.join(rootfolder, "figures", figure_extname, subdir)
        os.makedirs(path, exist_ok=True)

    # Read all case ID files upfront
    caseIDarray = {}
    cases = []
    for i in range(ncases):
        krylov_suffix  = "_krylov"  if "krylov"  in subcases[i].lower() else ""
        gravity_suffix = "_gravity" if "gravity" in subcases[i].lower() else ""
        cases.append(casenames[i] + '_' + releases[i] + krylov_suffix + gravity_suffix)
        path = os.path.join(rootfolder, "data", casenames[i] + '_' + releases[i])
        figurepath = os.path.join(path, "figures")
        for folder in [path, figurepath]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Folder created: {folder}")
            else:
                print(f"Folder already exists: {folder}")

        caseID_file = rootfolder + '/caseIDfiles/' + casenames[i] + '_' + releases[i] + '_' + subcases[i] + '.txt'
        print("caseID_file=", caseID_file)
        caseIDarray[i] = read_caseID(caseID_file)

    # Run for total forces, then for nonlinear residuals
    for totforce_flag, residual_flag in [(True, False), (False, True)]:
        print(f"\n{'#' * 60}")
        print(f"Running with totforce_flag={totforce_flag}, residual_flag={residual_flag}")

        # Select force coefficients or residual variables to plot
        if residual_flag:
            if SST_flags[0]:
                forcestoplot = ['0_cont', '1_momx', '2_momy', '4_energ', '5_k', '6_omega']
            else:
                forcestoplot = ['0_cont', '1_momx', '2_momy', '3_momz', '4_energ', '5_nuHat']
        else:
            if rotorflag:
                forcestoplot = ['CFx', 'CFy', 'CFz', 'CMx', 'CMy', 'CMz']  # rotor: body-axis forces
            else:
                forcestoplot = ['CL', 'CD', 'CFy', 'CMx', 'CMy', 'CMz']    # default: wing/component

        # Bootstrap: load first case to discover column keys
        bootstrap_id = caseIDarray[0][0]
        bootstrap_path = os.path.join(rootfolder, "data", cases[0])
        if totforce_flag:
            bootstrap_file = os.path.join(bootstrap_path, bootstrap_id + '_total_forces_v2.csv')
        else:
            bootstrap_file = os.path.join(bootstrap_path, bootstrap_id + '_nonlinear_residual_v2.csv')
        if not os.path.exists(bootstrap_file):
            bootstrap_file = fetch_totforce(bootstrap_id, totforce_flag, residual_flag)
        forces = get_convergence_data(bootstrap_file)

        for iaoa in range(nAOAs):
            forcearray = {}
            for key in forces.keys():
                forcearray[key] = []

            maxphstep = 0
            for icase in range(ncases):
                scale = scales[icase]
                path = os.path.join(rootfolder, "data", cases[icase])
                case_id = caseIDarray[icase][iaoa]
                print(f"  icase={icase}  iaoa={iaoa}  case_id={case_id}")

                if totforce_flag:
                    newname = os.path.join(path, case_id + '_total_forces_v2.csv')
                else:
                    newname = os.path.join(path, case_id + '_nonlinear_residual_v2.csv')

                if not os.path.exists(newname):
                    # Fetch from Flow360 and move into place
                    cvsforces_file = fetch_totforce(case_id, totforce_flag, residual_flag)
                    shutil.move(cvsforces_file, newname)
                else:
                    print(f"    Using existing local file: {newname}")

                forces = get_convergence_data(newname)
                if totforce_flag:
                    for key in forcestoplot:
                        forces[key] = [v / scale for v in forces[key]]

                forcearray[icase] = forces
                maxphstep = max(forces['physical_step'])

            AOA = AOAs[iaoa]
            if maxphstep == 0:
                # Steady case
                print(f"{'#' * 30}  steady case  {xlabel}={AOA}")
                plot_convergence_comparison(rootfolder, cases, forcearray, AOA, maxphstep,
                                            forcestoplot, totforce_flag, residual_flag, figure_extname, xlabel, title=test_label)
                plot_convergence_comparison_range(rootfolder, cases, forcearray, AOA, maxphstep,
                                                  forcestoplot, totforce_flag, residual_flag, figure_extname, xlabel, title=test_label)
            else:
                # Unsteady case: plot at 1%, 10%, and 100% of step span
                print(f"{'#' * 30}  unsteady case  {xlabel}={AOA}  iaoa={iaoa}")
                for plotscale in [0.01, 0.1, 1.0]:
                    unsteadyforces = extract_unsteady_convergence(ncases, forcearray, plotscale)
                    extractstep = maxphstep if abs(plotscale - 1) < 1e-3 else max(unsteadyforces[0]['physical_step'])
                    plot_convergence_comparison(rootfolder, cases, unsteadyforces, AOA, extractstep,
                                                forcestoplot, totforce_flag, residual_flag, figure_extname, xlabel, title=test_label)
                    plot_convergence_comparison_range(rootfolder, cases, unsteadyforces, AOA, extractstep,
                                                      forcestoplot, totforce_flag, residual_flag, figure_extname, xlabel, title=test_label)
                    if testdata and rotorflag:
                        testname = os.path.join(path, 'testdata',
                                                f"rotor_{xlabel}{AOA}_last{extractstep}step_scale{plotscale}.csv")
                        print("testname=", testname)
                        if os.path.exists(testname):
                            testforces = get_data_at_last_pseudo_step(testname)
                            plot_convergence_comparison(path, cases, [testforces], AOA, extractstep,
                                                        forcestoplot, totforce_flag, residual_flag, figure_extname, xlabel, title=test_label)


if __name__ == '__main__':
    main()
