"""
Wrapper script for AFT post-processing.

Optionally updates the post-processing config from a sweep run_config first,
then runs the two post-processing scripts in sequence:
  1. update_postconfig.py          — add new sweep run to the post-processing config (optional)
  2. post_AFT_total_force.py       — aggregate force coefficients and produce comparison plots
  3. post_AFT_forces_history_V4.py — convergence history and residual plots

When --case is provided, the script finds the matching run_config JSON (e.g.
run_config/HLPW_ANSAC_release-25.9.json for --case HLPW_ANSAC), waits for all
cases in the corresponding caseIDfiles to finish, updates the post-processing
config, then runs post-processing. Use --case all (or ALL) to process every
run_config JSON (excluding config.json).

Usage:
    # single case:
    python run_AFT_postprocess.py --case HLPW_ANSAC

    # all cases:
    python run_AFT_postprocess.py --case all

    # post-processing only (no config update):
    python run_AFT_postprocess.py
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback

import flow360 as fl

import update_postconfig
import post_AFT_total_force as total_force
import post_AFT_forces_history_V4 as forces_history

WAIT_INTERVAL_SECONDS = 30 * 60  # 30 minutes
RUN_CONFIG_DIR = "run_config"
POST_CONFIG_DIR = "config_files"


def _find_run_configs(case_name):
    """
    Return list of run_config JSON paths for the given case name.
    Excludes config.json (which is a symlink / generic default).
    For 'all'/'ALL', returns every JSON in RUN_CONFIG_DIR except config.json.
    Otherwise, globs for <case_name>_*.json.
    """
    if case_name.lower() == "all":
        paths = glob.glob(os.path.join(RUN_CONFIG_DIR, "*.json"))
    else:
        paths = glob.glob(os.path.join(RUN_CONFIG_DIR, f"{case_name}_*.json"))

    return sorted(p for p in paths if os.path.basename(p) != "config.json")


def _post_config_for(run_config_file):
    """Derive the post-processing config path from a run_config filename."""
    with open(run_config_file, "r") as f:
        cfg = json.load(f)
    case_name = cfg["caseID"]
    return os.path.join(POST_CONFIG_DIR, f"{case_name}.json")


def _read_case_ids(caseIDfile):
    """Return list of case IDs from a caseIDfile (one ID per line)."""
    with open(caseIDfile, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _detect_sst(parent_case_id):
    """
    Return True if the parent case uses a k-omega SST turbulence model, False otherwise.

    Checks the class name of param.turbulence_model_solver for 'SST' (case-insensitive).
    Falls back to False if the attribute is missing or the API call fails.
    """
    try:
        parent_case = fl.Case.from_cloud(parent_case_id)
        solver = parent_case.params.turbulence_model_solver
        return "sst" in type(solver).__name__.lower()
    except Exception as e:
        print(f"  [_detect_sst] Could not determine turbulence model: {e}")
        return False


def _init_post_config(run_config_file, post_config_file):
    """
    Create a skeleton post-processing config from a run_config if it does not exist.

    Derives settings from the run_config:
      - subcase   : set to casepost
      - nperiod   : 1 if casepost contains 'steady', else 2 (unsteady default)
      - SSTFlag   : True if parent case uses k-omega SST, False otherwise
      - AOAs      : taken from sweepvalue

    Does nothing if the post-config already exists.
    """
    if os.path.exists(post_config_file):
        return

    with open(run_config_file, "r") as f:
        run_cfg = json.load(f)

    case_id        = run_cfg["caseID"]
    root           = run_cfg["root"]
    sweepvalue     = run_cfg["sweepvalue"]
    casepost       = run_cfg["casepost"]
    parent_case_id = run_cfg["parent_case_id"]

    # nperiod: steady=1, unsteady=2
    nperiod_default = 1 if "steady" in casepost.lower() else 2

    # SSTFlag: detect from parent case turbulence model
    sst_flag = _detect_sst(parent_case_id)
    print(f"  [_init_post_config] casepost='{casepost}'  nperiod={nperiod_default}  SSTFlag={sst_flag}")

    version = run_cfg["version"]

    # Detect feature test: caseID with 3+ underscore-separated words (e.g. honda_subsonic_gravity).
    # In that case, compare feature case vs base case (first two parts) at the same version.
    parts = case_id.split("_")
    feature_test = len(parts) >= 3
    if feature_test:
        base_case_id = "_".join(parts[:2])
        print(f"  [_init_post_config] Feature test detected: '{case_id}' vs base '{base_case_id}'")
        casenames_list    = [case_id,          base_case_id]
        releases_list     = [version,          version]
        subcases_list     = [casepost,         casepost]
        datafileexist_list = [1,               1]
        nperiod_list      = [nperiod_default,  nperiod_default]
        scales_list       = [1,                1]
        npseduos_list     = [2,                2]
        sst_list          = [sst_flag,         sst_flag]
    else:
        casenames_list    = [case_id]
        releases_list     = [version]
        subcases_list     = [casepost]
        datafileexist_list = [1]
        nperiod_list      = [nperiod_default]
        scales_list       = [1]
        npseduos_list     = [2]
        sst_list          = [sst_flag]

    skeleton = {
        "rootfolder":     root,
        "casenames":      casenames_list,
        "releases":       releases_list,
        "subcases":       subcases_list,
        "datafileexist":  datafileexist_list,
        "AOAs":           sweepvalue,
        "figure_extname": case_id,
        "testdata":       True,
        "rotorflag":      False,
        "wholeplane":     False,
        "xlabel":         "AOA (deg)",
        "nperiod":        nperiod_list,
        "scales":         scales_list,
        "npseduos":       npseduos_list,
        "SSTFlag":        sst_list,
        "feature_test":   feature_test,
    }

    os.makedirs(os.path.dirname(post_config_file), exist_ok=True)
    with open(post_config_file, "w") as f:
        json.dump(skeleton, f, indent=4)
    print(f"Created new post-processing config: {post_config_file}")


def wait_for_cases(run_config_file):
    """
    Poll all cases in caseIDfiles matching <root>/caseIDfiles/<caseID>* until
    every case reaches a final status. Rechecks every 30 minutes if any are still running.
    """
    with open(run_config_file, "r") as f:
        run_cfg = json.load(f)
    root    = run_cfg["root"]
    case_id = run_cfg["caseID"]

    pattern = os.path.join(root, "caseIDfiles", f"{case_id}*")
    caseIDfiles = glob.glob(pattern)
    if not caseIDfiles:
        print(f"No caseIDfiles found matching: {pattern}")
        return

    all_case_ids = []
    for path in sorted(caseIDfiles):
        ids = _read_case_ids(path)
        print(f"  {path}: {len(ids)} case(s)")
        all_case_ids.extend(ids)
    all_case_ids = list(dict.fromkeys(all_case_ids))  # deduplicate, preserve order
    print(f"Total unique cases to check: {len(all_case_ids)}")

    while True:
        pending = []
        for cid in all_case_ids:
            case = fl.Case(cid)
            status = case.status
            if not status.is_final():
                pending.append((cid, str(status)))

        if not pending:
            print("All cases have reached a final status. Proceeding.")
            return

        print(f"{len(pending)} case(s) still running:")
        for cid, status in pending:
            print(f"  {cid}  [{status}]")
        print("Waiting 30 minutes before rechecking...")
        time.sleep(WAIT_INTERVAL_SECONDS)


def run(label, func):
    """Run a post-processing step and report success or failure."""
    print()
    print("=" * 70)
    print(f"  START: {label}")
    print("=" * 70)
    try:
        func()
        print("=" * 70)
        print(f"  DONE:  {label}")
        print("=" * 70)
        return True
    except SystemExit as e:
        if e.code == 0:
            print("=" * 70)
            print(f"  DONE:  {label} (skipped — already up to date)")
            print("=" * 70)
            return True
        print("=" * 70)
        print(f"  FAILED: {label} (exit code {e.code})")
        print("=" * 70)
        return False
    except Exception:
        print("=" * 70)
        print(f"  FAILED: {label}")
        traceback.print_exc()
        print("=" * 70)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default=None,
                        help="case name (e.g. HLPW_ANSAC) or 'all' to process every run_config")
    parser.add_argument("--force", action="store_true",
                        help="force config update even if the entry is already present")
    args = parser.parse_args()

    steps = []
    post_cfg_path = "./config_files/config.json"  # default if --case not provided

    if args.case:
        run_configs = _find_run_configs(args.case)
        if not run_configs:
            print(f"ERROR: no run_config files found for case '{args.case}' in {RUN_CONFIG_DIR}/")
            sys.exit(1)
        print(f"Found {len(run_configs)} run_config(s): {[os.path.basename(p) for p in run_configs]}")

        for rc in run_configs:
            post_cfg = _post_config_for(rc)
            rc_label = os.path.basename(rc)

            _init_post_config(rc, post_cfg)

            def _wait(rc=rc):
                wait_for_cases(rc)

            def _update(rc=rc, post_cfg=post_cfg):
                sys.argv = ["update_postconfig.py", rc, post_cfg] + (["--force"] if args.force else [])
                update_postconfig.main()

            steps.append((f"Wait for cases  [{rc_label}]", _wait))
            steps.append((f"Update config   [{rc_label}]", _update))

        # use the post-config for the last (or only) run_config as the post-processing input
        post_cfg_path = _post_config_for(run_configs[-1])

        # update config_files/config.json symlink to point to the case-specific config
        symlink_path = os.path.join(POST_CONFIG_DIR, "config.json")
        target = os.path.basename(post_cfg_path)
        if os.path.islink(symlink_path) or os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(target, symlink_path)
        print(f"Updated symlink: {symlink_path} -> {target}")

    steps += [
        ("Total force coefficients (post_AFT_total_force)", lambda cfg=post_cfg_path: total_force.main(cfg)),
        ("Force/residual history   (post_AFT_forces_history_V4)", lambda cfg=post_cfg_path: forces_history.main(cfg)),
    ]

    results = {}
    for label, func in steps:
        results[label] = run(label, func)

    # Summary
    print()
    print("=" * 70)
    print("  POST-PROCESSING SUMMARY")
    print("=" * 70)
    all_ok = True
    for label, ok in results.items():
        status = "OK     " if ok else "FAILED "
        print(f"  [{status}]  {label}")
        if not ok:
            all_ok = False
    print("=" * 70)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
