"""
Update a post-processing config JSON with a new sweep run.

Reads metadata from a run_config file (produced for run_sweep_V3.py), verifies the
corresponding caseIDfile exists, then prepends the new release entry to all per-case
arrays in the post-processing config. Creates the post-processing config if it does
not yet exist.

Usage:
    python update_postconfig.py <run_config> <post_config> [--force]

Example:
    python update_postconfig.py run_config/HLPW_ANSAC_release-25.9.4.json config_files/HLPW_ANSAC.json
    python update_postconfig.py run_config/HLPW_ANSAC_release-25.9.4.json config_files/HLPW_ANSAC.json --force

Arguments:
    run_config   : path to the run_config JSON used with run_sweep_V3.py
    post_config  : path to the post-processing config JSON to create or update
    --force      : overwrite existing entry if already present
"""

import argparse
import json
import os
import sys


# Default values for per-case scalar fields when creating a new entry
DEFAULTS = {
    "datafileexist": 1,
    "nperiod": 1,
    "scales": 1,
    "npseduos": 2,
    "SSTFlag": False,
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Written: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_config",  help="run_config JSON used with run_sweep_V3.py")
    parser.add_argument("post_config", help="post-processing config JSON to create or update")
    parser.add_argument("--force", action="store_true", help="overwrite existing entry if already present")
    args = parser.parse_args()

    run_config_file  = args.run_config
    post_config_file = args.post_config
    force            = args.force

    # --- read run config ---
    run_cfg = load_json(run_config_file)
    caseID  = run_cfg["caseID"]
    version = run_cfg["version"]
    subcase = run_cfg["casepost"]
    AOAs    = run_cfg["sweepvalue"]
    root    = run_cfg["root"]

    # --- verify caseIDfile exists ---
    caseIDfile = os.path.join(root, "caseIDfiles", f"{caseID}_{version}_{subcase}.txt")
    if not os.path.exists(caseIDfile):
        print(f"ERROR: caseIDfile not found: {caseIDfile}")
        sys.exit(1)
    print(f"Found caseIDfile: {caseIDfile}")

    # --- load or initialise post-processing config ---
    if os.path.exists(post_config_file):
        cfg = load_json(post_config_file)
        print(f"Updating existing config: {post_config_file}")
    else:
        cfg = {
            "rootfolder": root,
            "casenames": [],
            "releases": [],
            "subcases": [],
            "datafileexist": [],
            "AOAs": AOAs,
            "figure_extname": caseID,
            "testdata": True,
            "rotorflag": False,
            "wholeplane": False,
            "xlabel": "AOA (deg)",
            "nperiod": [],
            "scales": [],
            "npseduos": [],
            "SSTFlag": [],
        }
        print(f"Creating new config: {post_config_file}")

    # update AOAs if the new run has a different sweep
    cfg["AOAs"] = AOAs

    # --- check for duplicate ---
    for i, (cn, rel, sub) in enumerate(zip(cfg["casenames"], cfg["releases"], cfg["subcases"])):
        if cn == caseID and rel == version and sub == subcase:
            if not force:
                print(f"Already present at index {i}: {caseID} / {version} / {subcase} — skipping update.")
                sys.exit(0)
            # force: remove existing entry before re-inserting at index 0
            print(f"Force update: removing existing entry at index {i}: {caseID} / {version} / {subcase}")
            cfg["casenames"].pop(i)
            cfg["releases"].pop(i)
            cfg["subcases"].pop(i)
            for key in DEFAULTS:
                cfg[key].pop(i)
            break

    # --- prepend new entry to all per-case arrays ---
    cfg["casenames"].insert(0, caseID)
    cfg["releases"].insert(0, version)
    cfg["subcases"].insert(0, subcase)
    for key, default in DEFAULTS.items():
        cfg[key].insert(0, default)

    save_json(post_config_file, cfg)
    print(f"Added: {caseID} / {version} / {subcase}  ({len(AOAs)} AOA points)")
    print(f"Total cases in config: {len(cfg['casenames'])}")


if __name__ == "__main__":
    main()
