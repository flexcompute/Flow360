"""
Wrapper script for AFT post-processing.

Optionally updates the post-processing config from a sweep run_config first,
then runs the two post-processing scripts in sequence:
  1. update_postconfig.py      — add new sweep run to the post-processing config (optional)
  2. post_AFT_total_force.py   — aggregate force coefficients and produce comparison plots
  3. post_AFT_forces_history_V4.py — convergence history and residual plots

When --run-config and --post-config are provided, the script first waits for all
cases in matching caseIDfiles (release_test/caseIDfiles/HLPW_ANSAC*) to reach a
final status before updating the config. It rechecks every 30 minutes.

Usage:
    # with config update step:
    python run_AFT_postprocess.py --run-config run_config/HLPW_ANSAC_release-25.9.json --post-config config_files/HLPW_ANSAC.json

    # post-processing only (config already up to date):
    python run_AFT_postprocess.py
"""

import argparse
import glob
import sys
import time
import traceback

import flow360 as fl

import update_postconfig
import post_AFT_total_force as total_force
import post_AFT_forces_history_V4 as forces_history

WAIT_INTERVAL_SECONDS = 30 * 60  # 30 minutes


def _read_case_ids(caseIDfile):
    """Return list of case IDs from a caseIDfile (one ID per line)."""
    with open(caseIDfile, "r") as f:
        return [line.strip() for line in f if line.strip()]


def wait_for_cases(run_config_file):
    """
    Poll all cases in caseIDfiles matching release_test/caseIDfiles/HLPW_ANSAC*
    until every case reaches a final status (completed, error, diverged, etc.).
    Rechecks every 30 minutes if any case is still running.
    """
    import json
    with open(run_config_file, "r") as f:
        run_cfg = json.load(f)
    root = run_cfg["root"]

    pattern = f"{root}/caseIDfiles/HLPW_ANSAC*"
    caseIDfiles = glob.glob(pattern)
    if not caseIDfiles:
        print(f"No caseIDfiles found matching: {pattern}")
        return

    # Collect all unique case IDs across matching files
    all_case_ids = []
    for path in sorted(caseIDfiles):
        ids = _read_case_ids(path)
        print(f"  {path}: {len(ids)} case(s)")
        all_case_ids.extend(ids)
    all_case_ids = list(dict.fromkeys(all_case_ids))  # deduplicate, preserve order
    print(f"Total unique cases to check: {len(all_case_ids)}")

    while True:
        pending = []
        for case_id in all_case_ids:
            case = fl.Case(case_id)
            status = case.status
            if not status.is_final():
                pending.append((case_id, str(status)))

        if not pending:
            print("All cases have reached a final status. Proceeding.")
            return

        print(f"{len(pending)} case(s) still running:")
        for case_id, status in pending:
            print(f"  {case_id}  [{status}]")
        print(f"Waiting 30 minutes before rechecking...")
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
    except Exception:
        print("=" * 70)
        print(f"  FAILED: {label}")
        traceback.print_exc()
        print("=" * 70)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-config",  default=None, help="run_config JSON used with run_sweep_V3.py")
    parser.add_argument("--post-config", default=None, help="post-processing config JSON to create or update")
    args = parser.parse_args()

    steps = []

    if args.run_config and args.post_config:
        def _wait():
            wait_for_cases(args.run_config)
        steps.append(("Wait for all HLPW_ANSAC cases to complete", _wait))

        def _update_config():
            sys.argv = ["update_postconfig.py", args.run_config, args.post_config]
            update_postconfig.main()
        steps.append(("Update post-processing config (update_postconfig)", _update_config))

    steps += [
        ("Total force coefficients (post_AFT_total_force)", total_force.main),
        ("Force/residual history   (post_AFT_forces_history_V4)", forces_history.main),
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
