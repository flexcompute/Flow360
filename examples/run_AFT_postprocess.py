"""
Wrapper script for AFT post-processing.

Runs the following scripts in sequence:
  1. post_AFT_total_force.py   — aggregate force coefficients and produce comparison plots
  2. post_AFT_forces_history_V5.py — convergence history and residual plots

Both scripts read from the same ./config_files/config.json configuration file.

Usage:
    python run_AFT_postprocess.py
"""

import sys
import traceback

import post_AFT_total_force as total_force
import post_AFT_forces_history_V4 as forces_history


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
    steps = [
        ("Total force coefficients (post_AFT_total_force)", total_force.main),
        ("Force/residual history   (post_AFT_forces_history_V5)", forces_history.main),
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
