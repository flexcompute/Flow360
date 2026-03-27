"""
Full pipeline wrapper: sweep submission followed by post-processing.

Runs run_sweep_V3.py for each matching run_config, then runs
run_AFT_postprocess.py with --case <name> (which waits for all cases to
complete before updating the config and producing figures).

Usage:
    python run_full_pipeline.py <case_name> [<case_name2> ...] [--force]
    python run_full_pipeline.py all [--force]

Examples:
    python run_full_pipeline.py XV15_MRF
    python run_full_pipeline.py honda_subsonic
    python run_full_pipeline.py honda_subsonic_gravity
    python run_full_pipeline.py all
    python run_full_pipeline.py XV15_MRF --force

Arguments:
    case_name   : caseID (e.g. honda_subsonic).
                  Regular case: matches run_configs whose caseID == case_name exactly.
                  Feature test (3-part, e.g. honda_subsonic_gravity): also includes
                  the base case run_configs (honda_subsonic).
                  Use 'all' to run every JSON in run_config/ except config.json.
    --force     : force post-processing config update even if entry already present
"""

import glob
import json
import os
import subprocess
import sys

RUN_CONFIG_DIR  = "run_config"
POST_CONFIG_DIR = "config_files"
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))


def sync_aoas(run_config_file, post_config_dir=POST_CONFIG_DIR):
    """
    Ensure AOAs in the post-processing config matches sweepvalue in the run_config.
    If they differ, overwrites AOAs with sweepvalue and saves the post-config.
    """
    with open(run_config_file) as f:
        run_cfg = json.load(f)

    case_id    = run_cfg["caseID"]
    sweepvalue = run_cfg["sweepvalue"]
    post_config_file = os.path.join(post_config_dir, f"{case_id}.json")

    if not os.path.exists(post_config_file):
        print(f"  [sync_aoas] post-config not found, skipping: {post_config_file}")
        return

    with open(post_config_file) as f:
        post_cfg = json.load(f)

    if post_cfg.get("AOAs") == sweepvalue:
        print(f"  [sync_aoas] AOAs already match sweepvalue in {os.path.basename(post_config_file)}")
        return

    print(f"  [sync_aoas] AOAs mismatch — updating {os.path.basename(post_config_file)}")
    print(f"    was : {post_cfg.get('AOAs')}")
    print(f"    now : {sweepvalue}")
    post_cfg["AOAs"] = sweepvalue
    with open(post_config_file, "w") as f:
        json.dump(post_cfg, f, indent=4)
    print(f"  [sync_aoas] Saved: {post_config_file}")


def find_run_configs(case_name):
    """Return sorted list of run_config JSONs for case_name (or all).

    Regular case (e.g. honda_subsonic):
      matches only honda_subsonic_release-*.json (caseID == case_name exactly).

    Feature test (3-part name, e.g. honda_subsonic_gravity):
      matches honda_subsonic_gravity_release-*.json  (the feature case)
      AND honda_subsonic_release-*.json              (the base case for comparison).
    """
    if case_name.lower() == "all":
        paths = glob.glob(os.path.join(RUN_CONFIG_DIR, "*.json"))
        return sorted(p for p in paths if os.path.basename(p) != "config.json")

    parts = case_name.split("_")
    is_feature = len(parts) >= 3

    # Regular case: only release configs for this exact caseID
    paths = glob.glob(os.path.join(RUN_CONFIG_DIR, f"{case_name}_release*.json"))
    if is_feature:
        base_name = "_".join(parts[:2])
        paths += glob.glob(os.path.join(RUN_CONFIG_DIR, f"{base_name}_release*.json"))

    return sorted(set(p for p in paths if os.path.basename(p) != "config.json"))


def run_cmd(cmd, label):
    """Run a shell command, stream output, and raise on failure."""
    print()
    print("=" * 70)
    print(f"  START: {label}")
    print(f"  CMD:   {' '.join(cmd)}")
    print("=" * 70)
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"  FAILED: {label} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print("=" * 70)
    print(f"  DONE:  {label}")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    raw_args = sys.argv[1:]
    force = "--force" in raw_args
    # support both positional and --case flag
    filtered = [a for a in raw_args if a != "--force"]
    if len(filtered) == 2 and filtered[0] == "--case":
        case_args = [filtered[1]]
    else:
        case_args = filtered

    # Expand 'all' to every run_config except config.json
    if len(case_args) == 1 and case_args[0].lower() == "all":
        all_configs = sorted(
            p for p in glob.glob(os.path.join(RUN_CONFIG_DIR, "*.json"))
            if os.path.basename(p) != "config.json"
        )
        # Derive unique case names from filenames: XV15_MRF_release-25.9.json -> XV15_MRF
        case_names = []
        seen = set()
        for p in all_configs:
            with open(p) as f:
                cfg = json.load(f)
            name = cfg["caseID"]
            if name not in seen:
                case_names.append(name)
                seen.add(name)
    else:
        case_names = case_args

    for case_name in case_names:
        run_configs = find_run_configs(case_name)
        if not run_configs:
            print(f"WARNING: no run_config files found for '{case_name}' — skipping.")
            continue

        print(f"\n{'#' * 70}")
        print(f"  CASE: {case_name}  ({len(run_configs)} run_config(s))")
        print(f"{'#' * 70}")

        # 1. Submit sweep for each run_config
        for rc in run_configs:
            sync_aoas(rc)
            run_cmd(
                [sys.executable, "run_sweep_V3.py", rc],
                f"Sweep submission [{os.path.basename(rc)}]",
            )

        # 2. Post-process (waits for cases to finish, updates config, generates figures)
        post_cmd = [sys.executable, "run_AFT_postprocess.py", "--case", case_name]
        if force:
            post_cmd.append("--force")
        run_cmd(post_cmd, f"Post-processing [{case_name}]")

    print()
    print("=" * 70)
    print("  ALL PIPELINES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
