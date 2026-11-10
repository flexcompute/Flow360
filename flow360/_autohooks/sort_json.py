"""Autohooks plugin: sort keys in staged JSON files under tests/."""

import importlib.util
from pathlib import Path

from autohooks.api import ok
from autohooks.api.git import (
    get_staged_status,
    stage_files_from_status_list,
    stash_unstaged_changes,
)
from autohooks.api.path import match

_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "tools"
_spec = importlib.util.spec_from_file_location("sort_ref_json", _TOOLS_DIR / "sort_ref_json.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
process_file = _module.process_file

DEFAULT_INCLUDE = ("*.json",)
TESTS_DIR = Path(__file__).resolve().parent.parent.parent / "tests"


def precommit(config=None, report_progress=None, **kwargs):  # pylint: disable=unused-argument
    """Sort JSON keys in staged test reference files."""
    files = [
        f
        for f in get_staged_status()
        if match(f.path, DEFAULT_INCLUDE) and TESTS_DIR in f.absolute_path().parents
    ]

    if not files:
        ok("No staged JSON files under tests/.")
        return 0

    if report_progress:
        report_progress.init(len(files))

    with stash_unstaged_changes(files):
        for f in files:
            already_sorted = process_file(f.absolute_path(), check=False)
            if already_sorted:
                ok(f"Already sorted: {f.path}")
            else:
                ok(f"Sorted keys in: {f.path}")

            if report_progress:
                report_progress.update()

        stage_files_from_status_list(files)

    return 0
