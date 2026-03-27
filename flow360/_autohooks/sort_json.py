"""Autohooks plugin: sort keys in staged JSON files under tests/."""

import json
from pathlib import Path

from autohooks.api import ok
from autohooks.api.git import (
    get_staged_status,
    stage_files_from_status_list,
    stash_unstaged_changes,
)
from autohooks.api.path import match

DEFAULT_INCLUDE = ("*.json",)
TESTS_DIR = Path(__file__).resolve().parent.parent.parent / "tests"


def _sort_keys(obj):
    """Recursively sort dictionary keys. List order is preserved."""
    if isinstance(obj, dict):
        return {k: _sort_keys(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_sort_keys(item) for item in obj]
    return obj


def _sort_file(path: Path) -> bool:
    """Sort keys in-place. Returns True if file was already sorted."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    sorted_content = json.dumps(_sort_keys(data), indent=4) + "\n"

    if raw == sorted_content:
        return True

    path.write_text(sorted_content, encoding="utf-8")
    return False


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
            try:
                already_sorted = _sort_file(f.absolute_path())
                if already_sorted:
                    ok(f"Already sorted: {f.path}")
                else:
                    ok(f"Sorted keys in: {f.path}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                ok(f"Skipped (not valid JSON): {f.path}")

            if report_progress:
                report_progress.update()

        stage_files_from_status_list(files)

    return 0
