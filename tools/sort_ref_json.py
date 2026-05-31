"""Sort keys in reference JSON files under tests/ for clean diffs.

Usage:
    python tools/sort_ref_json.py          # Fix: sort all JSON files in-place
    python tools/sort_ref_json.py --check  # Check only (for CI), exit 1 if unsorted
"""

import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"


def sort_keys(obj):
    """Recursively sort dictionary keys. List order is preserved."""
    if isinstance(obj, dict):
        return {k: sort_keys(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [sort_keys(item) for item in obj]
    return obj


def process_file(path: Path, *, check: bool) -> bool:
    """Returns True if file is already sorted."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    sorted_content = json.dumps(sort_keys(data), indent=4) + "\n"

    if raw == sorted_content:
        return True

    if check:
        return False

    path.write_text(sorted_content, encoding="utf-8")
    return False


def main():
    check = "--check" in sys.argv
    json_files = sorted(TESTS_DIR.rglob("*.json"))
    unsorted = []

    for path in json_files:
        try:
            already_sorted = process_file(path, check=check)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not already_sorted:
            unsorted.append(path)

    if not unsorted:
        print(f"All {len(json_files)} JSON files have sorted keys.")
        return

    if check:
        print(f"{len(unsorted)} file(s) have unsorted keys:")
        for path in unsorted:
            print(f"  {path.relative_to(TESTS_DIR.parent)}")
        print("\nTo fix, run:\n  python tools/sort_ref_json.py")
        sys.exit(1)
    else:
        print(f"Sorted keys in {len(unsorted)} file(s):")
        for path in unsorted:
            print(f"  {path.relative_to(TESTS_DIR.parent)}")


if __name__ == "__main__":
    main()
