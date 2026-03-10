#!/usr/bin/env python3
"""Generate a coverage summary from Cobertura XML, with optional diff coverage."""

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def make_bar(pct, width=20):
    pct = max(0.0, min(100.0, pct))
    filled = round(pct / 100 * width)
    return "\u2593" * filled + "\u2591" * (width - filled)


def status_icon(pct):
    if pct >= 80:
        return "\U0001f7e2"
    if pct >= 60:
        return "\U0001f7e1"
    return "\U0001f534"


def _get_repo_root():
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


_repo_root = None


def _normalize_filename(filename, source_roots):
    """Normalize coverage XML filename to repo-relative path.

    Coverage XML records paths relative to <source> roots, while git diff
    produces paths relative to the repo root. This joins the two and strips
    the repo root prefix so both sides use the same reference frame.
    """
    global _repo_root
    if _repo_root is None:
        _repo_root = _get_repo_root()
    repo_prefix = _repo_root + "/"

    for root in source_roots:
        if os.path.isabs(filename):
            if filename.startswith(repo_prefix):
                return filename[len(repo_prefix):]
        else:
            full = os.path.join(root, filename)
            if full.startswith(repo_prefix):
                return full[len(repo_prefix):]
    return filename


def parse_coverage_xml(xml_path, depth):
    """Parse coverage.xml, return (groups_dict, file_line_coverage_dict)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    source_roots = [s.text.rstrip("/") for s in root.findall(".//source") if s.text]

    groups = defaultdict(lambda: {"hits": 0, "lines": 0})
    file_coverage = defaultdict(dict)

    for pkg in root.findall(".//package"):
        name = pkg.get("name", "")
        parts = name.split(".")
        key = ".".join(parts[:depth]) if len(parts) >= depth else name

        for cls in pkg.findall(".//class"):
            filename = cls.get("filename", "")
            filename = _normalize_filename(filename, source_roots)
            for line in cls.findall(".//line"):
                line_num = int(line.get("number", "0"))
                hit = int(line.get("hits", "0")) > 0
                file_coverage[filename][line_num] = (
                    file_coverage[filename].get(line_num, False) or hit
                )
                groups[key]["lines"] += 1
                if hit:
                    groups[key]["hits"] += 1

    return groups, file_coverage


def get_changed_lines(diff_branch):
    """Run git diff and return {filepath: set_of_changed_line_numbers} for non-test .py files."""
    result = subprocess.run(
        ["git", "diff", "--unified=0", diff_branch, "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    )

    changed = defaultdict(set)
    current_file = None
    hunk_re = re.compile(r"^@@ .+?\+(\d+)(?:,(\d+))? @@")

    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("@@") and current_file:
            m = hunk_re.match(line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) else 1
                if count > 0:
                    for i in range(start, start + count):
                        changed[current_file].add(i)

    return {
        f: lines
        for f, lines in changed.items()
        if f.endswith(".py") and not f.startswith("tests/") and f.startswith("flow360/")
    }


def build_diff_coverage_md(changed_lines, file_coverage):
    """Build diff coverage markdown section."""
    if not changed_lines:
        return "## Diff Coverage\n\nNo implementation files changed.\n"

    total_covered = 0
    total_changed = 0

    file_stats = []
    for filepath, line_nums in sorted(changed_lines.items()):
        cov_map = file_coverage.get(filepath, {})
        executable = {ln for ln in line_nums if ln in cov_map}
        covered = {ln for ln in executable if cov_map[ln]}
        missing = sorted(executable - covered)

        n_exec = len(executable)
        n_cov = len(covered)
        total_covered += n_cov
        total_changed += n_exec
        pct = (n_cov / n_exec * 100) if n_exec else -1
        file_stats.append((filepath, pct, n_cov, n_exec, missing))

    file_stats.sort(key=lambda x: x[1])
    total_pct = (total_covered / total_changed * 100) if total_changed else 100

    lines = []
    lines.append(f"## {status_icon(total_pct)} Diff Coverage — {total_pct:.0f}%")
    lines.append("")
    lines.append(
        f"`{make_bar(total_pct, 30)}` **{total_pct:.1f}%** ({total_covered} / {total_changed} changed lines covered)"
    )
    lines.append("")
    lines.append("| File | Coverage | Lines | Missing |")
    lines.append("|:-----|:--------:|:-----:|:--------|")

    for filepath, pct, n_cov, n_exec, missing in file_stats:
        icon = status_icon(pct) if pct >= 0 else "\u26aa"
        pct_str = f"{pct:.0f}%" if pct >= 0 else "N/A"
        missing_str = ", ".join(f"L{ln}" for ln in missing[:20])
        if len(missing) > 20:
            missing_str += f" \u2026 +{len(missing) - 20} more"
        lines.append(f"| `{filepath}` | {icon} {pct_str} | {n_cov} / {n_exec} | {missing_str} |")

    lines.append(f"| **Total** | **{total_pct:.1f}%** | **{total_covered} / {total_changed}** | |")
    lines.append("")
    return "\n".join(lines)


def build_full_coverage_md(groups):
    """Build full coverage markdown section (wrapped in <details>, collapsed by default)."""
    total_lines = sum(g["lines"] for g in groups.values())
    total_hits = sum(g["hits"] for g in groups.values())
    total_pct = (total_hits / total_lines * 100) if total_lines else 0

    sorted_groups = sorted(
        groups.items(),
        key=lambda x: (x[1]["hits"] / x[1]["lines"] * 100) if x[1]["lines"] else 0,
    )

    lines = []
    lines.append("<details>")
    lines.append(
        f"<summary><h3>{status_icon(total_pct)} Full Coverage Report — {total_pct:.0f}% ({total_hits} / {total_lines} lines)</h3></summary>"
    )
    lines.append("")
    lines.append(
        f"`{make_bar(total_pct, 30)}` **{total_pct:.1f}%** ({total_hits} / {total_lines} lines)"
    )
    lines.append("")
    lines.append("| Package | Coverage | Progress | Lines |")
    lines.append("|:--------|:--------:|:---------|------:|")

    for key, g in sorted_groups:
        pct = (g["hits"] / g["lines"] * 100) if g["lines"] else 0
        icon = status_icon(pct)
        lines.append(
            f"| `{key}` | {icon} {pct:.1f}% | `{make_bar(pct)}` | {g['hits']} / {g['lines']} |"
        )

    lines.append(f"| **Total** | **{total_pct:.1f}%** | | **{total_hits} / {total_lines}** |")
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


def main():
    xml_path = sys.argv[1] if len(sys.argv) > 1 else "coverage.xml"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "coverage-summary.md"
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    diff_branch = sys.argv[4] if len(sys.argv) > 4 else None

    groups, file_coverage = parse_coverage_xml(xml_path, depth)

    parts = []

    if diff_branch:
        changed_lines = get_changed_lines(diff_branch)
        parts.append(build_diff_coverage_md(changed_lines, file_coverage))

    parts.append(build_full_coverage_md(groups))

    with open(output_path, "w") as f:
        f.write("\n".join(parts))

    print(f"Coverage summary written to {output_path}")


if __name__ == "__main__":
    main()
