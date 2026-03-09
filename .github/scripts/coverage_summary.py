#!/usr/bin/env python3
"""Generate an HTML coverage summary from Cobertura XML, with optional diff coverage."""

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def pct_color(pct):
    if pct >= 80:
        return "#4caf50"
    if pct >= 60:
        return "#ff9800"
    return "#f44336"


def html_bar(pct, width=200):
    pct = max(0.0, min(100.0, pct))
    color = pct_color(pct)
    return (
        f'<div style="background:#e0e0e0;border-radius:4px;width:{width}px;height:14px;display:inline-block">'
        f'<div style="background:{color};border-radius:4px;width:{pct:.1f}%;height:14px"></div>'
        f"</div>"
    )


def _normalize_filename(filename, source_roots):
    """Normalize coverage XML filename to repo-relative path."""
    if not os.path.isabs(filename):
        return filename
    for root in source_roots:
        if filename.startswith(root + "/"):
            return filename[len(root) + 1 :]
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


def build_diff_coverage_html(changed_lines, file_coverage):
    """Build diff coverage HTML section."""
    if not changed_lines:
        return "<h3>Diff Coverage</h3>\n<p>No implementation files changed.</p>\n"

    rows = []
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

    for filepath, pct, n_cov, n_exec, missing in file_stats:
        color = pct_color(pct) if pct >= 0 else "#888"
        pct_str = f"{pct:.0f}%" if pct >= 0 else "N/A"
        missing_str = ", ".join(f"L{ln}" for ln in missing[:20])
        if len(missing) > 20:
            missing_str += f" … +{len(missing) - 20} more"
        rows.append(
            f"<tr>"
            f"<td><code>{filepath}</code></td>"
            f'<td style="color:{color};text-align:center"><b>{pct_str}</b></td>'
            f'<td style="text-align:center">{n_cov} / {n_exec}</td>'
            f"<td>{missing_str}</td>"
            f"</tr>"
        )

    total_color = pct_color(total_pct)

    html = f"<h3>Diff Coverage — {total_pct:.0f}%</h3>\n"
    html += f"{html_bar(total_pct, 300)} <b>{total_covered} / {total_changed}</b> changed lines covered\n"
    html += "<table>\n"
    html += "<tr><th>File</th><th>Coverage</th><th>Lines</th><th>Missing</th></tr>\n"
    html += "\n".join(rows) + "\n"
    html += (
        f'<tr style="font-weight:bold">'
        f"<td>Total</td>"
        f'<td style="color:{total_color};text-align:center">{total_pct:.1f}%</td>'
        f'<td style="text-align:center">{total_covered} / {total_changed}</td>'
        f"<td></td></tr>\n"
    )
    html += "</table>\n"
    return html


def build_full_coverage_html(groups):
    """Build full coverage HTML section (wrapped in <details>, collapsed by default)."""
    total_lines = sum(g["lines"] for g in groups.values())
    total_hits = sum(g["hits"] for g in groups.values())
    total_pct = (total_hits / total_lines * 100) if total_lines else 0

    sorted_groups = sorted(
        groups.items(),
        key=lambda x: (x[1]["hits"] / x[1]["lines"] * 100) if x[1]["lines"] else 0,
    )

    rows = []
    for key, g in sorted_groups:
        pct = (g["hits"] / g["lines"] * 100) if g["lines"] else 0
        color = pct_color(pct)
        rows.append(
            f"<tr>"
            f"<td><code>{key}</code></td>"
            f'<td style="color:{color};text-align:center"><b>{pct:.1f}%</b></td>'
            f"<td>{html_bar(pct, 150)}</td>"
            f'<td style="text-align:right">{g["hits"]} / {g["lines"]}</td>'
            f"</tr>"
        )

    total_color = pct_color(total_pct)

    html = "<details>\n"
    html += f"<summary><h3>Full Coverage Report — {total_pct:.0f}% ({total_hits} / {total_lines} lines)</h3></summary>\n\n"
    html += "<table>\n"
    html += "<tr><th>Package</th><th>Coverage</th><th>Progress</th><th>Lines</th></tr>\n"
    html += "\n".join(rows) + "\n"
    html += (
        f'<tr style="font-weight:bold">'
        f"<td>Total</td>"
        f'<td style="color:{total_color};text-align:center">{total_pct:.1f}%</td>'
        f"<td></td>"
        f'<td style="text-align:right">{total_hits} / {total_lines}</td>'
        f"</tr>\n"
    )
    html += "</table>\n"
    html += "</details>\n"
    return html


def main():
    xml_path = sys.argv[1] if len(sys.argv) > 1 else "coverage.xml"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "coverage-summary.md"
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    diff_branch = sys.argv[4] if len(sys.argv) > 4 else None

    groups, file_coverage = parse_coverage_xml(xml_path, depth)

    parts = []

    if diff_branch:
        changed_lines = get_changed_lines(diff_branch)
        parts.append(build_diff_coverage_html(changed_lines, file_coverage))

    parts.append(build_full_coverage_html(groups))

    with open(output_path, "w") as f:
        f.write("\n".join(parts))

    print(f"Coverage summary written to {output_path}")


if __name__ == "__main__":
    main()
