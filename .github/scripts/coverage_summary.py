#!/usr/bin/env python3
"""Generate a folder-level coverage summary from Cobertura XML."""

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def make_bar(pct, width=20):
    filled = round(pct / 100 * width)
    return "\u2593" * filled + "\u2591" * (width - filled)


def main():
    xml_path = sys.argv[1] if len(sys.argv) > 1 else "coverage.xml"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "coverage-summary.md"
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Aggregate by package at specified depth
    groups = defaultdict(lambda: {"hits": 0, "lines": 0})

    for pkg in root.findall(".//package"):
        name = pkg.get("name", "")
        parts = name.split(".")
        key = ".".join(parts[:depth]) if len(parts) >= depth else name

        for line in pkg.findall(".//line"):
            groups[key]["lines"] += 1
            if int(line.get("hits", "0")) > 0:
                groups[key]["hits"] += 1

    total_lines = sum(g["lines"] for g in groups.values())
    total_hits = sum(g["hits"] for g in groups.values())
    total_pct = (total_hits / total_lines * 100) if total_lines else 0

    # Sort by coverage ascending (worst first)
    sorted_groups = sorted(
        groups.items(),
        key=lambda x: (x[1]["hits"] / x[1]["lines"] * 100) if x[1]["lines"] else 0,
    )

    lines = []
    lines.append(f"## Coverage Report \u2014 {total_pct:.0f}%")
    lines.append("")
    lines.append(
        f"`{make_bar(total_pct, 30)}` **{total_pct:.1f}%** ({total_hits} / {total_lines} lines)"
    )
    lines.append("")
    lines.append("| Package | Coverage | Progress | Lines |")
    lines.append("|:--------|:--------:|:---------|------:|")

    for key, g in sorted_groups:
        pct = (g["hits"] / g["lines"] * 100) if g["lines"] else 0
        lines.append(f"| `{key}` | {pct:.1f}% | `{make_bar(pct)}` | {g['hits']} / {g['lines']} |")

    lines.append(f"| **Total** | **{total_pct:.1f}%** | | **{total_hits} / {total_lines}** |")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Coverage summary written to {output_path}")


if __name__ == "__main__":
    main()
