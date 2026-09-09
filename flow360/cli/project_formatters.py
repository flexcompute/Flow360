"""Presentation-only formatters for project CLI payloads."""

from __future__ import annotations

from datetime import datetime


def _format_created_at(value):
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except ValueError:
        return str(value)


def _append_optional_line(lines, label, value):
    if value is not None and value != "" and value != []:
        lines.append(f" {label:<13} {value}")


def _append_count_line(lines, label, statistics):
    if not statistics:
        return
    count = statistics.get("count")
    if count is not None:
        lines.append(f" {label:<19} {count}")


def format_project_list(payload, *, project_url_factory=None) -> str:
    """Format serialized project-list payload as user-facing text."""

    records = payload.get("records") or []
    if not records:
        return "No matching projects found. Try skip naming patterns to show all."

    lines = [">>> Projects sorted by creation time:"]
    for item in records:
        project_id = item.get("id")
        created_at = _format_created_at(item.get("created_at"))
        statistics = item.get("statistics") or {}

        _append_optional_line(lines, "Name:", item.get("name"))
        _append_optional_line(lines, "Created at:", created_at)
        _append_optional_line(lines, "Created with:", item.get("root_item_type"))
        _append_optional_line(lines, "Solver:", item.get("solver_version"))
        _append_optional_line(lines, "ID:", project_id)
        if project_id and project_url_factory is not None:
            _append_optional_line(lines, "Link:", project_url_factory(project_id))
        _append_optional_line(lines, "Tags:", item.get("tags"))
        _append_optional_line(lines, "Description:", item.get("description"))

        _append_count_line(lines, "Geometry count:", statistics.get("geometry"))
        _append_count_line(lines, "Surface Mesh count:", statistics.get("surface_mesh"))
        _append_count_line(lines, "Volume Mesh count:", statistics.get("volume_mesh"))
        _append_count_line(lines, "Case count:", statistics.get("case"))
        lines.append("")

    returned = payload.get("returned")
    total = payload.get("total")
    if total is not None:
        if returned is not None and returned != total:
            lines.append(f"Showing {returned} of {total} matching projects.")
        else:
            lines.append(f"Total number of matching projects on the cloud: {total}")

    return "\n".join(lines)
