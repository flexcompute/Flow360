"""
Utilities for version handling.
"""

from __future__ import annotations

import textwrap
from typing import List, Optional

from .log import log
from .version import __version__

_WARNED_PRERELEASE = False
_BOX_MAX_WIDTH = 110


def is_prerelease_version(version: Optional[str] = None) -> bool:
    """
    Determine whether the provided version string belongs to a prerelease build.

    Args:
        version: Version string to check. Defaults to the client __version__.

    Returns:
        True if the version string represents a prerelease build (beta), False otherwise.
    """
    version_to_check = version or __version__
    return "b" in version_to_check


def _build_warning_box(version: str) -> str:
    """
    Construct an ASCII warning box that scales with the message length.
    """
    title = "Flow360 Beta Version Warning"
    raw_lines = [
        f"You are using an unstable *beta* build of the Flow360 Python client ({version}).",
        "This build is not intended for public Flow360 releases or production workflows.",
        "Interfaces may change between beta drops, which can render scripts unusable.",
        "Backward compatibility for your data with upcoming releases is not guaranteed.",
        "Use at your own risk.",
        "Run `flow360 version` to find the latest stable release.",
    ]

    content_width = max(len(title), *(len(line) for line in raw_lines))
    border = "+" + "-" * (content_width + 2) + "+"

    formatted_lines = [
        border,
        f"| {title.center(content_width)} |",
        border,
    ]
    for line in raw_lines:
        formatted_lines.append(f"| {line.ljust(content_width)} |")
    formatted_lines.append(border)

    return "\n".join(formatted_lines)


def warn_if_prerelease_version() -> None:
    """
    Emit a warning once per process if the client is a prerelease build.
    """
    global _WARNED_PRERELEASE  # pylint: disable=global-statement
    if _WARNED_PRERELEASE:
        return
    if not is_prerelease_version():
        return

    warning_box = _build_warning_box(__version__)
    log.warning("\n%s", warning_box)
    _WARNED_PRERELEASE = True


if is_prerelease_version():
    warn_if_prerelease_version()
