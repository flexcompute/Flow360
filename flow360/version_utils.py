"""
Utilities for version handling.
"""

from __future__ import annotations

import os
from typing import Optional

from packaging.version import Version

from .log import log
from .version import __version__

_WARNED_PRERELEASE = False
_BOX_MAX_WIDTH = 110
_SUPPRESS_ENV_VAR = "FLOW360_SUPPRESS_BETA_WARNING"


def is_prerelease_version(version: Optional[str] = None) -> bool:
    """
    Determine whether the provided version string belongs to a prerelease build.

    "Prerelease" here means anything that is not an official stable release
    (the plain M.N.P versions published to PyPI). That covers PEP 440
    prereleases — public betas (25.11.2b2) and internal dev builds
    (25.11.2.dev20260819143000+gab12cd34) — and also shapes PEP 440 does not
    call prereleases but we never publish officially: builds carrying a local
    version segment (25.11.2+gab12cd34 — PyPI rejects local versions, so these
    can only come from internal tooling) and post releases.

    Args:
        version: Version string to check. Defaults to the client __version__.

    Returns:
        True if the version string represents a prerelease build, False otherwise.
    """
    parsed = Version(version or __version__)
    return parsed.is_prerelease or parsed.is_postrelease or parsed.local is not None


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
    if os.environ.get(_SUPPRESS_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}:
        _WARNED_PRERELEASE = True
        return

    warning_box = _build_warning_box(__version__)
    log.warning("\n%s", warning_box)
    _WARNED_PRERELEASE = True


if is_prerelease_version():
    warn_if_prerelease_version()
