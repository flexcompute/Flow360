"""
version
"""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import Version

DIST_NAME = "flow360"


def _read_local_pyproject_version() -> str | None:
    """Read version from the source-tree pyproject, gated on name = "flow360"."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        return None
    content = pyproject.read_text(encoding="utf-8")
    name_match = re.search(r'^name\s*=\s*"([^"]+)"\s*$', content, flags=re.MULTILINE)
    if name_match is None or name_match.group(1) != DIST_NAME:
        return None
    version_match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', content, flags=re.MULTILINE)
    return version_match.group(1) if version_match else None


def _resolve_version() -> str:
    try:
        return version(DIST_NAME)
    except PackageNotFoundError:
        local_version = _read_local_pyproject_version()
        if local_version is not None:
            return local_version
        raise RuntimeError(
            "Unable to determine flow360 version from metadata or pyproject.toml"
        ) from None


def _derive_solver_version(client_version: str) -> str:
    """The default solver line is always the client's first two digits.

    Contract: client A.B.x targets solver release-A.B — for betas
    (A.B.0bN) and internal dev builds (A.B.x.devTS+gSHA) alike. Derived,
    not hand-maintained: the previous hard-coded literal silently lagged
    behind __version__ for multiple release lines.
    """
    parsed = Version(client_version)
    return f"release-{parsed.major}.{parsed.minor}"


__version__ = _resolve_version()
__solver_version__ = _derive_solver_version(__version__)
