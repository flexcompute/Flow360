"""
Version control module
"""

__version__ = "22.2.3.0"

import re
from typing import Optional

from flow360.exceptions import Flow360RuntimeError
from flow360.log import log


class Flow360Version:
    """
    Flow360 version
    """

    def __init__(self, version):
        self.full = version
        ret = re.findall(r"^([a-zA-Z0-9\-]+)-([0-9\.]+)$", version)
        if len(ret) != 1:
            raise Flow360RuntimeError(f"solver version is not valid: {version}")
        self.head = ret[0][0]
        self.tail = [int(i) for i in ret[0][1].strip().split(".")]
        if self.head == "master":
            # Builds off master carry their own small numbering (`master-1.1.1.1`) while
            # releases are numbered by year (`release-25.11`). Scaling lifts a master build
            # above every release so it compares as the newest version, which is what makes
            # `examples/base_test_case.py` hand a master build the newest example assets.
            self.tail = [i * 100 for i in self.tail]

    @property
    def is_minor_release(self):
        """
        Whether the version pins a minor release inside a major release, like `release-25.8.7`
        does inside `release-25.8`. Versions predating the `release-<year>.<month>` scheme,
        such as `release-22.1.3.0`, have no major release to fall back to, and non-release
        builds are internal by nature, so neither counts as a minor release.
        """
        return self.head == "release" and len(self.tail) == 3

    @property
    def major_release(self):
        """
        The major release the version belongs to, e.g. `release-25.8` for `release-25.8.7`.
        Only meaningful for versions following the `release-<year>.<month>` scheme.
        """
        return f"{self.head}-{self.tail[0]}.{self.tail[1]}"

    @property
    def series(self):
        """
        Opaque key identifying the release series (head plus major.minor) the version belongs to.
        Versions sharing a series differ only in patch level, so `release-25.11` and
        `release-25.11.3` match while `release-25.11` and `beta-25.11` do not.
        """
        return self.head, *self.tail[:2]

    def __str__(self):
        return self.full

    def __lt__(self, other):
        return self.tail < other.tail

    def __le__(self, other):
        return self.tail <= other.tail

    def __gt__(self, other):
        return self.tail > other.tail

    def __ge__(self, other):
        return self.tail >= other.tail

    def __eq__(self, other):
        return self.tail == other.tail and self.head == other.head

    def __ne__(self, other):
        return self.tail != other.tail or self.head != other.head


def warn_if_minor_release(solver_version: Optional[str]) -> None:
    """
    Warn that a minor solver release was asked for, and point at its major release instead.

    Minor releases exist for internal testing, so a user pinning one is most likely pinning
    more than intended. The warning does not block the request. Call this wherever a user
    supplied version enters; `None` means the user pinned nothing and inherits a version.
    """
    if solver_version is None:
        return
    version = Flow360Version(solver_version)
    if not version.is_minor_release:
        return
    log.warning(
        f"You are running {version}, which is a minor solver release intended for internal "
        f"testing only and is not recommended for general use. Please use the major release "
        f"{version.major_release} instead."
    )
