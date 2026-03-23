import re

from flow360.version import __solver_version__, __version__


def test_version():
    assert __version__ == "25.10.0b1"


def test_solver_version_matches_module_version():
    """For non-beta versions vA.B.C, __solver_version__ must be 'release-A.B'."""
    match = re.match(r"^(\d+)\.(\d+)\.\d+$", __version__)
    if not match:
        # Beta or non-standard version, skip
        return
    expected = f"release-{match.group(1)}.{match.group(2)}"
    assert __solver_version__ == expected, (
        f"__solver_version__ is '{__solver_version__}' but expected '{expected}' "
        f"for module version '{__version__}'"
    )
