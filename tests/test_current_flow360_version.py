import re

import pytest

from flow360.version import __solver_version__, __version__, _derive_solver_version


@pytest.mark.parametrize(
    "client_version, expected",
    [
        ("25.11.1", "release-25.11"),
        ("25.12.0b1", "release-25.12"),
        ("25.11.2.dev20260819143000+gab12cd34", "release-25.11"),
        ("25.12.0b1.dev20260819143000+gab12cd34", "release-25.12"),
        ("26.1.0", "release-26.1"),
    ],
)
def test_derive_solver_version(client_version, expected):
    """The default solver line is always the client version's first two digits."""
    assert _derive_solver_version(client_version) == expected


def test_solver_version_derived_from_module_version():
    assert __solver_version__ == _derive_solver_version(__version__)
    assert re.fullmatch(r"release-\d+\.\d+", __solver_version__)
