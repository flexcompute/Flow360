import os

import pytest
import toml

from flow360.version import __version__


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_version_consistency():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    pyproject_path = os.path.join(project_root, "pyproject.toml")

    with open(pyproject_path, "r") as file:
        config = toml.load(file)

    pyproject_version = config["tool"]["poetry"]["version"]
    assert pyproject_version == __version__, (
        f"Version mismatch: pyproject.toml version is {pyproject_version}, "
        f"but __version__ is {__version__}"
    )
