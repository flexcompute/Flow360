import json
import os
import re

import pytest
import toml

from flow360.component.simulation.framework.updater import VERSION_MILESTONES
from flow360.component.simulation.framework.updater_utils import Flow360Version
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.validation.validation_context import ALL
from flow360.version import __solver_version__, __version__


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_version_consistency():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    pyproject_path = os.path.join(project_root, "pyproject.toml")

    with open(pyproject_path, "r") as file:
        config = toml.load(file)

    pyproject_version = config["tool"]["poetry"]["version"]
    assert pyproject_version == "v" + __version__, (
        f"Version mismatch: pyproject.toml version is {pyproject_version}, "
        f"but __version__ is {__version__}"
    )


def test_default_solver_version_matches_module_version():
    """For non-beta releases (vA.B.C), the default solver version must be 'release-A.B'."""
    version = Flow360Version(__version__)
    if re.search(r"b\d+$", __version__):
        pytest.skip("Beta version, skipping solver version check")
    expected_solver_version = f"release-{version.major}.{version.minor}"
    assert __solver_version__ == expected_solver_version, (
        f"Default solver version mismatch: __solver_version__ is '{__solver_version__}', "
        f"but expected '{expected_solver_version}' based on __version__ '{__version__}'"
    )


def test_version_greater_than_highest_updater_version():
    current_python_version = Flow360Version(__version__)
    assert (
        current_python_version >= VERSION_MILESTONES[-1][0]
    ), "Highest version updater can handle is higher than Python client version. This is not allowed."


def test_deserialization_with_updater():
    simulation_path = os.path.join("..", "data", "simulation", "simulation_24_11_0.json")
    with open(simulation_path, "r") as file:
        params = json.load(file)

    validate_model(
        params_as_dict=params,
        root_item_type="VolumeMesh",
        validated_by=ValidationCalledBy.LOCAL,
        validation_level=ALL,
    )
