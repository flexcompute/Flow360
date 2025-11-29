import os

from flow360.component.simulation.validation.validation_context import (
    ParamsValidationInfo,
    ValidationContext,
)

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg", force=True)

import tempfile

import pytest

from flow360.file_path import flow360_dir
from flow360.log import set_logging_file, toggle_rotation

"""
Before running all tests redirect all test logging to a temporary log file, turn off log rotation
due to multi-threaded rotation being unsupported at this time
"""

pytest_plugins = ["tests.utils", "tests.mock_server"]


def pytest_configure():
    fo = tempfile.NamedTemporaryFile()
    fo.close()  # Windows workaround for shared files
    pytest.tmp_log_file = fo.name
    pytest.log_test_file = os.path.join(flow360_dir, "logs", "flow360_log_test.log")
    if os.path.exists(pytest.log_test_file):
        os.remove(pytest.log_test_file)
    set_logging_file(fo.name, level="DEBUG")
    toggle_rotation(False)


@pytest.fixture
def before_log_test(request):
    set_logging_file(pytest.log_test_file, level="DEBUG")


@pytest.fixture
def after_log_test():
    yield
    set_logging_file(pytest.tmp_log_file, level="DEBUG")


@pytest.fixture
def mock_validation_context():
    return ValidationContext(
        levels=None, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )
