import os
import tempfile

import pytest

from flow360.file_path import flow360_dir
from flow360.log import log, set_logging_file

"""
Before running all tests redirect all test logging to a temporary log file, turn off log rotation 
due to multi-threaded rotation being unsupported at this time
"""


def pytest_configure():
    fo = tempfile.NamedTemporaryFile()
    pytest.tmp_log_file = fo.name
    pytest.log_test_file = os.path.join(flow360_dir, "logs", "flow360_log_test.log")
    if os.path.exists(pytest.log_test_file):
        os.remove(pytest.log_test_file)
    set_logging_file(fo.name, level="DEBUG")


@pytest.fixture
def before_log_test():
    set_logging_file(pytest.log_test_file, level="DEBUG")


@pytest.fixture
def after_log_test():
    yield
    set_logging_file(pytest.tmp_log_file, level="DEBUG")
