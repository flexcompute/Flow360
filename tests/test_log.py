import pytest

from flow360.log import Logger, log, set_logging_level


@pytest.mark.usefixtures("before_log_test", "after_log_test")
def test_debug():
    log.debug("Debug log")
    log.debug("Debug log string %s, number %d", "arg", 1)


@pytest.mark.usefixtures("before_log_test", "after_log_test")
def test_info():
    log.info("Basic info")
    log.info("Basic info string %s, number %d", "arg", 1)


@pytest.mark.usefixtures("before_log_test", "after_log_test")
def test_warning():
    log.warning("Warning log")
    log.warning("Warning log string %s, number %d", "arg", 1)


@pytest.mark.usefixtures("before_log_test", "after_log_test")
def test_error():
    log.error("Error log")
    log.error("Error log string %s, number %d", "arg", 1)


@pytest.mark.usefixtures("before_log_test", "after_log_test")
def test_critical():
    log.critical("Critical log string %s, number %d", "arg", 1)


test_debug()
