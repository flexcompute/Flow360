from flow360.log import log, set_logging_level

set_logging_level("DEBUG")


def test_debug():
    log.debug("Debug log")


def test_info():
    log.info("Basic info")


def test_warning():
    log.warning("Warning log")


def test_error():
    log.error("Error log")


def test_critical():
    log.critical("Critical log")


test_debug()
test_info()
test_warning()
test_error()
test_critical()
