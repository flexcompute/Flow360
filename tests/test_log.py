from flow360.log import log, set_logging_level

set_logging_level("DEBUG")


def test_debug():
    log.debug("Debug log")
    log.debug("Debug log string %s， number %d", "arg", 1)


def test_info():
    log.info("Basic info")
    log.info("Basic info string %s， number %d", "arg", 1)


def test_warning():
    log.warning("Warning log")
    log.warning("Warning log string %s， number %d", "arg", 1)


def test_error():
    log.error("Error log")
    log.error("Error log string %s， number %d", "arg", 1)


def test_critical():
    log.critical("Critical log string %s， number %d", "arg", 1)


test_debug()
test_info()
test_warning()
test_error()
test_critical()
