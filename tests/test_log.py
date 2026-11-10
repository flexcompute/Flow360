import functools
import io
import logging

import pytest
from rich.console import Console

from flow360.exceptions import Flow360ValueError
from flow360.log import LogHandler, log


@pytest.fixture
def log_output(monkeypatch):
    """Send log messages only to an in-memory console wide enough to keep each on one line."""
    output = io.StringIO()
    console = Console(file=output, width=1000, log_time=False, log_path=False)
    monkeypatch.setattr(log, "handlers", {"console": LogHandler(console, "DEBUG")})
    return output


def _logged_lines(output):
    return [line.rstrip() for line in output.getvalue().splitlines()]


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


@pytest.mark.parametrize(
    "log_method, level_name",
    [
        (log.debug, "DEBUG"),
        (log.info, "INFO"),
        (log.warning, "WARNING"),
        (log.error, "ERROR"),
        (log.critical, "CRITICAL"),
        (functools.partial(log.log, "INFO"), "INFO"),
    ],
    ids=["debug", "info", "warning", "error", "critical", "log"],
)
def test_message_without_args_keeps_percent(log_output, log_method, level_name):
    log_method("Draft 50% Tu, 100%")
    assert _logged_lines(log_output) == [f"{level_name}: Draft 50% Tu, 100%"]


def test_message_with_args_is_interpolated(log_output):
    log.info("Entity '%s' has %d faces", "wing", 3)
    assert _logged_lines(log_output) == ["INFO: Entity 'wing' has 3 faces"]


@pytest.mark.parametrize("method", ["info", "warning"])
@pytest.mark.parametrize(
    "message",
    [
        "Use asset[key] or draft.surfaces[key]",
        "Entity 'wing[/v2]' closes [/white] and [/]",
        "Faces Body1:Up:face and a:b:c",
        "Saved to C:\\results\\[run]\\ and C:\\results\\",
    ],
)
def test_markup_and_emoji_codes_are_logged_as_is(log_output, method, message):
    """rich would drop "[key]", raise on "[/v2]", turn ":Up:" into an emoji and swallow the
    backslash before "[run]"."""
    getattr(log, method)(message)
    assert _logged_lines(log_output) == [f"{method.upper()}: {message}"]


def test_schema_messages_are_logged_as_is(log_output):
    logging.getLogger("flow360_schema.test").warning("Schema text [/v2] :up:")
    with pytest.raises(Flow360ValueError):
        raise Flow360ValueError("Entity 'wing[/v2]' not found")
    assert _logged_lines(log_output) == [
        "WARNING: Schema text [/v2] :up:",
        "ERROR: Entity 'wing[/v2]' not found",
    ]


test_debug()
