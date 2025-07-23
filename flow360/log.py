"""Logging for Flow360."""

import os
from datetime import datetime
from typing import Union

from rich.console import Console
from typing_extensions import Literal

from .file_path import flow360_dir
from .version import __version__ as version

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogValue = Union[int, LogLevel]

# Logging levels compatible with logging module
_level_value = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

_level_name = {v: k for k, v in _level_value.items()}
_level_print_style = {
    "DEBUG": "DEBUG",
    "INFO": "[cyan]INFO[/cyan]",
    "WARNING": "[yellow]WARNING[/yellow]",
    "ERROR": "[bold red]ERROR[/bold red]",
    "CRITICAL": "[bold underline red]CRITICAL[/bold underline red]",
}

DEFAULT_LEVEL = "INFO"


def _get_level_int(level: LogValue) -> int:
    """Get the integer corresponding to the level string."""
    if isinstance(level, int):
        return level

    level_upper = level.upper()
    if level_upper != level:
        log.warning(
            f"'{level}' provided as a logging level. "
            "In the future, only upper-case logging levels may be specified. "
            f"This value will be converted to upper case '{level_upper}'."
        )
    if level_upper not in _level_value:
        # We don't want to import ConfigError to avoid a circular dependency
        raise ValueError(
            f"logging level {level_upper} not supported, must be "
            "'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'"
        )
    return _level_value[level_upper]


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-instance-attributes
class LogHandler:
    """Handle log messages depending on log level"""

    def __init__(
        self,
        console: Console,
        level: LogValue,
        fname: str = None,
    ):
        self.level = _get_level_int(level)
        self.console = console
        self.backup_count = 10
        self.max_bytes = 1000000
        self.fname = fname
        self.previous_logged_time = None
        self.previous_logged_version = None
        self.is_rotating = True

    def handle(self, level, level_name, message, log_time=False):
        """Output log messages depending on log level"""
        try:
            if self.fname is not None and self.is_rotating and self.should_roll_over(message):
                self.do_roll_over()
        except OSError as error:
            self.console.log(
                _level_print_style.get(_level_value["ERROR"], "unknown"),
                "Fail to Rollover " + str(error),
                sep=": ",
            )
        current_time = datetime.now().strftime("%Y-%m-%d-%H")
        if level >= self.level:
            if log_time and (
                self.previous_logged_time != current_time or self.previous_logged_version != version
            ):
                self.previous_logged_time = current_time
                self.previous_logged_version = version
                self.console.log(f"{current_time}, version {version}\n")
            self.console.log(
                _level_print_style.get(level_name, "unknown"),
                message,
                sep=": ",
            )

    def rotate(self, source, dest):
        """
        Rotate a log file by renaming the source file to the destination file.

        Args:
            source (str): Path of the source log file.
            dest (str): Path of the destination log file.

        Returns:
            None
        """
        if os.path.exists(source) and not os.path.exists(dest):
            os.rename(source, dest)

    def rotation_filename(self, name, counter):
        """
        Generate a rotated filename based on the original filename and a counter.

        Args:
            name (str): Original filename or base filename.
            counter (int): Counter value to be included in the rotated filename.

        Returns:
            str: Rotated filename with the format "{name}{formatted_time}.{counter}".

        """
        root_ext = os.path.splitext(name)

        return f"{root_ext[0]}_{counter}{root_ext[1]}"

    def do_roll_over(self):
        """
        Generate a rotated filename based on the original filename and a counter.

        Args:
            name (str): Original filename or base filename.
            counter (int): Counter value to be included in the rotated filename.

        Returns:
            str: Rotated filename with the format "{name}{formatted_time}.{counter}".
        """

        if self.backup_count > 0:
            self.console.file.close()
            for i in range(self.backup_count - 1, 0, -1):
                sfn = self.rotation_filename(self.fname, i)
                dfn = self.rotation_filename(self.fname, i + 1)
                if os.path.isfile(sfn):
                    if os.path.isfile(dfn):
                        os.remove(dfn)
                    self.rotate(sfn, dfn)
            dfn = self.rotation_filename(self.fname, 1)
            if os.path.isfile(dfn):
                os.remove(dfn)
            self.rotate(self.fname, dfn)
            # pylint: disable=consider-using-with,unspecified-encoding
            self.console.file = open(self.fname, self.console.file.mode)

    def should_roll_over(self, message):
        """
        Determine if a rollover should occur based on the supplied message.

        Args:
            message (str): The message to be logged.

        Returns:
            bool: True if a rollover should occur, False otherwise.
        """
        # See bpo-45401: Never rollover anything other than regular files
        if not os.path.exists(self.fname) or not os.path.isfile(self.fname):
            return False
        size = os.path.getsize(self.fname)
        if self.max_bytes > 0:  # are we rolling over?
            try:
                if size + len(message) >= self.max_bytes:
                    return True
            except OSError as error:
                self.console.log(
                    _level_print_style.get(_level_value["ERROR"], "unknown"),
                    str(error),
                    sep=": ",
                )
        return False


class Logger:
    """Custom logger to avoid the complexities of the logging module"""

    log_to_file = True

    def __init__(self):
        self.handlers = {}

    def _log(self, level: int, level_name: str, message: str) -> None:
        """Distribute log messages to all handlers"""
        for handler_type, handler in self.handlers.items():
            if handler_type == "file":
                if not self.log_to_file:
                    continue
                handler.handle(level, level_name, message, True)
            else:
                handler.handle(level, level_name, message)

    def log(self, level: LogValue, message: str, *args) -> None:
        """Log (message) % (args) with given level"""
        if isinstance(level, str):
            level_name = level
            level = _get_level_int(level)
        else:
            level_name = _level_name.get(level, "unknown")
        self._log(level, level_name, message % args)

    def debug(self, message: str, *args) -> None:
        """Log (message) % (args) at debug level"""
        self._log(_level_value["DEBUG"], "DEBUG", message % args)

    def info(self, message: str, *args) -> None:
        """Log (message) % (args) at info level"""
        self._log(_level_value["INFO"], "INFO", message % args)

    def warning(self, message: str, *args) -> None:
        """Log (message) % (args) at warning level"""
        message = message % args
        message = f"[white]{message}[/white]"
        self._log(_level_value["WARNING"], "WARNING", message)

    def error(self, message: str, *args) -> None:
        """Log (message) % (args) at error level"""
        message = message % args
        message = f"[white]{message}[/white]"
        self._log(_level_value["ERROR"], "ERROR", message)

    def critical(self, message: str, *args) -> None:
        """Log (message) % (args) at critical level"""
        message = message % args
        message = f"[white]{message}[/white]"
        self._log(_level_value["CRITICAL"], "CRITICAL", message)

    def status(self, status: str = ""):
        """Returns status context to show spinner"""
        return log.handlers["console"].console.status(status)


# Initialize FLow360's logger
log = Logger()


def set_logging_level(level: LogValue = DEFAULT_LEVEL) -> None:
    """Set flow360 console logging level priority.
    Parameters
    ----------
    level : str
        The lowest priority level of logging messages to display. One of ``{'DEBUG', 'INFO',
        'WARNING', 'ERROR', 'CRITICAL'}`` (listed in increasing priority).
    """
    if "console" in log.handlers:
        log.handlers["console"].level = _get_level_int(level)


def set_logging_console(stderr: bool = False) -> None:
    """Set stdout or stderr as console output
    Parameters
    ----------
    stderr : bool
        If False, logs are directed to stdout, otherwise to stderr.
    """
    if "console" in log.handlers:
        previous_level = log.handlers["console"].level
    else:
        previous_level = DEFAULT_LEVEL
    log.handlers["console"] = LogHandler(Console(stderr=stderr, log_path=False), previous_level)


def set_logging_file(
    fname: str,
    filemode: str = "a",
    level: LogValue = DEFAULT_LEVEL,
    back_up_count: int = 10,
    max_bytes: int = 1000000,
) -> None:
    """Set a file to write log to, independently from the stdout and stderr
    output chosen using :meth:`set_logging_level`.
    Parameters
    ----------
    fname : str
        Path to file to direct the output to.
    filemode : str
        'w' or 'a', defining if the file should be overwritten or appended.
    level : str
        One of ``{'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}``. This is set for the file
        independently of the console output level set by :meth:`set_logging_level`.
    back_up_count : int
        How many backup log files are preserved when rotating log files
    max_bytes : int
        Maximum log file size in bytes before a log rotation is performed
    """
    if filemode not in "wa":
        raise ValueError("filemode must be either 'w' or 'a'")

    # Close previous handler, if any
    if "file" in log.handlers:
        try:
            log.handlers["file"].console.file.close()
        except Exception as error:  # pylint: disable=broad-exception-caught
            del log.handlers["file"]
            log.warning(f"Log file could not be closed: {error}")

    try:
        # pylint: disable=consider-using-with,unspecified-encoding
        file = open(fname, filemode, encoding="utf-8")
    except OSError:
        log.warning(f"File {fname} could not be opened. Logging to file disabled.")
        return

    log.handlers["file"] = LogHandler(Console(file=file, log_path=False), level, fname)
    log.handlers["file"].back_up_count = back_up_count
    log.handlers["file"].max_bytes = max_bytes


def toggle_rotation(rotate: bool):
    """Toggle log file rotation (without log rotation logging
    is thread-safe on every platform, but this may generate
    large file sizes)
    Parameters
    ----------
    rotate : bool
        Enable or disable log rotation
    """
    if "file" in log.handlers:
        log.handlers["file"].is_rotating = rotate


# Set default logging output
set_logging_console()

log_dir = flow360_dir + "logs"
try:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

except OSError as err:
    log.warning(f"Could not setup file logging: {err}")
