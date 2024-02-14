"""Custom Flow360 exceptions"""

from typing import List

from .log import log


class Flow360Error(Exception):
    """Any error in flow360"""

    def __init__(self, message: str = None):
        """Log just the error message and then raise the Exception."""
        super().__init__(message)
        log.error(message)


# pylint: disable=redefined-builtin
class Flow360ValueError(Flow360Error):
    """Error with value."""


class Flow360TypeError(Flow360Error):
    """Error with type."""


class Flow360ConfigError(Flow360Error):
    """Error when configuring Flow360."""


class Flow360RuntimeError(Flow360Error):
    """Error when runtime."""


class Flow360KeyError(Flow360Error):
    """Could not find a key in a Flow360 dictionary."""


class Flow360ValidationError(Flow360Error):
    """Error when constructing FLow360 components."""


class Flow360ConfigurationError(Flow360Error):
    """Error with flow360 unit conversion."""

    msg: str
    field: List[str]
    dependency: List[str]

    def __init__(self, message: str = None, field: List[str] = None, dependency: List[str] = None):
        super().__init__(message)
        self.msg = message
        self.field = field
        self.dependency = dependency


class Flow360FileError(Flow360Error):
    """Error reading or writing to file."""


class Flow360CloudFileError(Flow360Error):
    """Error when getting file from cloud."""


class Flow360WebError(Flow360Error):
    """Error with the webAPI."""


class Flow360WebNotFoundError(Flow360Error):
    """Error with the webAPI."""


class Flow360AuthenticationError(Flow360Error):
    """Error authenticating a user through webapi webAPI."""


class Flow360AuthorisationError(Flow360Error):
    """Error authenticating a user through webapi webAPI."""


class Flow360DataError(Flow360Error):
    """Error accessing data."""


class Flow360ImportError(Flow360Error):
    """Error importing a package needed for Flow360."""


class Flow360NotImplementedError(Flow360Error):
    """Error when a functionality is not (yet) supported."""
