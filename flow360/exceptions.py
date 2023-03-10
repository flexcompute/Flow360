"""Custom Flow360 exceptions"""

from .log import log


class Flow360Error(Exception):
    """Any error in flow360"""

    def __init__(self, message: str = None):
        """Log just the error message and then raise the Exception."""
        super().__init__(message)
        log.error(message)


# pylint: disable=redefined-builtin
class ValueError(Flow360Error):
    """Error with value."""


class ConfigError(Flow360Error):
    """Error when configuring Flow360."""


class Flow360KeyError(Flow360Error):
    """Could not find a key in a Flow360 dictionary."""


class ValidationError(Flow360Error):
    """Error when constructing FLow360 components."""


class FileError(Flow360Error):
    """Error reading or writing to file."""


class WebError(Flow360Error):
    """Error with the webAPI."""


class WebNotFoundError(Flow360Error):
    """Error with the webAPI."""


class AuthenticationError(Flow360Error):
    """Error authenticating a user through webapi webAPI."""


class AuthorisationError(Flow360Error):
    """Error authenticating a user through webapi webAPI."""


class DataError(Flow360Error):
    """Error accessing data."""


class Flow360ImportError(Flow360Error):
    """Error importing a package needed for tidy3d."""


class Flow360NotImplementedError(Flow360Error):
    """Error when a functionality is not (yet) supported."""
