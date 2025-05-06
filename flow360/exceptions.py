"""Custom Flow360 exceptions"""

from typing import Any, List

from flow360.version import __version__

from .log import log


class Flow360Error(Exception):
    """Any error in flow360"""

    def __init__(self, message: str = None):
        """Log just the error message and then raise the Exception."""
        super().__init__(message)
        log.error(message + " [Flow360 client version: " + __version__ + "]")


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


class Flow360ErrorWithLocation(Exception):
    """
    Error with metadata on where the error is in the SimulationParams.
    This is used when NOT raising error from pydantic but we still want something similar to pd.ValidationError.
    """

    error_message: str
    input_value: Any
    location: list[str]

    def __init__(self, error_message, input_value, location: list[str] = None) -> None:
        """Log the error message and raise"""
        self.error_message = error_message
        self.input_value = input_value
        self.location = location
        log.error(error_message)
        super().__init__(error_message)

    def __str__(self) -> str:
        """Return a formatted string representing the error and its location."""
        if self.location is not None:
            error_location = "SimulationParams -> " + " -> ".join(self.location)
            return f"At {error_location}: {self.error_message}. [input_value = {self.input_value}]."
        return f"{self.error_message}. [input_value = {self.input_value}]."


class Flow360TranslationError(Flow360ErrorWithLocation):
    """Error when translating to SurfaceMeshing/VolumeMeshing/Case JSON."""


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

    auxiliary_json: dict

    def __init__(self, message: str = None, auxiliary_json: dict = None):
        super().__init__(message)
        self.auxiliary_json = auxiliary_json


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
