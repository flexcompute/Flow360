"""Custom Flow360 exceptions"""

# pylint: disable=unused-import
from typing import List

from flow360_schema.exceptions import (
    Flow360DeprecationError,
    Flow360Error,
    Flow360ErrorWithLocation,
    Flow360TranslationError,
    Flow360ValueError,
)


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


class Flow360BoundaryMissingError(Flow360Error):
    """Error when a boundary in simulation.json is not found in mesh metadata"""


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

    def __init__(self, message: str = None):
        Exception.__init__(self, message)


class Flow360DataError(Flow360Error):
    """Error accessing data."""


class Flow360ImportError(Flow360Error):
    """Error importing a package needed for Flow360."""


class Flow360NotImplementedError(Flow360Error):
    """Error when a functionality is not (yet) supported."""
