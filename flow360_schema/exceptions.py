"""Flow360 schema exception classes."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Flow360Error(Exception):
    """Base error for all Flow360 schema exceptions."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message)
        # TODO: logging at construction is problematic — if the exception is caught
        # and handled, this still emits an error log. Move logging to the catch site.
        logger.error(message)


class Flow360FileError(Flow360Error):
    """Error reading or writing to file."""


class Flow360ValueError(Flow360Error):
    """Error with value."""


class Flow360DeprecationError(Flow360Error):
    """Error when a deprecated feature is used."""


class Flow360ErrorWithLocation(Flow360Error):
    """
    Error with metadata on where the error is in the SimulationParams.
    Used when NOT raising error from pydantic but we still want something similar to
    pd.ValidationError.
    """

    error_message: str
    input_value: Any
    location: list[str] | None

    def __init__(
        self,
        error_message: str,
        input_value: Any,
        location: list[str] | None = None,
    ) -> None:
        self.error_message = error_message
        self.input_value = input_value
        self.location = location
        # Flow360Error.__init__ handles both Exception.__init__ and logger.error.
        super().__init__(error_message)

    def __str__(self) -> str:
        if self.location is not None:
            error_location = "SimulationParams -> " + " -> ".join(self.location)
            return f"At {error_location}: {self.error_message}. [input_value = {self.input_value}]."
        return f"{self.error_message}. [input_value = {self.input_value}]."


class Flow360TranslationError(Flow360ErrorWithLocation):
    """Error when translating to SurfaceMeshing/VolumeMeshing/Case JSON."""
