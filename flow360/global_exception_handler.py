"""
Module: custom_exception_wrapper for ValidationError
"""

import sys
import traceback

from pydantic.v1 import ValidationError as PydanticValidationError

from flow360.exceptions import Flow360ValidationError


def custom_exception_handler(exctype, value, trace_back):
    """
    handle global exceptions
    """
    if isinstance(value, PydanticValidationError):
        error_messages = ", ".join(map(str, value.errors()))  # Convert list of errors to a string
        error_traceback = "".join(
            traceback.format_exception(exctype, value, trace_back)
        )  # Get string representation of traceback
        raise Flow360ValidationError(f"{error_messages}\n{error_traceback}")
    sys.__excepthook__(exctype, value, trace_back)


sys.excepthook = custom_exception_handler
