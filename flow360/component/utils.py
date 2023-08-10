"""
Utility functions
"""
import os
import uuid
from functools import wraps
from tempfile import NamedTemporaryFile

import zstandard as zstd

from ..cloud.utils import _get_progress, _S3Action
from ..exceptions import TypeError as FlTypeError
from ..exceptions import ValueError as FlValueError
from ..log import log


# pylint: disable=redefined-builtin
def is_valid_uuid(id, allow_none=False):
    """
    Checks if id is valid
    """
    if id is None and allow_none:
        return
    try:
        uuid.UUID(str(id))
    except Exception as exc:
        raise FlValueError(f"{id} is not a valid UUID.") from exc


def beta_feature(feature_name: str):
    """Prints warning message when used on a function which is BETA feature.

    Parameters
    ----------
    feature_name : str
        Name of the feature used in warning message
    """

    def wrapper(func):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            log.warning(f"{feature_name} is a beta feature.")
            value = func(*args, **kwargs)
            return value

        return wrapper_func

    return wrapper


# pylint: disable=bare-except
def _get_value_or_none(callable):
    try:
        return callable()
    except:
        return None


def validate_type(value, parameter_name: str, expected_type):
    """validate type

    Parameters
    ----------
    value :
        value to be validated
    parameter_name : str
        paremeter name - used for error message
    expected_type : type
        expected type for value

    Raises
    ------
    TypeError
        when value is not expected_type
    """
    if not isinstance(value, expected_type):
        raise FlTypeError(
            f"Expected type={expected_type} for {parameter_name}, but got value={value} (type={type(value)})"
        )


# pylint: disable=consider-using-with
def zstd_compress(file_path, output_file_path=None, compression_level=3):
    """
    Compresses the file located at 'file_path' using Zstandard compression.

    Args:
        file_path (str): The path to the input file that needs to be compressed.
        output_file_path (str, optional): The path where the compressed data will be written as a new file.
                                         If not provided, a temporary file with a ".zst" suffix will be created.
        compression_level (int, optional): The compression level used by the Zstandard compressor (default is 3).

    Returns:
        str or None: The path to the compressed file if successful, or None if an error occurred.
    """
    try:
        cctx = zstd.ZstdCompressor(level=compression_level)
        if not output_file_path:
            output_file_path = NamedTemporaryFile(suffix=".zst").name
        with open(file_path, "rb") as f_in, open(output_file_path, "wb") as f_out:
            with cctx.stream_writer(f_out) as compressor, _get_progress(
                _S3Action.COMPRESSING
            ) as progress:
                task_id = progress.add_task(
                    "Compressing file",
                    filename=os.path.basename(file_path),
                    total=os.path.getsize(file_path),
                )
                while True:
                    chunk = f_in.read(1024)
                    if not chunk:
                        break
                    compressor.write(chunk)
                    progress.update(task_id, advance=len(chunk))
        return output_file_path
    except (zstd.ZstdError, FileNotFoundError, IOError) as error:
        log.error(f"Error occurred while compressing the file: {error}")
        return None
