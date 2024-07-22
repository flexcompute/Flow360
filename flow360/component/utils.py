"""
Utility functions
"""

import os
import re
import uuid
from functools import wraps
from tempfile import NamedTemporaryFile

import zstandard as zstd

from ..accounts_utils import Accounts
from ..cloud.utils import _get_progress, _S3Action
from ..error_messages import shared_submit_warning
from ..exceptions import Flow360TypeError, Flow360ValueError
from ..log import log

SUPPORTED_GEOMETRY_FILE_PATTERNS = [
    ".csm",
    ".egads",
    ".sat",
    ".sab",
    ".asat",
    ".asab",
    ".iam",
    ".catpart",
    ".catproduct",
    ".gt",
    ".prt",
    ".prt.*",
    ".asm.*",
    ".par",
    ".asm",
    ".psm",
    ".sldprt",
    ".sldasm",
    ".stp",
    ".step",
    ".x_t",
    ".xmt_txt",
    ".x_b",
    ".xmt_bin",
    ".3dm",
    ".ipt",
]


def match_file_pattern(patterns, filename):
    """
    Check if filename matches a pattern
    """
    for pattern in patterns:
        if re.search(pattern + "$", filename.lower()) is not None:
            return True
    return False


# pylint: disable=redefined-builtin
def is_valid_uuid(id, allow_none=False, valid_prefixes=None):
    """
    Checks if id is valid
    """
    if valid_prefixes is None:
        valid_prefixes = ["folder-", "g-","geo-"]
    if id is None and allow_none:
        return
    try:
        if id:
            for prefix in valid_prefixes:
                if id.startswith(prefix):
                    id = id[len(prefix) :]
                    break
        uuid.UUID(str(id))
    except ValueError as exc:
        raise Flow360ValueError(f"{id} is not a valid UUID.") from exc


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


def shared_account_confirm_proceed():
    """
    Prompts confirmation from user when submitting a resource from a shared account
    """
    email = Accounts.shared_account_info()
    if email is not None and not Accounts.shared_account_submit_is_confirmed():
        log.warning(shared_submit_warning(email))
        print("Are you sure you want to proceed? (y/n): ")
        while True:
            try:
                value = input()
                if value.lower() == "y":
                    Accounts.shared_account_confirm_submit()
                    return True
                if value.lower() == "n":
                    return False
                print("Enter a valid value (y/n): ")
                continue
            except ValueError:
                print("Invalid input type")
                continue
    else:
        return True


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
        raise Flow360TypeError(
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
    except (OSError, zstd.ZstdError, FileNotFoundError) as error:
        log.error(f"Error occurred while compressing the file: {error}")
        return None


##::  -------- Expression preprocessing functions --------


def convert_if_else(expression: str):
    """
    Convert if else to use ? : syntax
    """
    if expression.find("if") != -1:
        regex = r"\s*if\s*\(\s*(.*?)\s*\)\s*(.*?)\s*;\s*else\s*(.*?)\s*;\s*"
        subst = r"(\1) ? (\2) : (\3);"
        expression = re.sub(regex, subst, expression)
    return expression


def convert_caret_to_power(input_str):
    """
    Convert caret to pow function to comply with C++ syntax
    """
    enclosed = r"\([^(^)]+\)"
    non_negative_num = r"\d+(?:\.\d+)?(?:e[-+]?\d+)?"
    number = r"[+-]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
    symbol = r"\b[a-zA-Z_][a-zA-Z_\d]*\b"
    base = rf"({enclosed}|{symbol}|{non_negative_num})"
    exponent = rf"({enclosed}|{symbol}|{number})"
    pattern = rf"{base}\s*\^\s*{exponent}"
    result = input_str
    while re.search(pattern, result):
        result = re.sub(pattern, r"powf(\1, \2)", result)
    return result


def add_trailing_semicolon(input_str):
    """
    Add trailing semicolon to comply with C++ syntax
    """
    regex = r";\s*$"
    if not re.search(regex, input_str):
        input_str += ";"
    return input_str


def convert_legacy_names(input_str):
    """
    Convert legacy var name to new ones.
    """
    old_names = ["rotMomentX", "rotMomentY", "rotMomentZ", "xyz"]
    new_names = ["momentX", "momentY", "momentZ", "coordinate"]
    result = input_str
    for old_name, new_name in zip(old_names, new_names):
        pattern = r"\b(" + old_name + r")\b"
        while re.search(pattern, result):
            result = re.sub(pattern, new_name, result)
    return result


def _process_string_expression(expression: str):
    """
    All in one funciton to precess string expressions
    """
    if not isinstance(expression, str):
        return expression
    expression = str(expression)
    expression = add_trailing_semicolon(expression)
    expression = convert_if_else(expression)
    expression = convert_caret_to_power(expression)
    expression = convert_legacy_names(expression)
    return expression


def process_expressions(input_expressions):
    """
    All in one funciton to precess expressions in form of tuple or single string
    """
    if isinstance(input_expressions, (str, float, int)):
        return _process_string_expression(str(input_expressions))

    if isinstance(input_expressions, tuple):
        prcessed_expressions = []
        for expression in input_expressions:
            prcessed_expressions.append(_process_string_expression(expression))
        return tuple(prcessed_expressions)
    return input_expressions


##::  -------- dict preprocessing functions --------


def remove_properties_with_prefix(data, prefix):
    """
    Recursively removes properties from a nested dictionary and its lists
    whose keys start with a specified prefix.

    Parameters
    ----------
    data : dict or list or scalar
        The input data, which can be a nested dictionary, a list, or a scalar value.

    prefix : str
        The prefix used to filter properties. Properties with keys starting with
        this prefix will be removed.

    Returns
    -------
    dict or list or scalar
        Processed data with properties removed based on the specified prefix.
    """

    if isinstance(data, dict):
        return {
            key: remove_properties_with_prefix(value, prefix)
            for key, value in data.items()
            if not key.startswith(prefix)
        }
    if isinstance(data, list):
        return [remove_properties_with_prefix(item, prefix) for item in data]
    return data


def remove_properties_by_name(data, name_to_remove):
    """
    Recursively removes properties from a nested dictionary and its lists
    whose keys start with a specified prefix.

    Parameters
    ----------
    data : dict or list or scalar
        The input data, which can be a nested dictionary, a list, or a scalar value.

    name_to_remove : str
        The name_to_remove used to filter properties. Properties with keys equal to
        this name_to_remove will be removed.

    Returns
    -------
    dict or list or scalar
        Processed data with properties removed based on the specified prefix.
    """

    if isinstance(data, dict):
        return {
            key: remove_properties_by_name(value, name_to_remove)
            for key, value in data.items()
            if not key == name_to_remove
        }
    if isinstance(data, list):
        return [remove_properties_by_name(item, name_to_remove) for item in data]
    return data
