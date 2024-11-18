"""
Utility functions
"""

import itertools
import os
import re
import shutil
from enum import Enum
from functools import wraps
from tempfile import NamedTemporaryFile
from typing import Generic, Iterable, Literal, Protocol, TypeVar

import zstandard as zstd

from ..accounts_utils import Accounts
from ..cloud.s3_utils import get_local_filename_and_create_folders
from ..cloud.utils import _get_progress, _S3Action
from ..error_messages import shared_submit_warning
from ..exceptions import (
    Flow360FileError,
    Flow360RuntimeError,
    Flow360TypeError,
    Flow360ValueError,
)
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


def _valid_resource_id(resource_id) -> bool:
    """
    Returns:
    1. Whether the resource_id is valid
    2. The content of the resource_id
    """
    if not isinstance(resource_id, str):
        raise ValueError(f"resource_id must be a string, but got {type(resource_id)}")

    pattern = re.compile(
        r"""
        ^                     # Start of the string
        ROOT\.FLOW360|        # accept root folder
        (?P<content>          # Start of the content group
        [0-9a-zA-Z,-]{16,}    # Content: at least 16 characters, alphanumeric, comma, or dash
        )$                    # End of the string
        """,
        re.VERBOSE,
    )

    match = pattern.match(resource_id)
    if not match:
        return False

    return True


# pylint: disable=redefined-builtin
def is_valid_uuid(id, allow_none=False):
    """
    Checks if id is valid
    """

    if id is None:
        if allow_none is True:
            return
        raise Flow360ValueError("None is not a valid id.")

    try:
        is_valid = _valid_resource_id(id)
        if is_valid is False:
            raise ValueError(f"{id} is not a valid UUID.")
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


def get_mapbc_from_ugrid(ugrid):
    """
    return associated mapbc file name from the ugrid mesh file
    """
    mapbc = ugrid.replace(".lb8.ugrid", ".mapbc")
    mapbc = mapbc.replace(".b8.ugrid", ".mapbc")
    mapbc = mapbc.replace(".ugrid", ".mapbc")
    return mapbc


class MeshFileFormat(Enum):
    """
    Mesh file format
    """

    UGRID = "aflr3"
    CGNS = "cgns"
    STL = "stl"

    def ext(self) -> str:
        """
        Get the extention for a file name.
        :return:
        """
        if self is MeshFileFormat.UGRID:
            return ".ugrid"
        if self is MeshFileFormat.CGNS:
            return ".cgns"
        if self is MeshFileFormat.STL:
            return ".stl"
        return ""

    @classmethod
    def detect(cls, file: str):
        """
        detects mesh format from filename
        """
        ext = os.path.splitext(file)[1].lower()
        if ext == MeshFileFormat.UGRID.ext():
            return MeshFileFormat.UGRID
        if ext == MeshFileFormat.CGNS.ext():
            return MeshFileFormat.CGNS
        if ext == MeshFileFormat.STL.ext():
            return MeshFileFormat.STL
        raise Flow360FileError(f"Unsupported file format {file}")


class UGRIDEndianness(Enum):
    """
    UGRID endianness
    """

    LITTLE = "little"
    BIG = "big"
    NONE = None

    def ext(self) -> str:
        """
        Get the extention for a file name.
        :return:
        """
        if self is UGRIDEndianness.LITTLE:
            return ".lb8"
        if self is UGRIDEndianness.BIG:
            return ".b8"
        return ""

    @classmethod
    def detect(cls, file: str):
        """
        detects endianess UGRID mesh from filename
        """
        if MeshFileFormat.detect(file) is not MeshFileFormat.UGRID:
            return UGRIDEndianness.NONE
        basename = os.path.splitext(file)[0]
        ext = os.path.splitext(basename)[1]
        if ext == UGRIDEndianness.LITTLE.ext():
            return UGRIDEndianness.LITTLE
        if ext == UGRIDEndianness.BIG.ext():
            return UGRIDEndianness.BIG
        if ext == UGRIDEndianness.NONE.ext():
            return UGRIDEndianness.NONE
        raise Flow360RuntimeError(f"Unknown endianness for file {file}")


class CompressionFormat(Enum):
    """
    Volume mesh file format
    """

    GZ = "gz"
    TARGZ = "tar.gz"
    BZ2 = "bz2"
    ZST = "zst"
    NONE = None

    def ext(self) -> str:
        """
        Get the extention for a file name.
        :return:
        """
        if self is CompressionFormat.GZ:
            return ".gz"
        if self is CompressionFormat.TARGZ:
            return ".tar.gz"
        if self is CompressionFormat.BZ2:
            return ".bz2"
        if self is CompressionFormat.ZST:
            return ".zst"
        return ""

    @classmethod
    def detect(cls, file: str):
        """
        detects compression from filename
        """
        if file.lower().endswith(CompressionFormat.TARGZ.ext()):
            return CompressionFormat.TARGZ, file[: -1 * len(CompressionFormat.TARGZ.ext())]

        file_name, ext = os.path.splitext(file)
        ext = ext.lower()
        if ext == CompressionFormat.GZ.ext():
            return CompressionFormat.GZ, file_name
        if ext == CompressionFormat.BZ2.ext():
            return CompressionFormat.BZ2, file_name
        if ext == CompressionFormat.ZST.ext():
            return CompressionFormat.ZST, file_name
        return CompressionFormat.NONE, file


class MeshNameParser:
    """
    parse a given mesh name to handle endianness, format and compression
    """

    def __init__(self, input_mesh_file):
        self._compression, self._file_name_no_compression = CompressionFormat.detect(
            input_mesh_file
        )
        self._format = MeshFileFormat.detect(self._file_name_no_compression)
        self._endianness = UGRIDEndianness.detect(self._file_name_no_compression)

    # pylint: disable=missing-function-docstring
    @property
    def compression(self):
        return self._compression

    # pylint: disable=missing-function-docstring
    @property
    def file_name_no_compression(self):
        return self._file_name_no_compression

    # pylint: disable=missing-function-docstring
    @property
    def format(self):
        return self._format

    # pylint: disable=missing-function-docstring
    @property
    def endianness(self):
        return self._endianness

    # pylint: disable=missing-function-docstring
    def is_ugrid(self):
        return self.format is MeshFileFormat.UGRID

    # pylint: disable=missing-function-docstring
    def is_compressed(self):
        return self.compression is not CompressionFormat.NONE

    # pylint: disable=missing-function-docstring
    def is_valid_surface_mesh(self):
        return self.format in [MeshFileFormat.UGRID, MeshFileFormat.CGNS, MeshFileFormat.STL]

    # pylint: disable=missing-function-docstring
    def is_valid_volume_mesh(self):
        return self.format in [MeshFileFormat.UGRID, MeshFileFormat.CGNS]

    # pylint: disable=missing-function-docstring
    def get_associated_mapbc_filename(self):
        if not self.is_ugrid():
            raise RuntimeError(
                "Invalid operation to get mapbc file,  since the mesh is not in UGRID format."
            )
        mapbc_file = get_mapbc_from_ugrid(self.file_name_no_compression)
        return mapbc_file

    # pylint: disable=missing-function-docstring
    @staticmethod
    def all_patterns(mesh_type: Literal["surface", "volume"]):
        endian_format = [el.ext() for el in UGRIDEndianness]
        mesh_format = [MeshFileFormat.UGRID.ext()]

        prod = itertools.product(endian_format, mesh_format)

        mesh_format = [endianness + file for (endianness, file) in prod]

        allowed = []
        if mesh_type == "surface":
            allowed = [MeshFileFormat.UGRID, MeshFileFormat.CGNS, MeshFileFormat.STL]
        elif mesh_type == "volume":
            allowed = [MeshFileFormat.UGRID, MeshFileFormat.CGNS]

        mesh_format = mesh_format + [el.ext() for el in allowed]
        compression = [el.ext() for el in CompressionFormat]

        prod = itertools.product(mesh_format, compression)

        return [file + compression for (file, compression) in prod]


def storage_size_formatter(size_in_bytes):
    """
    Format the size in bytes into a human-readable format (B, kB, MB, GB).

    Parameters
    ----------
    size_in_bytes : int
        The size in bytes to be formatted.

    Returns
    -------
    str
        A string representing the size in the most appropriate unit (B, kB, MB, GB).
    """
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    if size_in_bytes < 1024**2:
        return f"{size_in_bytes / 1024:.2f} kB"
    if size_in_bytes < 1024**3:
        return f"{size_in_bytes / (1024 ** 2):.2f} MB"
    return f"{size_in_bytes / (1024 ** 3):.2f} GB"


# pylint: disable=too-few-public-methods
class HasId(Protocol):
    """
    Protocol for objects that have an `id` attribute.

    Attributes
    ----------
    id : str
        Unique identifier for the asset.
    """

    id: str


AssetT = TypeVar("AssetT", bound=HasId)


class ProjectAssetCache(Generic[AssetT]):
    """
    A cache to manage project assets with a unique ID system.

    Attributes
    ----------
    current_asset_id : str, optional
        The ID of the currently set asset.
    asset_cache : dict of str to AssetT
        Dictionary storing assets with their IDs as keys.
    """

    current_asset_id: str = None
    asset_cache: dict[str, AssetT] = {}

    def get_asset(self, asset_id: str = None) -> AssetT:
        """
        Retrieve an asset from the cache by ID.

        Parameters
        ----------
        asset_id : str, optional
            The ID of the asset to retrieve. If None, retrieves the asset with `current_id`.

        Returns
        -------
        AssetT
            The asset associated with the specified `asset_id`.

        Raises
        ------
        Flow360ValueError
            If the cache is empty or if the asset is not found.
        """
        if not self.asset_cache:
            raise Flow360ValueError("Cache is empty, no assets are available")

        asset = self.asset_cache.get(self.current_asset_id if not asset_id else asset_id)

        if not asset:
            raise Flow360ValueError(f"{asset_id} is not available in the project.")

        return asset

    def get_ids(self) -> Iterable[str]:
        """
        Retrieve all asset IDs in the cache.

        Returns
        -------
        Iterable[str]
            An iterable of asset IDs.
        """
        return list(self.asset_cache.keys())

    def add_asset(self, asset: AssetT):
        """
        Add an asset to the cache.

        Parameters
        ----------
        asset : AssetT
            The asset to add. Must have a unique `id` attribute.
        """
        self.asset_cache[asset.id] = asset
        self.current_asset_id = asset.id

    def set_id(self, asset_id: str):
        """
        Set the current ID to the given `asset_id`, if it exists in the cache.

        Parameters
        ----------
        asset_id : str
            The ID to set as the current ID.

        Raises
        ------
        Flow360ValueError
            If the specified `asset_id` does not exist in the cache.
        """
        if asset_id not in self.asset_cache:
            raise Flow360ValueError(f"{asset_id} is not available in the project.")

        self.current_asset_id = asset_id


def _local_download_overwrite(local_storage_path, class_name):
    def _local_download_file(
        file_name: str,
        to_file: str = None,
        to_folder: str = ".",
        **_,
    ):
        expected_local_file = os.path.join(local_storage_path, file_name)
        if not os.path.exists(expected_local_file):
            raise RuntimeError(
                f"File {expected_local_file} not found. Make sure the file exists when using "
                f"{class_name}.from_local_storage()."
            )
        new_local_file = get_local_filename_and_create_folders(file_name, to_file, to_folder)
        if new_local_file != expected_local_file:
            shutil.copy(expected_local_file, new_local_file)

    return _local_download_file
