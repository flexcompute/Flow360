"""
Flow360 base Model
"""

import json
import os
import re
import shutil
import time
import traceback
from abc import ABCMeta
from datetime import datetime
from enum import Enum
from functools import wraps
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import List, Optional, Union

import pydantic as pd_v2
import pydantic.v1 as pd

from flow360 import error_messages
from flow360.cloud.rest_api import RestApi
from flow360.cloud.webbrowser import open_browser
from flow360.component.interfaces import BaseInterface
from flow360.component.utils import LocalResourceCache, is_valid_uuid, validate_type
from flow360.exceptions import Flow360RuntimeError
from flow360.log import LogLevel, log


# pylint: disable=R0801
class Flow360Status(Enum):
    """
    Flow360Status component
    """

    COMPLETED = "completed"
    ERROR = "error"
    DIVERGED = "diverged"
    UPLOADED = "uploaded"
    CASE_UPLOADED = "case_uploaded"
    UPLOADING = "uploading"
    RUNNING = "running"
    PREPROCESSING = "preprocessing"
    GENERATING = "generating"
    PROCESSED = "processed"
    STOPPED = "stopped"
    PAUSED = "paused"
    DELETED = "deleted"
    PENDING = "pending"
    UNKNOWN = "unknown"
    SUBMITTED = "submitted"
    QUEUED = "queued"

    def is_final(self):
        """Checks if status is final

        Returns
        -------
        bool
            True if status is final, False otherwise.
        """
        if self in [
            Flow360Status.COMPLETED,
            Flow360Status.DIVERGED,
            Flow360Status.ERROR,
            Flow360Status.UPLOADED,
            Flow360Status.PROCESSED,
            Flow360Status.DELETED,
        ]:
            return True
        return False


class AssetMetaBaseModel(pd.BaseModel):
    """
    Flow360 base Model
    """

    name: str = pd.Field()
    user_id: str = pd.Field(alias="userId")
    id: str = pd.Field()
    parent_id: Union[str, None] = pd.Field(alias="parentId")
    solver_version: Union[str, None] = pd.Field(alias="solverVersion")
    status: Flow360Status = pd.Field()
    tags: Optional[List[str]]
    created_at: Optional[str] = pd.Field(alias="createdAt")
    updated_at: Optional[datetime] = pd.Field(alias="updatedAt")
    updated_by: Optional[str] = pd.Field(alias="updatedBy")
    deleted: Optional[bool]
    cloud_path_prefix: Optional[str] = None

    # pylint: disable=no-self-argument
    @pd.validator("*", pre=True)
    def handle_none_str(cls, value):
        """handle None strings"""
        if value == "None":
            value = None
        return value

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config:
        extra = pd.Extra.allow
        allow_mutation = False
        allow_population_by_field_name = True


def before_submit_only(func):
    """
    Wrapper for before submit functions only
    """

    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        if obj.is_cloud_resource():
            raise Flow360RuntimeError(
                'Resource already have "id", cannot call this method. To modify and re-submit create a copy.'
            )
        return func(obj, *args, **kwargs)

    return wrapper


class AssetMetaBaseModelV2(pd_v2.BaseModel):
    """
    Flow360 base Model
    """

    name: str = pd_v2.Field()
    user_id: str = pd_v2.Field(alias="userId")
    id: str = pd_v2.Field()
    solver_version: str = pd_v2.Field(None, alias="solverVersion")
    project_id: Optional[str] = pd_v2.Field(None, alias="projectId")
    parent_id: Optional[str] = pd_v2.Field(None, alias="parentId")
    status: Flow360Status = pd_v2.Field()
    tags: List[str] = pd_v2.Field([])
    created_at: str = pd_v2.Field(alias="createdAt")
    updated_at: datetime = pd_v2.Field(alias="updatedAt")
    updated_by: Optional[str] = pd_v2.Field(None, alias="updatedBy")
    deleted: bool
    cloud_path_prefix: Optional[str] = None

    model_config = pd_v2.ConfigDict(extra="allow", frozen=True, populate_by_name=True)

    # pylint: disable=no-self-argument
    @pd_v2.field_validator("*", mode="before")
    def handle_none_str(cls, value):
        """handle None strings"""
        if value == "None":
            value = None
        return value


# pylint: disable=redefined-builtin
def local_metadata_builder(
    id,
    name,
    parent_id=None,
    cloud_path_prefix=None,
    **kwargs,
) -> dict:
    """
    Constructs a metadata dictionary for local resources.

    This function is intended to be used with `.from_local_storage()` resource constructors
    to create metadata for resources that are loaded from local storage instead of the cloud.
    It ensures that the metadata format aligns with the expected structure for Flow360 resources.

    Parameters:
    -----------
    id : str
        Unique identifier for the resource.
    name : str
        Name of the resource.
    parent_id : str
        Identifier of the parent resource.
    cloud_path_prefix : str, optional
        Cloud storage path prefix, used for mapping local resources to cloud paths if applicable.
    **kwargs:
        Additional metadata fields (and possibly overrides).

    Returns:
    --------
    dict
        A dictionary containing the structured metadata for the local resource.
    """
    meta_data = AssetMetaBaseModelV2(
        id=id,
        name=name,
        parent_id=parent_id,
        cloud_path_prefix=cloud_path_prefix,
        userId="local",
        solver_version="unknown",
        status="completed",
        createdAt="unknown",
        updatedAt=datetime.now(),
        deleted=False,
    )

    data = meta_data.model_dump()
    for k, v in kwargs.items():
        data[k] = v

    for key, val in list(data.items()):
        if isinstance(val, Enum):
            data[key] = val.value

    return data


class ResourceDraft(metaclass=ABCMeta):
    """
    Abstract base class for resources in draft state (before submission).
    """

    _id = None
    traceback = None

    def __init__(self):
        # remove from traceback:
        # 1. This line (self.traceback)
        # 2. Call of this init
        self.traceback = traceback.format_stack()[:-2]

    @property
    def id(self):
        """
        returns id of resource
        """
        return self._id

    def is_cloud_resource(self):
        """checks if resource is before submission or after

        Returns
        -------
        bool
            True if resource is cloud resources (after submission), False otherwise
        """
        if self.id is None:
            return False
        return True

    def __del__(self):
        if self.is_cloud_resource() is False and self.traceback is not None:
            print(error_messages.submit_warning(self.__class__.__name__))
            for line in self.traceback:
                print(line.strip())


class Flow360Resource(RestApi):
    """
    Flow360 base resource model
    """

    # pylint: disable=redefined-builtin
    def __init__(self, interface: BaseInterface, meta_class, id=None):
        is_valid_uuid(id, allow_none=False)
        self._resource_type = interface.resource_type
        self.s3_transfer_method = interface.s3_transfer_method
        self.meta_class = meta_class
        self._info = None
        self.logs = RemoteResourceLogs(self)
        self._local_resource_cache = LocalResourceCache()
        super().__init__(endpoint=interface.endpoint, id=id)

    def __str__(self):
        return self.info.__str__()

    def __repr__(self):
        return self.info.__repr__()

    def is_cloud_resource(self):
        """
        returns true if is cloud resource
        """
        return self.id is not None

    def _set_meta(self, meta: AssetMetaBaseModel):
        """
        set metadata info for resource
        """
        if self._info is None:
            validate_type(meta, "meta", self.meta_class)
            self._info = meta
        else:
            raise Flow360RuntimeError(
                f"Resource already have metadata {self._info}. Cannot assign."
            )

    @classmethod
    def _from_meta(cls, meta):
        raise NotImplementedError(
            "This is abstract method. Needs to be implemented by specialised class."
        )

    def get_info(self, force=False) -> AssetMetaBaseModel:
        """
        returns metadata info for resource
        """
        if self._info is None or force:
            self._info = self.meta_class(**self.get())
        return self._info

    @property
    def info(self) -> AssetMetaBaseModel:
        """
        returns metadata info for resource
        """
        return self.get_info()

    @property
    def status(self) -> Flow360Status:
        """
        returns status for resource
        """
        force = not self.info.status.is_final() and not self.info.deleted
        return self.get_info(force).status

    @property
    def id(self):
        """
        returns id of resource
        """
        return self._id

    @property
    def name(self):
        """
        returns name of resource
        """
        return self.info.name

    def wait(self, timeout_minutes=60):
        """
        Wait until the Resource finishes processing.

        While waiting, an animated dot sequence is displayed using the current non-final status value.
        The status is dynamically updated every few seconds with an increasing number of dots:
        ⠇ running..............................
        This implementation leverages Rich's `status()` method via our custom logger (log.status) to perform in-place
        status updates. If the process does not finish within the specified timeout, a TimeoutError is raised.
        """
        max_dots = 30
        update_every_seconds = 2
        start_time = time.time()

        with log.status() as status_logger:
            while not self.status.is_final():

                elapsed = time.time() - start_time
                dot_count = int((elapsed // update_every_seconds) % max_dots)
                status_logger.update(f"{self.status.value}{'.' * dot_count}")

                if time.time() - start_time > timeout_minutes * 60:
                    raise TimeoutError(
                        "Timeout: Process did not finish within the specified timeout period"
                    )

                time.sleep(update_every_seconds)

    def short_description(self) -> str:
        """short_description

        Returns
        -------
        str
            generates short description of resource (type, name, id, status)
        """
        return f"""
        type   = {self._resource_type}
        name   = {self.name}
        id     = {self.id}
        status = {self.status.value}
        """

    @property
    def solver_version(self):
        """
        returns solver version of resource
        """
        return self.info.solver_version

    def get_download_file_list(self) -> List:
        """return list of files available for download

        Returns
        -------
        List
            List of files available for download
        """
        return self.get(method="files")

    def get_cloud_path_prefix(self):
        """
        Retrieves the cloud path prefix for the resource.

        If the `cloud_path_prefix` is already available in the resource's metadata, it is returned directly.
        Otherwise, this method determines the prefix based on the associated files of the resource.

        Returns
        -------
        str
            The cloud path prefix for the resource.

        Raises
        ------
        ValueError
            If no files are associated with the resource, making it impossible to determine the cloud path prefix.
        """
        if self.info.cloud_path_prefix is not None:
            return self.info.cloud_path_prefix
        files = self.get_download_file_list()
        if len(files) == 0:
            raise ValueError(
                "Cannot determine cloud path prefix. Not files associated with this resource."
            )
        return self.s3_transfer_method.get_cloud_path_prefix(self.id, files[0]["fileName"])

    @property
    def local_resource_cache(self):
        """local resource cache"""
        return self._local_resource_cache

    # pylint: disable=too-many-arguments
    def _download_file(
        self,
        file_name,
        to_file=None,
        to_folder=".",
        overwrite: bool = True,
        progress_callback=None,
        **kwargs,
    ):
        """
        Download a specific file associated with the resource.

        Parameters
        ----------
        file_name : str
            Name of the file to be downloaded.
        to_file : str, optional
            File name to save the downloaded file. If None, the file will be saved in the current directory.
            If provided without an extension, the extension will be automatically added based on the file type.
        to_folder : str, optional
            Folder name to save the downloaded file. If None, the file will be saved in the current directory.
        overwrite : bool, optional
            If True, overwrite existing files with the same name in the destination.
        progress_callback : callable, optional
            A callback function to track the download progress.
        **kwargs : dict, optional
            Additional arguments to be passed to the download process.

        Returns
        -------
        str
            File path of the downloaded file.
        """

        return self.s3_transfer_method.download_file(
            self.id,
            file_name,
            to_file=to_file,
            to_folder=to_folder,
            overwrite=overwrite,
            progress_callback=progress_callback,
            **kwargs,
        )

    def _parse_json_from_cloud(self, filename) -> dict:
        """
        Return dictionary from cloud JSON file.
        """
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            temp_path = tmp_file.name
        try:
            self._download_file(filename, to_file=temp_path, log_error=False, verbose=False)
            with open(temp_path, "r", encoding="utf-8") as fh:
                data_as_dict = json.load(fh)
            return data_as_dict

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _upload_file(self, remote_file_name: str, file_name: str, progress_callback=None):
        """
        general upload functionality
        """
        self.s3_transfer_method.upload_file(
            self.id, remote_file_name, file_name, progress_callback=progress_callback
        )

    def create_multipart_upload(self, remote_file_name: str):
        """
        Creates a multipart upload for the specified remote file name and file.

        Args:
            remote_file_name (str): The name of the remote file.

        Returns:
            UploadID
        """
        return self.s3_transfer_method.create_multipart_upload(self.id, remote_file_name)

    def upload_part(
        self,
        remote_file_name: str,
        upload_id: str,
        part_number: int,
        compressed_chunk,
    ):
        """
        Uploads a part of the file as part of a multipart upload.

        Args:
            remote_file_name (str): The name of the remote file.
            upload_id (str): The ID of the multipart upload.
            part_number (int): The part number of the upload.
            compressed_chunk: The compressed chunk data to upload.

        Returns:
            {"ETag": response["ETag"], "PartNumber": part_number}
        """
        return self.s3_transfer_method.upload_part(
            self.id, remote_file_name, upload_id, part_number, compressed_chunk
        )

    def complete_multipart_upload(
        self, remote_file_name: str, upload_id: str, uploaded_parts: dict
    ):
        """
        Completes a multipart upload for the specified remote file name and upload ID.

        Args:
            remote_file_name (str): The name of the remote file.
            upload_id (str): The ID of the multipart upload.
            uploaded_parts (dict): A dictionary containing information about the uploaded parts.
                The dictionary should have the following structure:
                {
                    "ETag": "string",       # The ETag of each uploaded part.
                    "part_number": int      # The part number of each uploaded part.
                }

        Returns:
            None
        """
        self.s3_transfer_method.complete_multipart_upload(
            self.id, remote_file_name, upload_id, uploaded_parts
        )

    # pylint: disable=no-member
    def open_in_browser(self):
        """
        Open resource in browser
        """
        open_browser(f"{self._interface().endpoint}/{self.id}")

    def _complete_upload(self):
        """
        [Simulation V2 API] Inform the server that the upload is complete. This kicks off the following pipeline.
        """
        self.patch({"action": "Success"}, method="files")


def is_object_cloud_resource(resource: Flow360Resource):
    """
    checks if object is cloud resource, raises RuntimeError
    """
    if resource is not None:
        if not resource.is_cloud_resource():
            raise Flow360RuntimeError(error_messages.not_a_cloud_resource)
        return True
    return False


class Position(Enum):
    """
    Enumeration class for log file positions.

    Available positions:
    - HEAD: Specifies the head position, representing the first lines of the log file.
    - TAIL: Specifies the tail position, representing the last lines of the log file.
    - ALL: Specifies the all position, representing the entire log file.
    """

    HEAD = "head"
    TAIL = "tail"
    ALL = "all"


class Flow360ResourceListBase(list, RestApi):
    """
    Flow360 ResourceList base component
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        ancestor_id: str = None,
        from_cloud: bool = True,
        include_deleted: bool = False,
        limit: int = 100,
        resourceClass: Flow360Resource = None,
    ):
        if from_cloud:
            endpoint = resourceClass._interface().endpoint
            if limit is not None and not include_deleted:
                endpoint += "/page"

            RestApi.__init__(self, endpoint=endpoint)

            params = {"includeDeleted": include_deleted, "limit": limit, "start": 0}
            if ancestor_id is not None:
                params[resourceClass._params_ancestor_id_name()] = ancestor_id

            resp = self.get(params=params)

            if isinstance(resp, dict):
                resp = resp["data"]

            if limit is None:
                limit = -1

            list.__init__(
                self,
                [
                    resourceClass._from_meta(meta=resourceClass._meta_class().parse_obj(item))
                    for item in resp[:limit]
                ],
            )

    @classmethod
    def from_cloud(cls):
        """
        get ResourceList from cloud
        """
        return cls(from_cloud=True)


class TemporaryLogDirectory:
    """
    A class representing a temporary log directory.

    This class creates a temporary directory for storing log files and provides methods for managing the directory.

    Attributes:
        _dir (TemporaryDirectory): A temporary directory object created using the `tempfile.TemporaryDirectory` class.

    Properties:
        name (str): The name of the temporary directory.

    Methods:
        delete_file(): Deletes the temporary directory if it exists.

    """

    # pylint: disable=consider-using-with
    def __init__(self):
        """
        Initializes a new instance of the TemporaryLogDirectory class.

        Upon initialization, a temporary directory is created using the `tempfile.TemporaryDirectory` class.

        """
        self._dir = TemporaryDirectory()

    @property
    def name(self):
        """
        str: Returns the name of the temporary directory.
        """
        return self._dir.name

    def delete_file(self):
        """
        Deletes the temporary directory if it exists.

        This method checks if the temporary directory exists and deletes it if it does.
        """
        if os.path.exists(self.name):
            self._dir.cleanup()

    def __del__(self):
        """
        Destructor method for the TemporaryLogDirectory class.

        When the instance of the class is destroyed, this method is automatically
        called to clean up the temporary directory.
        """
        self.delete_file()


class RemoteResourceLogs:
    """
    Logs class for getting remote logs from flow360 resources
    """

    def __init__(self, flow360_resource: Flow360Resource):
        self.flow360_resource = flow360_resource
        self._tmp_file_name = None
        self._tmp_dir = None
        self._remote_file_name = None

    def _get_log_file_names(self) -> List[str]:
        file_names = [
            file["fileName"]
            for file in self.flow360_resource.get_download_file_list()
            if "fileName" in file
        ]
        pattern = re.compile(r"logs/(.*\.log)")
        log_file_names = [
            pattern.search(string).group(1) for string in file_names if pattern.search(string)
        ]
        return log_file_names

    def set_remote_log_file_name(self, file_name: str):
        """
        Set the name of the remote log file.

        This method sets the name of the log file that is stored remotely.

        Args:
            file_name (str): The name of the remote log file.

        Returns:
            None
        """
        self._remote_file_name = file_name

    def _has_multiple_files(self) -> bool:
        logs = self._get_log_file_names()
        if len(logs) > 1:
            log.warning(
                f"There are multiple log files: {logs}. \n The default file will be {logs[0]}."
                "Call set_remote_log_file_name(file_name) to override the default log file."
            )
            return True
        return False

    def _get_tmp_file_name(self):
        if self._tmp_file_name is None:
            if self._tmp_dir is None:
                self._tmp_dir = TemporaryLogDirectory()
            if self._remote_file_name is None:
                self._remote_file_name = self._get_log_file_names()[0]
            self._tmp_file_name = os.path.join(self._tmp_dir.name, self._remote_file_name)
        return self._tmp_file_name

    # pylint: disable=protected-access
    def _refresh_file(self):
        tmp_file = self._get_tmp_file_name()
        self.flow360_resource._download_file(self._remote_file_name, tmp_file, overwrite=True)

    # pylint: disable=protected-access
    @property
    def _cached_file(self):
        tmp_file = self._get_tmp_file_name()
        self.flow360_resource._download_file(self._remote_file_name, tmp_file, overwrite=False)
        return tmp_file

    def _get_log_by_pos(self, pos: Position = None, num_lines: int = 100):
        """
        Get log lines based on position (head, tail, all).

        :param pos: Position enum (HEAD, TAIL, or ALL).
        :param num_lines: Number of lines to retrieve (for HEAD and TAIL positions).
        :return: List of log lines.
        """
        try:
            with open(self._cached_file, encoding="utf-8") as file:
                lines = file.read().splitlines()
                if pos == Position.HEAD:
                    return lines[:num_lines]
                if pos == Position.TAIL:
                    return lines[-num_lines:]
                return lines

        except OSError as error:
            log.error("invalid path to log files", error)
            return None

    def _get_log_by_level(self, level: LogLevel = None):
        """
        Get log lines filtered by log level.

        :param level: Log level (ERROR, WARNING, INFO, or None for all).
        :return: List of filtered log lines.
        """
        try:
            with open(self._cached_file, encoding="utf-8") as file:
                log_contents = file.read()
                if level == "ERROR":
                    filt = r"(?mi)^(.*?error.*)$"
                elif level == "WARNING":
                    filt = r"(?mi)^.*?(?:error|warning).+$"
                elif level == "INFO":
                    filt = r"(?mi)^(?!.*USERDBG)(?=.*\S).*$"
                else:
                    filt = r".*"
                return [line for line in re.findall(filt, log_contents) if line.strip() != ""]
        except OSError as error:
            log.error("invalid path to log files", error)
            return None

    def head(self, num_lines: int = 100):
        """
        Print the first n lines of the log file.

        :param num_lines: Number of lines to print.
        """
        log_message = self._get_log_by_pos(Position.HEAD, num_lines)
        print("\n".join(log_message))

    def tail(self, num_lines: int = 100):
        """
        Print the last n lines of the log file.

        :param num_lines: Number of lines to print.
        """
        log_message = self._get_log_by_pos(Position.TAIL, num_lines)
        print("\n".join(log_message))

    def print(self):
        """
        Print the entire log file.
        """
        log_message = self._get_log_by_pos(Position.ALL)
        print("\n".join(log_message))

    def errors(self):
        """
        Print log lines containing error messages.
        """
        log_message = self._get_log_by_level("ERROR")
        print("\n".join(log_message))

    def warnings(self):
        """
        Print log lines containing warning messages.
        """
        log_message = self._get_log_by_level("WARNING")
        print("\n".join(log_message))

    def info(self):
        """
        Print log lines containing info messages.
        """
        log_message = self._get_log_by_level("INFO")
        print("\n".join(log_message))

    def to_file(self, file_name: str):
        """
        Write log lines to a file.

        :param file_name: File name or path to write the log lines.
        """
        try:
            shutil.copyfile(self._cached_file, file_name)
        except shutil.SameFileError as error:
            log.error(f"Write to a different file location, {error}")
        except OSError as error:
            log.error(f"{file_name} not writable {error}")
