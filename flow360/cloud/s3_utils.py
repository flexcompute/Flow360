"""
s3 util file for material uploading and downloading.
"""

import math
import os
import urllib
from abc import ABCMeta, abstractmethod
from datetime import datetime
from enum import Enum

import boto3
from boto3.s3.transfer import TransferConfig

# pylint: disable=unused-import
from botocore.exceptions import ClientError as CloudFileNotFoundError
from pydantic.v1 import BaseModel, Field

from ..environment import Env
from ..exceptions import Flow360ValueError
from ..log import log
from .http_util import http
from .utils import _get_progress, _S3Action


class ProgressCallbackInterface(metaclass=ABCMeta):
    """
    Progress callback abstract class
    """

    @abstractmethod
    def total(self, total: int):
        """
        total bytes to transfer
        """

    @abstractmethod
    def __call__(self, bytes_chunk_transferred):
        pass


def _get_dynamic_upload_config(file_size) -> TransferConfig:
    # pylint: disable=invalid-name
    # Constant definition: https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html
    MIN_CHUNK_SIZE = 5 * 1024 * 1024
    MAX_PART_COUNT = 100000

    return TransferConfig(
        max_concurrency=50,
        multipart_chunksize=max(int(MIN_CHUNK_SIZE), math.ceil(file_size / MAX_PART_COUNT)),
        use_threads=True,
    )


def get_local_filename_and_create_folders(
    target_name: str, to_file: str = None, to_folder: str = "."
):
    """
    Create a base folder and return the target file path for downloading cloud data.

    Parameters
    ----------

    target_name : str
        The file path on the cloud, same value as the key for S3 client download.

    to_file : str, optional
        The destination file path where the downloaded file will be saved. If None, the current directory
        will be used with target_name as file name.

    to_folder : str, optional
        The folder name to save the downloaded file. If provided, the downloaded file will be saved inside this folder.

    Returns
    -------
    str
        The target file path for downloading cloud data.

    """

    if to_file is not None and to_folder != ".":
        raise Flow360ValueError(
            "Only one of 'to_file' or 'to_folder' should be provided, not both."
        )

    if to_file is not None and os.path.isdir(to_file):
        raise Flow360ValueError(
            "to_file should be a file name, not directory, use to_folder instead"
        )

    if to_file is None:
        to_file = os.path.basename(target_name)

    if to_folder != ".":
        to_file = os.path.join(to_folder, os.path.basename(to_file))

    _, file_ext = os.path.splitext(target_name)
    _, to_file_ext = os.path.splitext(to_file)
    if to_file_ext != file_ext:
        to_file = to_file + file_ext

    dirname = os.path.dirname(to_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    return to_file


class _UserCredential(BaseModel):
    access_key_id: str = Field(alias="accessKeyId")
    expiration: datetime
    secret_access_key: str = Field(alias="secretAccessKey")
    session_token: str = Field(alias="sessionToken")
    region: str


class _S3STSToken(BaseModel):
    cloud_path_prefix: str = Field(alias="cloudpathPrefix")
    cloud_path: str = Field(alias="cloudpath")
    user_credential: _UserCredential = Field(alias="userCredentials")

    def get_bucket(self):
        """
        Get bucket name.
        :return:
        """
        path = urllib.parse.urlparse(self.cloud_path)
        return path.netloc

    def get_s3_key(self):
        """
        Get s3 key.
        :return:
        """
        path = urllib.parse.urlparse(self.cloud_path)
        return path.path[1:]

    def get_client(self):
        """
        Get s3 client.
        :return:
        """
        # pylint: disable=no-member
        kwargs = {
            "region_name": self.user_credential.region,
            "aws_access_key_id": self.user_credential.access_key_id,
            "aws_secret_access_key": self.user_credential.secret_access_key,
            "aws_session_token": self.user_credential.session_token,
        }

        if Env.current.s3_endpoint_url is not None:
            kwargs["endpoint_url"] = Env.current.s3_endpoint_url

        return boto3.client("s3", **kwargs)

    def is_expired(self):
        """
        Check if the token is expired.
        :return:
        """
        # pylint: disable=no-member
        return (
            self.user_credential.expiration
            - datetime.now(tz=self.user_credential.expiration.tzinfo)
        ).total_seconds() < 300


class S3TransferType(Enum):
    """
    Enum for s3 transfer type
    """

    GEOMETRY = "Geometry"
    VOLUME_MESH = "VolumeMesh"
    SURFACE_MESH = "SurfaceMesh"
    CASE = "Case"
    REPORT = "Report"

    def _get_grant_url(self, resource_id, file_name: str) -> str:
        """
        Get the grant url for a file.
        :param resource_id:
        :param file_name:
        :return:
        """
        if self is S3TransferType.VOLUME_MESH:
            return f"v2/volume-meshes/{resource_id}/file?filename={file_name}"
        if self is S3TransferType.SURFACE_MESH:
            return f"v2/surface-meshes/{resource_id}/file?filename={file_name}"
        if self is S3TransferType.CASE:
            return f"v2/cases/{resource_id}/file?filename={file_name}"
        if self is S3TransferType.GEOMETRY:
            return f"v2/geometries/{resource_id}/file?filename={file_name}"
        if self is S3TransferType.REPORT:
            return f"v2/report/{resource_id}/file?filename={file_name}"

        raise Flow360ValueError(f"unknown download method for {self}")

    def get_cloud_path_prefix(self, resource_id, file_name):
        """
        returns path prefix (without resource_id) to corresponding bucket
        """
        token = self._get_s3_sts_token(resource_id, file_name)
        base_path = token.cloud_path_prefix.rsplit("/", 1)[0]
        return base_path

    def create_multipart_upload(
        self,
        resource_id: str,
        remote_file_name: str,
    ):
        """
        Creates a multipart upload for the specified resource ID and remote file name.

        Args:
            resource_id (str): The ID of the resource.
            remote_file_name (str): The name of the remote file.

        Returns:
            str: The upload ID of the multipart upload.
        """
        token = self._get_s3_sts_token(resource_id, remote_file_name)
        client = token.get_client()
        return client.create_multipart_upload(
            Bucket=token.get_bucket(),
            Key=token.get_s3_key(),
        )["UploadId"]

    # pylint: disable=too-many-arguments
    def upload_part(
        self,
        resource_id: str,
        remote_file_name: str,
        upload_id: str,
        part_number: int,
        compressed_chunk,
    ):
        """
        Uploads a part of the file as part of a multipart upload.

        Args:
            resource_id (str): The ID of the resource.
            remote_file_name (str): The name of the remote file.
            upload_id (str): The ID of the multipart upload.
            part_number (int): The part number of the upload.
            compressed_chunk: The compressed chunk data to upload.

        Returns:
            dict: A dictionary containing the e_tag and part_number of the uploaded part.
        """
        token = self._get_s3_sts_token(resource_id, remote_file_name)
        client = token.get_client()
        response = client.upload_part(
            Bucket=token.get_bucket(),
            Key=token.get_s3_key(),
            PartNumber=part_number,
            UploadId=upload_id,
            Body=compressed_chunk,
        )

        # Return the e_tag of the uploaded part
        return {"ETag": response["ETag"], "PartNumber": part_number}

    def complete_multipart_upload(
        self, resource_id: str, remote_file_name: str, upload_id: str, uploaded_parts: dict
    ):
        """
        Completes a multipart upload for the specified resource ID, remote file name, upload ID, e_tag, and part number.

        Args:
            resource_id (str): The ID of the resource.
            remote_file_name (str): The name of the remote file.
            upload_id (str): The ID of the multipart upload.
            e_tag (str): The e_tag of the uploaded part.
            part_number (int): The part number of the completed upload.

        Returns:
            None
        """
        token = self._get_s3_sts_token(resource_id, remote_file_name)
        client = token.get_client()
        client.complete_multipart_upload(
            Bucket=token.get_bucket(),
            Key=token.get_s3_key(),
            UploadId=upload_id,
            MultipartUpload={"Parts": uploaded_parts},
        )

    def upload_file(
        self, resource_id: str, remote_file_name: str, file_name: str, progress_callback=None
    ):
        """
        Upload a file to s3.
        :param resource_id:
        :param remote_file_name:
        :param file_name:
        :return:
        """
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"file {file_name} does not Exist!")

        token = self._get_s3_sts_token(resource_id, remote_file_name)
        client = token.get_client()
        if progress_callback:
            progress_callback.total = float(os.path.getsize(file_name))
            client.upload_file(
                Bucket=token.get_bucket(),
                Filename=file_name,
                Key=token.get_s3_key(),
                Callback=progress_callback,
                Config=_get_dynamic_upload_config(os.path.getsize(file_name)),
            )
        else:
            with _get_progress(_S3Action.UPLOADING) as progress:
                task_id = progress.add_task(
                    "upload", filename=os.path.basename(file_name), total=os.path.getsize(file_name)
                )

                def _call_back(bytes_in_chunk):
                    progress.update(task_id, advance=bytes_in_chunk)

                token = self._get_s3_sts_token(resource_id, remote_file_name)
                client.upload_file(
                    Bucket=token.get_bucket(),
                    Filename=file_name,
                    Key=token.get_s3_key(),
                    Callback=_call_back,
                    Config=_get_dynamic_upload_config(os.path.getsize(file_name)),
                )

    # pylint: disable=too-many-arguments
    def download_file(
        self,
        resource_id: str,
        remote_file_name: str,
        to_file: str = None,
        to_folder: str = ".",
        overwrite: bool = True,
        progress_callback=None,
        log_error=True,
        verbose=True,
    ):
        """
        Download a file from s3.
        :param resource_id:
        :param remote_file_name: file name with path in s3
        :param to_file: local file name or local folder name.
        in the same folder as the file on cloud. Only works when to_file is a folder name.
        :param overwrite: if True overwrite if file exists, otherwise don't download
        :param progress_callback: provide custom callback for progress
        :return:
        """

        to_file = get_local_filename_and_create_folders(remote_file_name, to_file, to_folder)
        if os.path.exists(to_file) and not overwrite:
            log.info(f"Skipping {remote_file_name}, file exists.")
            return to_file

        token = self._get_s3_sts_token(resource_id, remote_file_name)
        client = token.get_client()
        try:
            meta_data = client.head_object(Bucket=token.get_bucket(), Key=token.get_s3_key())
        except CloudFileNotFoundError:
            if log_error:
                log.error(f"{remote_file_name} not found. id={resource_id}")
            raise

        if progress_callback:
            progress_callback.total = meta_data.get("ContentLength", 0)
            client.download_file(
                Bucket=token.get_bucket(),
                Filename=to_file,
                Key=token.get_s3_key(),
                Callback=progress_callback,
            )
        else:
            with _get_progress(_S3Action.DOWNLOADING) as progress:
                if verbose is True:
                    progress.start()
                    task_id = progress.add_task(
                        "download",
                        filename=os.path.basename(remote_file_name),
                        total=meta_data.get("ContentLength", 0),
                    )

                    def _call_back(bytes_in_chunk):
                        progress.update(task_id, advance=bytes_in_chunk)

                else:
                    _call_back = None

                client.download_file(
                    Bucket=token.get_bucket(),
                    Filename=to_file,
                    Key=token.get_s3_key(),
                    Callback=_call_back,
                )
        if verbose is True:
            log.info(f"Saved to {to_file}")
        return to_file

    def _get_s3_sts_token(self, resource_id: str, file_name: str) -> _S3STSToken:
        session_key = f"{resource_id}:{self.value}"
        if session_key not in _s3_sts_tokens or _s3_sts_tokens[session_key].is_expired():
            path = self._get_grant_url(resource_id, file_name)
            resp = http.get(path)
            token = _S3STSToken.parse_obj(resp)
            _s3_sts_tokens[session_key] = token
            return token
        token = _s3_sts_tokens[session_key]
        return token.copy(
            deep=True, update={"cloud_path": token.cloud_path_prefix + "/" + file_name}
        )


_s3_sts_tokens: dict[str, _S3STSToken] = {}
