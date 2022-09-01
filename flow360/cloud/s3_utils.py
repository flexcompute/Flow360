"""
s3 util file for material uploading and downloading.
"""
import os
import urllib
from datetime import datetime
from enum import Enum

import boto3
from boto3.s3.transfer import TransferConfig
from pydantic import BaseModel, Field
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from flow360.cloud.http_util import http
from flow360.environment import Env

_s3_config = TransferConfig(
    multipart_threshold=1024 * 25,
    max_concurrency=50,
    multipart_chunksize=1024 * 25,
    use_threads=True,
)


def create_base_folder(path: str, target_name: str, to_file: str = ".", keep_folder: bool = True):
    """
    :param path: source id
    :param target_name: path to file on cloud, same value as key for s3 client download.
    :param to_file: could be either folder or file name.
    :param keep_folder: If true, the downloaded file will
    be put in the same folder as the file on cloud. Only work
    when file_name is a folder name.
    :return:
    """
    if os.path.isdir(to_file):
        to_file = (
            os.path.join(to_file, path, target_name)
            if keep_folder
            else os.path.join(to_file, os.path.basename(target_name))
        )

    os.makedirs(os.path.dirname(to_file), exist_ok=True)
    return to_file


class _S3Action(Enum):
    """
    Enum for s3 action
    """

    UPLOADING = "↑"
    DOWNLOADING = "↓"


def _get_progress(action: _S3Action):
    col = (
        TextColumn(f"[bold green]{_S3Action.DOWNLOADING.value}")
        if action == _S3Action.DOWNLOADING
        else TextColumn(f"[bold red]{_S3Action.UPLOADING.value}")
    )
    return Progress(
        col,
        TextColumn("[bold blue]{task.fields[filename]}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )


class _UserCredential(BaseModel):
    access_key_id: str = Field(alias="accessKeyId")
    expiration: datetime
    secret_access_key: str = Field(alias="secretAccessKey")
    session_token: str = Field(alias="sessionToken")


class _S3STSToken(BaseModel):
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
        return boto3.client(
            "s3",
            region_name=Env.current.aws_region,
            aws_access_key_id=self.user_credential.access_key_id,
            aws_secret_access_key=self.user_credential.secret_access_key,
            aws_session_token=self.user_credential.session_token,
        )

    def is_expired(self):
        """
        Check if the token is expired.
        :return:
        """
        return (
            self.user_credential.expiration
            - datetime.now(tz=self.user_credential.expiration.tzinfo)
        ).total_seconds() > 300


class S3TransferType(Enum):
    """
    Enum for s3 transfer type
    """

    VOLUME_MESH = "VolumeMesh"
    SURFACE_MESH = "SurfaceMesh"
    CASE = "Case"
    STUDIO = "Studio"

    def _get_grant_url(self, resource_id, file_name: str) -> str:
        """
        Get the grant url for a file.
        :param resource_id:
        :param file_name:
        :return:
        """
        if self is S3TransferType.VOLUME_MESH:
            return f"volumemeshes/{resource_id}/file?filename={file_name}"
        if self is S3TransferType.SURFACE_MESH:
            return f"surfacemeshes/{resource_id}/file?filename={file_name}"
        if self is S3TransferType.CASE:
            return f"cases/{resource_id}/file?filename={file_name}"
        if self is S3TransferType.STUDIO:
            return f"volumemeshes/{resource_id}/file?filename={file_name}"

        return None

    def upload_file(self, resource_id: str, remote_file_name: str, file_name: str):
        """
        Upload a file to s3.
        :param resource_id:
        :param remote_file_name:
        :param file_name:
        :return:
        """
        if not os.path.exists(file_name):
            print(f"mesh file {file_name} does not Exist!")
            raise FileNotFoundError()
        with _get_progress(_S3Action.UPLOADING) as progress:
            task_id = progress.add_task(
                "upload", filename=os.path.basename(file_name), total=os.path.getsize(file_name)
            )

            def _call_back(bytes_in_chunk):
                progress.update(task_id, advance=bytes_in_chunk)

            token = self._get_s3_sts_token(resource_id, remote_file_name)
            token.get_client().upload_file(
                Bucket=token.get_bucket(),
                Filename=file_name,
                Key=token.get_s3_key(),
                Callback=_call_back,
                Config=_s3_config,
            )

    def download_file(
        self, resource_id: str, remote_file_name: str, to_file: str, keep_folder: bool = True
    ):
        """
        Download a file from s3.
        :param resource_id:
        :param remote_file_name: file name with path in s3
        :param to_file: local file name or local folder name.
        :param keep_folder: If true, the downloaded file will be put
        in the same folder as the file on cloud. Only works when to_file is a folder name.
        :return:
        """
        to_file = create_base_folder(resource_id, remote_file_name, to_file, keep_folder)
        token = self._get_s3_sts_token(resource_id, remote_file_name)
        client = token.get_client()
        meta_data = client.head_object(Bucket=token.get_bucket(), Key=token.get_s3_key())
        with _get_progress(_S3Action.DOWNLOADING) as progress:
            progress.start()
            task_id = progress.add_task(
                "download",
                filename=os.path.basename(remote_file_name),
                total=meta_data.get("ContentLength", 0),
            )

            def _call_back(bytes_in_chunk):
                progress.update(task_id, advance=bytes_in_chunk)

            client.download_file(
                Bucket=token.get_bucket(),
                Filename=to_file,
                Key=token.get_s3_key(),
                Callback=_call_back,
            )

    def _get_s3_sts_token(self, resource_id: str, file_name: str) -> _S3STSToken:
        session_key = f"{resource_id}:{self.value}:{file_name}"
        if session_key not in _s3_sts_tokens or _s3_sts_tokens[session_key].is_expired():
            path = self._get_grant_url(resource_id, file_name)
            resp = http.get(path)
            token = _S3STSToken.parse_obj(resp)
            _s3_sts_tokens[session_key] = token
        return _s3_sts_tokens[session_key]


_s3_sts_tokens: [str, _S3STSToken] = {}
