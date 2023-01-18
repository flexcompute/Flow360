"""
Flow360 base Model
"""
from datetime import datetime
from typing import List, Optional, Union
from functools import wraps

from pydantic import BaseModel, Extra, Field

from ..cloud.rest_api import RestApi


class Flow360BaseModel(BaseModel, extra=Extra.allow, allow_mutation=False):
    """
    Flow360 base Model
    """

    name: str
    user_id: str = Field(alias="userId")
    solver_version: Union[str, None] = Field(alias="solverVersion")
    status: str
    tags: Optional[List[str]]
    created_at: Optional[str] = Field(alias="createdAt")
    updated_at: Optional[datetime] = Field(alias="updatedAt")
    updated_by: Optional[str] = Field(alias="updatedBy")


def on_cloud_resource_only(func):
    """
    Wrapper for cloud functions only
    """

    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        if not obj.is_cloud_resource():
            raise RuntimeError(
                'Resource does not have "id", it is not a cloud resource. Provide "id" before calling this function.'
            )
        return func(obj, *args, **kwargs)

    return wrapper


def before_submit_only(func):
    """
    Wrapper for before submit functions only
    """

    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        if obj.is_cloud_resource():
            raise RuntimeError(
                'Resource already have "id", cannot call this method. To modify and re-submit create a copy.'
            )
        return func(obj, *args, **kwargs)

    return wrapper


class Flow360Resource(RestApi):
    """
    Flow360 base resource model
    """

    def __init__(self, resource_type, info_type_class, *args, s3_transfer_method=None, **kwargs):
        self._resource_type = resource_type
        self.s3_transfer_method = s3_transfer_method
        self.info_type_class = info_type_class
        super().__init__(*args, **kwargs)

    def __str__(self):
        if self._info is not None:
            return self._info.__str__()
        return f"{self._resource_type} is not yet submitted."

    def is_cloud_resource(self):
        """
        returns true if is cloud resource
        """
        return self.id is not None

    @on_cloud_resource_only
    def get_info(self, force=False):
        """
        returns metadata info for resource
        """
        if self._info is None or force:
            self._info = self.info_type_class(**self.get())
        return self._info

    @property
    def info(self):
        """
        returns metadata info for resource
        """
        return self.get_info()

    @property
    @on_cloud_resource_only
    def status(self):
        """
        returns status for resource
        """
        return self.get_info(True).status

    @property
    def id(self):
        """
        returns id of resource
        """
        return self._id

    @property
    @on_cloud_resource_only
    def name(self):
        """
        returns name of resource
        """
        return self.info.name

    @property
    @on_cloud_resource_only
    def solver_version(self):
        """
        returns solver version of resource
        """
        return self.info.solver_version

    # pylint: disable=too-many-arguments
    @on_cloud_resource_only
    def download_file(
        self,
        file_name,
        to_file=".",
        keep_folder: bool = True,
        overwrite: bool = True,
        progress_callback=None,
    ):
        """
        general download functionality
        """
        return self.s3_transfer_method.download_file(
            self.id,
            file_name,
            to_file,
            keep_folder,
            overwrite=overwrite,
            progress_callback=progress_callback,
        )

    @on_cloud_resource_only
    def upload_file(self, remote_file_name: str, file_name: str, progress_callback=None):
        """
        general upload functionality
        """
        self.s3_transfer_method.upload_file(
            self.id, remote_file_name, file_name, progress_callback=progress_callback
        )


def is_object_cloud_resource(resource: Flow360Resource):
    """
    checks if object is cloud resource, raises RuntimeError
    """
    msg = "Reference resource is not a cloud resource. "
    msg += "If a case was retried or forked from other case, submit the other case first before submitting this case."
    if resource is not None:
        if not resource.is_cloud_resource():
            raise RuntimeError(msg)
        return True
    return False
