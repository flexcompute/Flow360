"""
Flow360 base Model
"""
import traceback
from abc import ABC
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import List, Optional, Union

import pydantic as pd

from .. import error_messages
from ..cloud.rest_api import RestApi
from ..exceptions import RuntimeError as FlRuntimeError
from ..log import log
from ..user_config import UserConfig
from .utils import is_valid_uuid, validate_type


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
    DELETED = "deleted"
    PENDING = "pending"
    UNKNOWN = "unknown"

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


class Flow360ResourceBaseModel(pd.BaseModel):
    """
    Flow360 base Model
    """

    name: str = pd.Field()
    user_id: str = pd.Field(alias="userId")
    id: str = pd.Field()
    solver_version: Union[str, None] = pd.Field(alias="solverVersion")
    status: Flow360Status = pd.Field()
    tags: Optional[List[str]]
    created_at: Optional[str] = pd.Field(alias="createdAt")
    updated_at: Optional[datetime] = pd.Field(alias="updatedAt")
    updated_by: Optional[str] = pd.Field(alias="updatedBy")
    deleted: bool

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


def on_cloud_resource_only(func):
    """
    Wrapper for cloud functions only
    """

    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        if not obj.is_cloud_resource():
            raise FlRuntimeError(
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
            raise FlRuntimeError(
                'Resource already have "id", cannot call this method. To modify and re-submit create a copy.'
            )
        return func(obj, *args, **kwargs)

    return wrapper


class ResourceDraft(ABC):
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
        if not UserConfig.is_suppress_submit_warning():
            log.warning(error_messages.submit_reminder(self.__class__.__name__))

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
    def __init__(
        self, resource_type, info_type_class, *args, s3_transfer_method=None, id=None, **kwargs
    ):
        is_valid_uuid(id, ignore_none=False)
        self._resource_type = resource_type
        self.s3_transfer_method = s3_transfer_method
        self.info_type_class = info_type_class
        self._info = None
        super().__init__(*args, id=id, **kwargs)

    def __str__(self):
        return self.info.__str__()

    def __repr__(self):
        return self.info.__repr__()

    def is_cloud_resource(self):
        """
        returns true if is cloud resource
        """
        return self.id is not None

    def _set_meta(self, meta: Flow360ResourceBaseModel):
        """
        set metadata info for resource
        """
        if self._info is None:
            validate_type(meta, "meta", self.info_type_class)
            self._info = meta
        else:
            raise FlRuntimeError(f"Resource already have metadata {self._info}. Cannot assign.")

    @classmethod
    def _from_meta(cls, meta):
        raise NotImplementedError(
            "This is abstract method. Needs to be implemented by specialised class."
        )

    @on_cloud_resource_only
    def get_info(self, force=False) -> Flow360ResourceBaseModel:
        """
        returns metadata info for resource
        """
        if self._info is None or force:
            self._info = self.info_type_class(**self.get())
        return self._info

    @property
    def info(self) -> Flow360ResourceBaseModel:
        """
        returns metadata info for resource
        """
        return self.get_info()

    @property
    @on_cloud_resource_only
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
    @on_cloud_resource_only
    def name(self):
        """
        returns name of resource
        """
        return self.info.name

    def short_description(self) -> str:
        """short_description

        Returns
        -------
        str
            generates short description of resource (name, id, status)
        """
        return f"""
        name   = {self.name}
        id     = {self.id}
        status = {self.status.value}
        """

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
        **kwargs,
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
            **kwargs,
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
    if resource is not None:
        if not resource.is_cloud_resource():
            raise FlRuntimeError(error_messages.not_a_cloud_resource)
        return True
    return False


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
            endpoint = resourceClass._endpoint
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
