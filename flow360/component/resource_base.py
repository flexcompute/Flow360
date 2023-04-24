"""
Flow360 base Model
"""
import traceback
from abc import ABC
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import List, Optional, Union

from pydantic import BaseModel, Extra, Field

from ..cloud.rest_api import RestApi
from ..log import log
from ..user_config import UserConfig


class Flow360Status(Enum):
    """
    Flow360Status component
    """

    COMPLETED = "completed"
    ERROR = "error"
    DIVERGED = "diverged"
    UPLOADED = "uploaded"
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


class Flow360ResourceBaseModel(BaseModel):
    """
    Flow360 base Model
    """

    name: str = Field()
    user_id: str = Field(alias="userId")
    id: str = Field()
    solver_version: Union[str, None] = Field(alias="solverVersion")
    status: Flow360Status
    tags: Optional[List[str]]
    created_at: Optional[str] = Field(alias="createdAt")
    updated_at: Optional[datetime] = Field(alias="updatedAt")
    updated_by: Optional[str] = Field(alias="updatedBy")
    deleted: bool

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config:
        extra = Extra.allow
        allow_mutation = False


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

        if not UserConfig.suppress_submit_warning():
            log.warning(
                f"""\
Remeber to submit your {self.__class__.__name__} to cloud to have it processed.
Please run .submit() after .create()
To suppress this message run: flow360 configure --suppress-submit-warning"""
            )
            for line in self.traceback:
                print(line.strip())

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
            print(
                f"\
WARNING: You have not submitted your {self.__class__.__name__} to cloud. \
It will not be process. Please run .submit() after .create()"
            )
            for line in self.traceback:
                print(line.strip())


class Flow360Resource(RestApi):
    """
    Flow360 base resource model
    """

    def __init__(self, resource_type, info_type_class, *args, s3_transfer_method=None, **kwargs):
        self._resource_type = resource_type
        self.s3_transfer_method = s3_transfer_method
        self.info_type_class = info_type_class
        self._info = None
        super().__init__(*args, **kwargs)

    def __str__(self):
        return self.info.__str__()

    def __repr__(self):
        return self.info.__repr__()

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
                    resourceClass(meta_info=resourceClass._meta_class().parse_obj(item))
                    for item in resp[:limit]
                ],
            )

    @classmethod
    def from_cloud(cls):
        """
        get ResourceList from cloud
        """
        return cls(from_cloud=True)
