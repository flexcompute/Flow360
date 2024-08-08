"""Base class of resource/asset (geometry, surface mesh, volume mesh and case)"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import List, Union

from flow360.cloud.requests import LengthUnitType
from flow360.component.interfaces import BaseInterface
from flow360.component.resource_base import (
    AssetMetaBaseModel,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.utils import validate_type


class AssetBase(metaclass=ABCMeta):
    """Base class for resource asset"""

    _interface_class: type[BaseInterface] = None
    _meta_class: type[AssetMetaBaseModel] = None
    _draft_class: type[ResourceDraft] = None
    id: str
    project_id: str
    solver_version: str

    @abstractmethod
    def _get_metadata(self) -> None:
        # get the metadata when initializing the object (blocking)
        pass

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        self._webapi = Flow360Resource(
            interface=self._interface_class,
            meta_class=self._meta_class,
            id=id,
        )
        self.id = id
        self._get_metadata()
        # get the project id according to resource id
        resp = self._webapi.get()
        project_id = resp["projectId"]
        solver_version = resp["solverVersion"]
        self.project_id = project_id
        self.solver_version = solver_version

    @classmethod
    # pylint: disable=protected-access
    def _from_meta(cls, meta: AssetMetaBaseModel):
        validate_type(meta, "meta", cls._meta_class)
        resource = cls(id=meta.id)
        resource._webapi._set_meta(meta)
        return resource

    @property
    def info(self) -> AssetMetaBaseModel:
        """Return the metadata of the resource"""
        return self._webapi.info

    @classmethod
    def _interface(cls):
        return cls._interface_class

    @classmethod
    def _meta_class(cls):
        return cls._meta_class

    @classmethod
    def from_cloud(cls, id: str):
        """Create asset with the given ID"""
        return cls(id)

    @classmethod
    # pylint: disable=too-many-arguments
    def from_file(
        cls,
        file_names: Union[List[str], str],
        project_name: str = None,
        solver_version: str = None,
        length_unit: LengthUnitType = "m",
        tags: List[str] = None,
    ):
        """
        Create asset draft from files
        :param file_names:
        :param name:
        :param tags:
        :return:
        """
        if isinstance(file_names, str):
            file_names = [file_names]
        # pylint: disable=not-callable
        return cls._draft_class(
            file_names=file_names,
            project_name=project_name,
            solver_version=solver_version,
            tags=tags,
            length_unit=length_unit,
        )
