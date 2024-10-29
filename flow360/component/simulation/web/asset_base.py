"""Base class of resource/asset (geometry, surface mesh, volume mesh and case)"""

from __future__ import annotations

import json
import time
from abc import ABCMeta
from typing import List, Union

from flow360.cloud.requests import LengthUnitType
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import BaseInterface, ProjectInterface
from flow360.component.resource_base import (
    AssetMetaBaseModel,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.simulation.entity_info import EntityInfoModel
from flow360.component.utils import remove_properties_by_name, validate_type
from flow360.log import log


class AssetBase(metaclass=ABCMeta):
    """Base class for resource asset"""

    _interface_class: type[BaseInterface] = None
    _meta_class: type[AssetMetaBaseModel] = None
    _draft_class: type[ResourceDraft] = None
    _web_api_class: type[Flow360Resource] = None
    _entity_info_class: type[EntityInfoModel] = None
    _entity_info: EntityInfoModel = None

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        # pylint: disable=not-callable
        self._webapi = self.__class__._web_api_class(
            interface=self._interface_class,
            meta_class=self._meta_class,
            id=id,
        )
        self.id = id
        # get the project id according to resource id
        resp = self._webapi.get()
        project_id = resp["projectId"]
        solver_version = resp["solverVersion"]
        self.project_id: str = project_id
        self.solver_version: str = solver_version
        self.internal_registry = None

    @classmethod
    # pylint: disable=protected-access
    def _from_meta(cls, meta: AssetMetaBaseModel):
        validate_type(meta, "meta", cls._meta_class)
        resource = cls(id=meta.id)
        resource._webapi._set_meta(meta)
        return resource

    @classmethod
    def _get_simulation_json(cls, asset: AssetBase) -> dict:
        """Get the simulation json AKA birth setting of the asset. Do we want to cache it in the asset object?"""
        ##>> Check if the current asset is project's root item.
        ##>> If so then we need to wait for its pipeline to finish generating the simulation json.
        _resp = RestApi(ProjectInterface.endpoint, id=asset.project_id).get()
        if asset.id == _resp["rootItemId"]:
            log.debug("Current asset is project's root item. Waiting for pipeline to finish.")
            # pylint: disable=protected-access
            asset.wait()

        # pylint: disable=protected-access
        simulation_json = asset._webapi.get(
            method="simulation/file", params={"type": "simulation"}
        )["simulationJson"]
        return json.loads(simulation_json)

    @property
    def info(self) -> AssetMetaBaseModel:
        """Return the metadata of the asset"""
        return self._webapi.info

    @property
    def entity_info(self):
        """Return the entity info associated with the asset (copy to prevent unintentional overwrites)"""
        return self._entity_info_class.model_validate(self._entity_info.model_dump())

    @classmethod
    def _interface(cls):
        return cls._interface_class

    @classmethod
    def _meta_class(cls):
        return cls._meta_class

    @classmethod
    def from_cloud(cls, id: str):
        """Create asset with the given ID"""
        asset_obj = cls(id)
        # populating the entityInfo object
        simulation_dict = cls._get_simulation_json(asset_obj)
        if "private_attribute_asset_cache" not in simulation_dict:
            raise KeyError(
                "[Internal] Could not find private_attribute_asset_cache in the asset's simulation settings."
            )
        asset_cache = simulation_dict["private_attribute_asset_cache"]

        if "project_entity_info" not in asset_cache:
            raise KeyError(
                "[Internal] Could not find project_entity_info in the asset's simulation settings."
            )
        entity_info = asset_cache["project_entity_info"]
        # Note: There is no need to exclude _id here since the birth setting of root item will never have _id.
        # Note: Only the draft's and non-root item simulation.json will have it.
        # Note: But we still add this because it is not clear currently if Asset is alywas the root item.
        # Note: This should be addressed when we design the new project client interface.
        remove_properties_by_name(entity_info, "_id")
        asset_obj._entity_info = cls._entity_info_class.model_validate(entity_info)
        return asset_obj

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
        :param project_name:
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

    def wait(self, timeout_minutes=60):
        """Wait until the Asset finishes processing, refresh periodically"""

        start_time = time.time()
        while self._webapi.status.is_final() is False:
            if time.time() - start_time > timeout_minutes * 60:
                raise TimeoutError(
                    "Timeout: Process did not finish within the specified timeout period"
                )
            time.sleep(2)
