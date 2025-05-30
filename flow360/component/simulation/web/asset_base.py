"""Base class of resource/asset (geometry, surface mesh, volume mesh and case)"""

from __future__ import annotations

import json
import os
import time
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

from requests.exceptions import HTTPError

from flow360.cloud.flow360_requests import LengthUnitType
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import BaseInterface, ProjectInterface
from flow360.component.resource_base import (
    AssetMetaBaseModelV2,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.simulation import services
from flow360.component.simulation.entity_info import (
    EntityInfoModel,
    parse_entity_info_model,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.utils import (
    _local_download_overwrite,
    remove_properties_by_name,
    validate_type,
)
from flow360.exceptions import Flow360ValidationError, Flow360WebError
from flow360.log import log


class AssetBase(metaclass=ABCMeta):
    """Base class for resource asset"""

    _interface_class: type[BaseInterface] = None
    _meta_class: type[AssetMetaBaseModelV2] = None
    _draft_class: type[ResourceDraft] = None
    _web_api_class: type[Flow360Resource] = None
    _entity_info_class: type[EntityInfoModel] = None
    _entity_info: EntityInfoModel = None
    _cloud_resource_type_name: str = None

    # pylint: disable=redefined-builtin
    def __init__(self, id: Union[str, None]):
        """When id is None, the asset is meant to be operating in local mode."""
        # pylint: disable=not-callable
        self.id = id
        self.internal_registry = None
        # The default_settings will only be used when the current instance is project's root
        self.default_settings = {}
        if id is None:
            return
        self._webapi = self.__class__._web_api_class(
            interface=self._interface_class,
            meta_class=self._meta_class,
            id=id,
        )

    @property
    def project_id(self):
        """
        get project ID
        """
        return self.info.project_id

    @property
    def solver_version(self):
        """
        get solver version
        """
        return self.info.solver_version

    @classmethod
    # pylint: disable=protected-access
    def _from_meta(cls, meta: AssetMetaBaseModelV2):
        validate_type(meta, "meta", cls._meta_class)
        resource = cls(id=meta.id)
        resource._webapi._set_meta(meta)
        return resource

    def short_description(self) -> str:
        """short_description
        Returns
        -------
        str
            generates short description of resource (name, id, status)
        """
        return self._webapi.short_description()

    @property
    def name(self):
        """
        returns name of resource
        """
        return self.info.name

    @classmethod
    def _from_supplied_entity_info(
        cls,
        simulation_dict: dict,
        asset_obj: AssetBase,
    ):
        if "private_attribute_asset_cache" not in simulation_dict:
            raise KeyError(
                "[Internal] Could not find private_attribute_asset_cache in the asset's simulation settings."
            )
        asset_cache = simulation_dict["private_attribute_asset_cache"]

        if "project_entity_info" not in asset_cache:
            raise KeyError(
                "[Internal] Could not find project_entity_info in the asset's simulation settings."
            )
        entity_info_dict = asset_cache["project_entity_info"]
        entity_info_dict = remove_properties_by_name(entity_info_dict, "_id")
        # pylint: disable=protected-access
        asset_obj._entity_info = parse_entity_info_model(entity_info_dict)
        return asset_obj

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
        try:
            simulation_json = asset._webapi.get(
                method="simulation/file", params={"type": "simulation"}
            )["simulationJson"]
        except HTTPError:
            # pylint:disable = raise-missing-from
            raise Flow360WebError(
                f"Failed to get simulation json for {asset._cloud_resource_type_name}."
            )

        updated_params_as_dict, _ = SimulationParams._update_param_dict(json.loads(simulation_json))
        return updated_params_as_dict

    @property
    def info(self) -> AssetMetaBaseModelV2:
        """Return the metadata of the asset"""
        return self._webapi.info

    @property
    def entity_info(self):
        """Return the entity info associated with the asset (copy to prevent unintentional overwrites)"""
        return self._entity_info_class.model_validate(self._entity_info.model_dump())

    @property
    def params(self):
        """Return the simulation parameters associated with the asset"""
        params_as_dict = self._get_simulation_json(self)

        # pylint: disable=duplicate-code
        param, errors, _ = services.validate_model(
            params_as_dict=params_as_dict,
            validated_by=services.ValidationCalledBy.LOCAL,
            root_item_type=None,
            validation_level=None,
        )

        if errors is not None:
            raise Flow360ValidationError(
                f"Error found in simulation params. The param may be created by an incompatible version. {errors}",
            )

        return param

    @classmethod
    def _interface(cls):
        return cls._interface_class

    @classmethod
    def _meta_class(cls):
        return cls._meta_class

    def get_download_file_list(self) -> List:
        """return list of files available for download

        Returns
        -------
        List
            List of files available for download
        """
        return self._webapi.get_download_file_list()

    @abstractmethod
    def get_default_settings(self, simulation_dict):
        """Get the default settings of the asset from the non-entity part of root asset's simulation dict"""

    @classmethod
    def from_cloud(cls, id: str, **kwargs):
        """
        Create asset with the given ID.
        """
        asset_obj = cls(id)
        entity_info_supplier_dict = None
        entity_info_param: Optional[SimulationParams] = kwargs.pop("entity_info_param", None)
        if entity_info_param:
            # Use user requested json.
            entity_info_supplier_dict = entity_info_param.model_dump(mode="json")
        # Get the json from bucket, same as before.
        asset_simulation_dict = cls._get_simulation_json(asset_obj)

        asset_obj = cls._from_supplied_entity_info(
            entity_info_supplier_dict if entity_info_supplier_dict else asset_simulation_dict,
            asset_obj,
        )
        # The default_settings will only make a difference when the asset is project root asset,
        # but we try to get it regardless to save the logic differentiating whether it is root or not.
        asset_obj.get_default_settings(asset_simulation_dict)

        # Attempting constructing entity registry.
        # This ensure that once from_cloud() returns, the entity_registry will be available.
        asset_obj.internal_registry = asset_obj._entity_info.get_registry(
            asset_obj.internal_registry
        )
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
        # pylint: disable=not-callable
        return cls._draft_class(
            file_names=file_names,
            project_name=project_name,
            solver_version=solver_version,
            tags=tags,
            length_unit=length_unit,
        )

    @classmethod
    def _from_local_storage(
        cls, asset_id: str = None, local_storage_path="", meta_data: AssetMetaBaseModelV2 = None
    ):
        """
        Create asset from local storage
        :param asset_id: ID of the asset
        :param local_storage_path: The folder of the project, defaults to current working directory
        :return: asset object
        """

        _local_download_file = _local_download_overwrite(local_storage_path, cls.__name__)
        _local_download_file(file_name="simulation.json", to_folder=local_storage_path)
        with open(os.path.join(local_storage_path, "simulation.json"), encoding="utf-8") as f:
            params_dict = json.load(f)

        asset_obj = cls._from_supplied_entity_info(params_dict, cls(asset_id))
        # pylint: disable=protected-access
        if not hasattr(asset_obj, "_webapi"):
            # Handle local test case execution which has no valid ID
            return asset_obj
        asset_obj._webapi._download_file = _local_download_file
        if meta_data is not None:
            asset_obj._webapi._set_meta(meta_data)
        return asset_obj

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
            while not self._webapi.status.is_final():

                elapsed = time.time() - start_time
                dot_count = int((elapsed // update_every_seconds) % max_dots)
                status_logger.update(f"{self._webapi.status.value}{'.' * dot_count}")

                if time.time() - start_time > timeout_minutes * 60:
                    raise TimeoutError(
                        "Timeout: Process did not finish within the specified timeout period"
                    )

                time.sleep(update_every_seconds)
