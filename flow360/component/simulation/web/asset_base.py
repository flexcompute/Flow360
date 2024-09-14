"""Base class of resource/asset (geometry, surface mesh, volume mesh and case)"""

from __future__ import annotations

from abc import ABCMeta
from typing import List, Union

from flow360.cloud.requests import LengthUnitType
from flow360.component.interfaces import BaseInterface
from flow360.component.resource_base import (
    AssetMetaBaseModel,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.simulation.entity_info import EntityInfoModel
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.utils import _model_attribute_unlock
from flow360.component.simulation.web.draft import _get_simulation_json_from_cloud
from flow360.component.utils import validate_type


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
        asset_obj = cls(id)
        # populating the entityInfo
        simulation_dict = _get_simulation_json_from_cloud(asset_obj.project_id)
        if "private_attribute_asset_cache" not in simulation_dict:
            raise KeyError(
                "[Internal] Could not find private_attribute_asset_cache in the draft's simulation settings."
            )
        asset_cache = simulation_dict["private_attribute_asset_cache"]

        if "project_entity_info" not in asset_cache:
            raise KeyError(
                "[Internal] Could not find project_entity_info in the draft's simulation settings."
            )
        entity_info = asset_cache["project_entity_info"]

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

    def _inject_entity_info_to_params(self, params):
        """Inject the length unit into the SimulationParams"""
        # Add used cylinder, box, point and slice entities to the entityInfo.
        # pylint: disable=protected-access
        registry: EntityRegistry = params._get_used_entity_registry()
        old_draft_entities = self._entity_info.draft_entities
        # Step 1: Update old ones:
        for _, old_entity in enumerate(old_draft_entities):
            try:
                _ = registry.find_by_naming_pattern(old_entity.name)
            except ValueError:  # old_entity did not apperar in params.
                continue

        # pylint: disable=protected-access
        with _model_attribute_unlock(params.private_attribute_asset_cache, "project_entity_info"):
            params.private_attribute_asset_cache.project_entity_info = self._entity_info
        return params
