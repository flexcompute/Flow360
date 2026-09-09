"""Base class of resource/asset (geometry, surface mesh, volume mesh and case)"""

from __future__ import annotations

import copy
import json
import os
import time
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union

from flow360_schema import __version__ as _schema_version
from flow360_schema.models.entity_info import EntityInfoModel, parse_entity_info_model
from flow360_schema.models.simulation.simulation_params import SimulationParams
from pydantic import ValidationError
from requests.exceptions import HTTPError
from wcmatch import glob as wcglob

from flow360.cloud.flow360_requests import LengthUnitType, RenameAssetRequestV2
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import BaseInterface, ProjectInterface
from flow360.component.resource_base import (
    AssetMetaBaseModelV2,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.resource_status import is_success_resource_status
from flow360.component.simulation import services
from flow360.component.simulation.folder import Folder
from flow360.component.simulation.web.utils import (
    get_project_dependency_resource_metadata,
)
from flow360.component.utils import (
    _local_download_overwrite,
    formatting_validation_errors,
    validate_type,
)
from flow360.environment import current_environment
from flow360.exceptions import (
    Flow360FileError,
    Flow360RuntimeError,
    Flow360ValidationError,
    Flow360ValueError,
    Flow360WebError,
)
from flow360.log import log
from flow360.solver_version import warn_if_minor_release

# Segment-aware, case-insensitive globbing for cloud file paths (always '/'):
# '*'/'?' stay within one path segment, '**' matches across segments (any depth).
_DOWNLOAD_MATCH_FLAGS = wcglob.GLOBSTAR | wcglob.IGNORECASE | wcglob.FORCEUNIX


class AssetBase(metaclass=ABCMeta):
    """Base class for resource asset"""

    _interface_class: type[BaseInterface] = None
    _meta_class: type[AssetMetaBaseModelV2] = None
    _draft_class: type[ResourceDraft] = None
    _web_api_class: type[Flow360Resource] = None
    _entity_info: EntityInfoModel = None
    _cloud_resource_type_name: str = None
    # Raw simulation.json payload, cached after the first successful fetch. Birth
    # settings are immutable once the pipeline has produced them, so one download
    # serves every later read; parsing/updating re-run per call.
    _raw_simulation_json: Optional[str] = None
    # Default glob patterns used by download() when the caller passes none.
    # Child resources override this with their input-file extensions.
    _default_download_patterns: Optional[List[str]] = None

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
    def tags(self) -> List[str]:
        """
        get asset tags
        """
        return self.info.tags

    @property
    def solver_version(self):
        """
        get solver version
        """
        return self.info.solver_version

    def rename(self, new_name: str):
        """
        Rename the current asset.

        Parameters
        ----------
        new_name : str
            The new name for the asset.
        """
        RestApi(self._interface_class.endpoint, environment_provider=current_environment).patch(
            RenameAssetRequestV2(name=new_name).dict(), method=self.id
        )

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
        return self._webapi.short_description(project_id=self.project_id)

    @property
    def name(self):
        """
        returns name of resource
        """
        return self.info.name

    @classmethod
    def _from_supplied_simulation_dict(
        cls,
        simulation_dict: dict,
        asset_obj: AssetBase,
    ):
        # pylint: disable=protected-access
        simulation_dict, forward_compatibility_mode = SimulationParams._update_param_dict(
            simulation_dict
        )
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
        entity_info_dict = SimulationParams._sanitize_params_dict(entity_info_dict)
        # pylint: disable=protected-access
        try:
            asset_obj._entity_info = parse_entity_info_model(entity_info_dict)
        except ValidationError as e:
            errors = e.errors()
            log.error(formatting_validation_errors(errors=errors))
            cloud_version_str = SimulationParams._get_version_from_dict(model_dict=simulation_dict)
            if forward_compatibility_mode:
                raise Flow360RuntimeError(
                    "The cloud `SimulationParam` (version: "
                    + cloud_version_str
                    + ") is too new for your local schema package (version: "
                    + _schema_version
                    + ") and validation error occurred. Please try updating your local Python client."
                ) from None
            raise Flow360RuntimeError("Parsing cloud resource's entity info failed.") from None
        return asset_obj

    @classmethod
    def _get_simulation_json(cls, asset: AssetBase, clean_front_end_keys: bool = False) -> dict:
        """Get the simulation json AKA birth setting of the asset.

        The raw payload is cached on the asset instance, so only the first call
        downloads (and, for root/dependency items, waits on the pipeline that
        produces the file). Parsing and the updater re-run per call, so callers
        may freely mutate the returned dict.
        """
        # pylint: disable=protected-access
        if asset._raw_simulation_json is None:
            ##>> Check if the current asset is project's root item or the dependency assets is still processing
            ##>> If so then we need to wait for its pipeline to finish generating the simulation json.
            _resp = RestApi(
                ProjectInterface.endpoint,
                id=asset.project_id,
                environment_provider=current_environment,
            ).get()
            dependency_ids = []
            if asset._cloud_resource_type_name in ["Geometry", "SurfaceMesh"]:
                _dependency_metadata = get_project_dependency_resource_metadata(
                    project_id=asset.project_id, resource_type=asset._cloud_resource_type_name
                )
                dependency_ids = [_item.resource_id for _item in _dependency_metadata]
            if asset.id == _resp["rootItemId"] or asset.id in dependency_ids:
                log.debug(
                    "Current asset is project's root/dependency item. Waiting for pipeline to finish."
                )
                asset.wait()
                status = asset._webapi.status
                if not is_success_resource_status(asset._cloud_resource_type_name, status):
                    raise Flow360RuntimeError(
                        f"Cannot load {asset._cloud_resource_type_name} {asset.id} because its "
                        f"status is {status.value}."
                    )

            try:
                asset._raw_simulation_json = asset._webapi.get(
                    method="simulation/file", params={"type": "simulation"}
                )["simulationJson"]
            except HTTPError:
                # pylint:disable = raise-missing-from
                raise Flow360WebError(
                    f"Failed to get simulation json for {asset._cloud_resource_type_name}."
                )

        updated_params_as_dict, _ = SimulationParams._update_param_dict(
            json.loads(asset._raw_simulation_json)
        )
        if clean_front_end_keys:
            updated_params_as_dict = SimulationParams._sanitize_params_dict(updated_params_as_dict)
        return updated_params_as_dict

    @property
    def info(self) -> AssetMetaBaseModelV2:
        """Return the metadata of the asset"""
        return self._webapi.info

    @property
    def entity_info(self):
        """Return the entity info associated with the asset (copy to prevent unintentional overwrites)"""
        return self._entity_info.deserialize(self._entity_info.model_dump())

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

    def download(
        self,
        patterns: Optional[Union[str, List[str]]] = None,
        to_folder: str = ".",
        overwrite: bool = True,
    ) -> List[str]:
        """Download files matching the given glob pattern(s) from the cloud.

        Patterns are matched (case-insensitively) against each file's full cloud
        path, with segment-aware globbing: ``*`` and ``?`` do not cross ``/``, so
        ``*.cgns`` matches only root-level files, ``results/*.cgns`` matches one
        level under ``results``, and ``**/*.cgns`` matches any depth. If no
        pattern matches any file a :class:`Flow360FileError` is raised. Matched
        files keep their cloud subfolder layout beneath ``to_folder``.

        Parameters
        ----------
        patterns : Optional[Union[str, List[str]]]
            One or more glob patterns matched against the cloud path, e.g.
            ``"*.csm"``, ``"results/*.csv"``, ``["surface.cgns", "**/*.log"]``.
            When omitted, the resource's default input-file patterns are used
            (e.g. the geometry/mesh source files it was created from); those
            match root-level files only, so pipeline output folders such as
            ``results/`` and ``logs/`` are skipped.
        to_folder : str
            Local destination folder (created if missing). Defaults to the
            current directory.
        overwrite : bool
            Overwrite existing local files with the same name.

        Returns
        -------
        List[str]
            Absolute paths of the downloaded files.
        """
        patterns = patterns if patterns is not None else self._default_download_patterns
        if patterns is None:
            raise Flow360ValueError(
                f"No download patterns provided and {self._cloud_resource_type_name} has no "
                "default input-file patterns. Pass an explicit `patterns` argument."
            )
        patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        available = [f["fileName"] for f in self.get_download_file_list()]

        matched = sorted(
            name
            for name in available
            if wcglob.globmatch(name, patterns, flags=_DOWNLOAD_MATCH_FLAGS)
        )
        if not matched:
            raise Flow360FileError(
                f"No files on {self._cloud_resource_type_name} '{self.name}' match patterns "
                f"{patterns}. Available files: {available}."
            )

        # Preserve each file's cloud subfolder layout under to_folder so distinct
        # cloud paths that share a base name do not overwrite each other.
        # pylint: disable=protected-access
        paths = []
        for name in matched:
            dest_folder = os.path.join(to_folder, os.path.dirname(name))
            os.makedirs(dest_folder, exist_ok=True)
            downloaded = self._webapi._download_file(
                name, to_folder=dest_folder, overwrite=overwrite
            )
            paths.append(os.path.abspath(downloaded))
        log.info(
            f"Downloaded {len(paths)} file(s) to '{os.path.abspath(to_folder)}': "
            + ", ".join(matched)
        )
        return paths

    @abstractmethod
    def _apply_dynamic_default_settings(self, simulation_dict):
        """Populate the asset's dynamic defaults from the non-entity part of the root asset's
        simulation dict. Part of asset initialization; mutates the asset in place."""

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

        asset_obj = cls._from_supplied_simulation_dict(
            entity_info_supplier_dict if entity_info_supplier_dict else asset_simulation_dict,
            asset_obj,
        )
        # The default_settings will only make a difference when the asset is project root asset,
        # but we try to get it regardless to save the logic differentiating whether it is root or not.
        asset_obj._apply_dynamic_default_settings(asset_simulation_dict)

        # Attempting constructing entity registry.
        # This ensure that once from_cloud() returns, the entity_registry will be available.
        asset_obj.internal_registry = asset_obj._entity_info.get_persistent_entity_registry(
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
        length_unit: Optional[LengthUnitType] = None,
        tags: List[str] = None,
        folder: Optional[Folder] = None,
    ):
        """
        Create asset draft from files
        :param file_names:
        :param project_name:
        :param tags:
        :param folder: Folder object where the asset will be created (optional; defaults to root if unspecified)
        :return:
        """
        warn_if_minor_release(solver_version)
        # pylint: disable=not-callable
        return cls._draft_class(
            file_names=file_names,
            project_name=project_name,
            solver_version=solver_version,
            tags=tags,
            length_unit=length_unit,
            folder=folder,
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

        # pylint: disable=protected-access
        asset_obj = cls._from_supplied_simulation_dict(params_dict, cls(asset_id))
        asset_obj._apply_dynamic_default_settings(params_dict)

        # _simulation_dict_cache_for_local_mode for local mode to avoid hitting cloud APIs.
        # pylint: disable=attribute-defined-outside-init
        asset_obj._simulation_dict_cache_for_local_mode = copy.deepcopy(params_dict)
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
