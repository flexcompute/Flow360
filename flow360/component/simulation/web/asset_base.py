"""Base class of resource/asset (geometry, surface mesh, volume mesh and case)"""

from __future__ import annotations

import json
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import List, Union

from flow360.cloud.requests import LengthUnitType
from flow360.cloud.rest_api import RestApi
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.web.interfaces import (
    BaseInterface,
    DraftInterface,
    ProjectInterface,
)
from flow360.component.simulation.web.resource_base import (
    AssetMetaBaseModel,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.utils import validate_type
from flow360.log import log

TIMEOUT_MINUTES = 60


class AssetBase(metaclass=ABCMeta):
    """Base class for resource asset"""

    _interface: type[BaseInterface] = None
    _meta_class: type[AssetMetaBaseModel] = None
    _draft_class: type[ResourceDraft] = None

    @abstractmethod
    def _retrieve_metadata(self) -> None:
        # get the metadata when initializing the object (blocking)
        pass

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        self._web = Flow360Resource(
            interface=self._interface,
            meta_class=self._meta_class,
            id=id,
        )
        self._retrieve_metadata()
        # get the project id according to resource id
        resp = RestApi(self._interface.endpoint, id=id).get()
        project_id = resp["projectId"]
        solver_version = resp["solverVersion"]
        self.project_id = project_id
        self.solver_version = solver_version

    @classmethod
    # pylint: disable=protected-access
    def _from_meta(cls, meta: AssetMetaBaseModel):
        validate_type(meta, "meta", cls._meta_class)
        resource = cls(id=meta.id)
        resource._web._set_meta(meta)
        return resource

    @property
    def info(self) -> AssetMetaBaseModel:
        """Return the metadata of the resource"""
        return self._web.info

    @classmethod
    def _interface(cls):
        return cls._interface

    @classmethod
    def _meta_class(cls):
        return cls._meta_class

    @classmethod
    def from_cloud(cls, id: str):
        """Create asset with the given ID"""
        return cls(id)

    @classmethod
    def from_file(
        cls,
        file_names: Union[List[str], str],
        name: str = None,
        tags: List[str] = None,
        length_unit: LengthUnitType = "m",
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
            name=name,
            tags=tags,
            length_unit=length_unit,
        )

    def run(
        self,
        params: SimulationParams,
        destination: AssetBase,
        async_mode: bool = True,
    ) -> AssetBase:
        """
        Generate surface mesh with given simulation params.
        async_mode: if True, returns SurfaceMesh object immediately, otherwise waits for the meshing to finish.
        """
        assert isinstance(params, SimulationParams), "params must be a SimulationParams object."
        ##-- Get the latest draft of the project:
        draft_id = RestApi(ProjectInterface.endpoint, id=self.project_id).get()["lastOpenDraftId"]
        if draft_id is None:  # No saved online session
            ##-- Get new draft
            draft_id = RestApi(DraftInterface.endpoint).post(
                {
                    "name": "Client " + datetime.now().strftime("%m-%d %H:%M:%S"),
                    "projectId": self.project_id,
                    "sourceItemId": self._web.id,
                    "sourceItemType": "Geometry",
                    "solverVersion": self.solver_version,
                    "forkCase": False,
                }
            )["id"]
        ##-- Post the simulation param:
        req = {"data": params.model_dump_json(), "type": "simulation", "version": ""}
        RestApi(DraftInterface.endpoint, id=draft_id).post(json=req, method="simulation/file")
        ##-- Kick off draft run:
        run_response = RestApi(DraftInterface.endpoint, id=draft_id).post(
            json={"upTo": destination.__name__, "useInHouse": True},
            method="run",
            ignore_request_error=True,
        )

        if "error" in run_response:
            # Error found when translating/runing the simulation
            print(run_response)
            detailed_error = json.loads(run_response.json()["detail"])["detail"]
            raise RuntimeError(f"Run failed: {detailed_error}")

        destination_id = run_response["id"]
        ##-- Patch project
        RestApi(ProjectInterface.endpoint, id=self.project_id).patch(
            json={
                "lastOpenItemId": destination_id,
                "lastOpenItemType": destination.__name__,
            }
        )
        destination_obj = destination.from_cloud(destination_id)
        if async_mode is False:
            start_time = time.time()
            while destination_obj.status.is_final() is False:
                if time.time() - start_time > TIMEOUT_MINUTES * 60:
                    raise TimeoutError(
                        "Timeout: Process did not finish within the specified timeout period"
                    )
                log.info("Waiting for the process to finish...")
                time.sleep(30)
        return destination_obj
