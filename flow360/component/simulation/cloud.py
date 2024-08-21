"""Cloud communication for simulation related tasks"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Annotated, Literal

import pydantic as pd

from flow360.cloud.rest_api import RestApi
from flow360.component.case import Case
from flow360.component.interfaces import DraftInterface, ProjectInterface
from flow360.component.resource_base import (
    AssetMetaBaseModel,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.utils import _model_attribute_unlock
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.surface_mesh import SurfaceMesh
from flow360.component.utils import is_valid_uuid, validate_type
from flow360.component.volume_mesh import VolumeMesh
from flow360.exceptions import Flow360WebError
from flow360.log import log

TIMEOUT_MINUTES = 60


def _valid_id_validator(input_id: str):
    is_valid_uuid(input_id)
    return input_id


IDStringType = Annotated[str, pd.AfterValidator(_valid_id_validator)]


class DraftPostRequest(pd.BaseModel):
    """Data model for draft post request"""

    name: str = pd.Field(
        default_factory=lambda: "Draft " + datetime.now().strftime("%m-%d %H:%M:%S")
    )
    project_id: IDStringType = pd.Field(serialization_alias="projectId")
    source_item_id: IDStringType = pd.Field(serialization_alias="sourceItemId")
    source_item_type: Literal[
        "Project", "Folder", "Geometry", "SurfaceMesh", "VolumeMesh", "Case", "Draft"
    ] = pd.Field(serialization_alias="sourceItemType")
    solver_version: str = pd.Field(serialization_alias="solverVersion")
    fork_case: bool = pd.Field(serialization_alias="forkCase")


class DraftDraft(ResourceDraft):
    """
    Draft Draft component
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        name: str,
        project_id: str,
        source_item_id: str,
        source_item_type: Literal[
            "Project", "Folder", "Geometry", "SurfaceMesh", "VolumeMesh", "Case", "Draft"
        ],
        solver_version: str,
        fork_case: bool,
    ):
        self._request = DraftPostRequest(
            name=name,
            project_id=project_id,
            source_item_id=source_item_id,
            source_item_type=source_item_type,
            solver_version=solver_version,
            fork_case=fork_case,
        )
        ResourceDraft.__init__(self)

    def submit(self) -> Draft:
        """
        Submit draft to cloud and under a given project
        """
        draft_id = RestApi(DraftInterface.endpoint).post(self._request.model_dump(by_alias=True))
        return Draft.from_cloud(draft_id["id"])


class Draft(Flow360Resource):
    """Project Draft component"""

    def __init__(self, draft_id: IDStringType):
        super().__init__(
            interface=DraftInterface,
            meta_class=AssetMetaBaseModel,  # We do not have dedicated meta class for Draft
            id=draft_id,
        )

    @classmethod
    # pylint: disable=protected-access
    def _from_meta(cls, meta: AssetMetaBaseModel):
        validate_type(meta, "meta", AssetMetaBaseModel)
        resource = cls(draft_id=meta.id)
        return resource

    # pylint: disable=too-many-arguments
    @classmethod
    def create(
        cls,
        name: str = None,
        project_id: IDStringType = None,
        source_item_id: IDStringType = None,
        source_item_type: Literal[
            "Project", "Folder", "Geometry", "SurfaceMesh", "VolumeMesh", "Case", "Draft"
        ] = None,
        solver_version: str = None,
        fork_case: bool = None,
    ) -> DraftDraft:
        """Create a new instance of DraftDraft"""
        return DraftDraft(
            name=name,
            project_id=project_id,
            source_item_id=source_item_id,
            source_item_type=source_item_type,
            solver_version=solver_version,
            fork_case=fork_case,
        )

    @classmethod
    def from_cloud(cls, draft_id: IDStringType) -> Draft:
        """Load draft from cloud"""
        return Draft(draft_id=draft_id)

    def update_simulation_params(self, params: SimulationParams):
        """update the SimulationParams of the draft"""

        self.post(
            json={"data": params.model_dump_json(), "type": "simulation", "version": ""},
            method="simulation/file",
        )

    def retrieve_simulation_dict(self) -> dict:
        """retrieve the SimulationParams of the draft"""
        response = self.get(method="simulation-config")
        return json.loads(response["simulationJson"])

    def run_up_to_target_asset(self, target_asset: type[AssetBase]) -> str:
        """run the draft up to the target asset"""

        try:
            run_response = self.post(
                json={"upTo": target_asset.__name__, "useInHouse": True},
                method="run",
            )
        except Flow360WebError as err:
            # Error found when translating/runing the simulation
            detailed_error = json.loads(err.auxiliary_json["detail"])["detail"]
            log.error(f"Failure detail: {detailed_error}")
            raise RuntimeError(f"Failure detail: {detailed_error}") from err

        destination_id = run_response["id"]
        return destination_id


def _check_project_path_status(project_id: str, item_id: str, item_type: str) -> None:
    RestApi(ProjectInterface.endpoint, id=project_id).get(
        method="path", params={"itemId": item_id, "itemType": item_type}
    )
    # pylint: disable=fixme
    # TODO: check all status on the given path


# pylint: disable=too-many-arguments
def _run(
    source_asset: AssetBase,
    params: SimulationParams,
    target_asset: type[AssetBase],
    draft_name: str = None,
    fork_case: bool = False,
    async_mode: bool = True,
) -> AssetBase:
    """
    Generate surface mesh with given simulation params.
    async_mode: if True, returns SurfaceMesh object immediately, otherwise waits for the meshing to finish.
    """
    if not isinstance(params, SimulationParams):
        raise ValueError(
            f"params argument must be a SimulationParams object but is of type {type(params)}"
        )

    ##-- Getting the project length unit from draft and store in the SimulationParams
    _resp = RestApi(ProjectInterface.endpoint, id=source_asset.project_id).get()
    last_opened_draft_id = _resp["lastOpenDraftId"]
    assert last_opened_draft_id is not None
    _draft_with_length_unit = Draft.from_cloud(last_opened_draft_id)
    length_unit = _draft_with_length_unit.retrieve_simulation_dict()[
        "private_attribute_asset_cache"
    ]["project_length_unit"]
    with _model_attribute_unlock(params.private_attribute_asset_cache, "project_length_unit"):
        # pylint: disable=no-member
        params.private_attribute_asset_cache.project_length_unit = LengthType.validate(length_unit)

    ##-- Get new draft
    _draft = Draft.create(
        name=draft_name,
        project_id=source_asset.project_id,
        source_item_id=source_asset.id,
        source_item_type=source_asset.__class__.__name__,
        solver_version=source_asset.solver_version,
        fork_case=fork_case,
    ).submit()

    ##-- Post the simulation param:
    _draft.update_simulation_params(params)

    ##-- Kick off draft run:
    destination_id = _draft.run_up_to_target_asset(target_asset)

    ##-- Patch project
    RestApi(ProjectInterface.endpoint, id=source_asset.project_id).patch(
        json={
            "lastOpenItemId": destination_id,
            "lastOpenItemType": target_asset.__name__,
        }
    )
    destination_obj = target_asset.from_cloud(destination_id)

    if async_mode is False:
        start_time = time.time()
        while destination_obj.status.is_final() is False:
            if time.time() - start_time > TIMEOUT_MINUTES * 60:
                raise TimeoutError(
                    "Timeout: Process did not finish within the specified timeout period"
                )
            _check_project_path_status(
                source_asset.project_id, source_asset.id, source_asset.__class__.__name__
            )
            log.info("Waiting for the process to finish...")
            time.sleep(2)
    return destination_obj


def generate_surface_mesh(
    source_asset: AssetBase,
    params: SimulationParams,
    draft_name: str = None,
    async_mode: bool = True,
):
    """generate surface mesh from the geometry"""
    return _run(source_asset, params, SurfaceMesh, draft_name, False, async_mode)


def generate_volume_mesh(
    source_asset: AssetBase,
    params: SimulationParams,
    draft_name: str = None,
    async_mode: bool = True,
):
    """generate volume mesh from the geometry"""
    return _run(source_asset, params, VolumeMesh, draft_name, False, async_mode)


def run_case(
    source_asset: AssetBase,
    params: SimulationParams,
    draft_name: str = None,
    async_mode: bool = True,
):
    """run case from the geometry"""
    return _run(source_asset, params, Case, draft_name, False, async_mode)
