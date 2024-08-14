"""Cloud communication for simulation related tasks"""

import json
import time
from datetime import datetime
from typing import Annotated, Literal, Optional

import pydantic as pd

from flow360.cloud.rest_api import RestApi
from flow360.component.case import Case
from flow360.component.interfaces import DraftInterface, ProjectInterface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.surface_mesh import SurfaceMesh
from flow360.component.utils import is_valid_uuid
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

    ##-- Get new draft
    draft_id = RestApi(DraftInterface.endpoint).post(
        DraftPostRequest(
            name=draft_name,
            project_id=source_asset.project_id,
            source_item_id=source_asset.id,
            source_item_type=source_asset.__class__.__name__,
            solver_version=source_asset.solver_version,
            fork_case=fork_case,
        ).model_dump(by_alias=True)
    )["id"]

    ##-- Post the simulation param:
    RestApi(DraftInterface.endpoint, id=draft_id).post(
        json={"data": params.model_dump_json(), "type": "simulation", "version": ""},
        method="simulation/file",
    )
    ##-- Kick off draft run:
    try:
        run_response = RestApi(DraftInterface.endpoint, id=draft_id).post(
            json={"upTo": target_asset.__name__, "useInHouse": True},
            method="run",
        )
    except Flow360WebError as err:
        # Error found when translating/runing the simulation
        detailed_error = json.loads(err.auxiliary_json["detail"])["detail"]
        log.error(f"Failure detail: {detailed_error}")
        raise RuntimeError(f"Failure detail: {detailed_error}") from err

    destination_id = run_response["id"]
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
            time.sleep(10)
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
