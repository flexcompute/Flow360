"""Cloud communication for simulation related tasks"""

import json
import time
from datetime import datetime

from flow360.cloud.rest_api import RestApi
from flow360.component.case import Case
from flow360.component.interfaces import DraftInterface, ProjectInterface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.surface_mesh import SurfaceMesh
from flow360.component.volume_mesh import VolumeMesh
from flow360.exceptions import Flow360WebError
from flow360.log import log

TIMEOUT_MINUTES = 60


def _check_project_path_status(project_id: str, item_id: str, item_type: str) -> None:
    RestApi(ProjectInterface.endpoint, id=project_id).get(
        method="path", params={"itemId": item_id, "itemType": item_type}
    )
    # pylint: disable=fixme
    # TODO: check all status on the given path


def _run(
    starting_point: AssetBase,
    params: SimulationParams,
    destination: type[AssetBase],
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

    ##-- Get the latest draft of the project:
    draft_id = RestApi(ProjectInterface.endpoint, id=starting_point.project_id).get()[
        "lastOpenDraftId"
    ]
    if draft_id is None:  # No saved online session
        ##-- Get new draft
        draft_id = RestApi(DraftInterface.endpoint).post(
            {
                "name": "Client " + datetime.now().strftime("%m-%d %H:%M:%S"),
                "projectId": starting_point.project_id,
                "sourceItemId": starting_point.id,
                "sourceItemType": "Geometry",
                "solverVersion": starting_point.solver_version,
                "forkCase": False,
            }
        )["id"]
    ##-- Post the simulation param:
    req = {"data": params.model_dump_json(), "type": "simulation", "version": ""}
    RestApi(DraftInterface.endpoint, id=draft_id).post(json=req, method="simulation/file")
    ##-- Kick off draft run:
    try:
        run_response = RestApi(DraftInterface.endpoint, id=draft_id).post(
            json={"upTo": destination.__name__, "useInHouse": True},
            method="run",
        )
    except Flow360WebError as err:
        # Error found when translating/runing the simulation
        detailed_error = json.loads(err.auxiliary_json["detail"])["detail"]
        log.error(f"Failure detail: {detailed_error}")
        raise RuntimeError(f"Failure detail: {detailed_error}") from err

    destination_id = run_response["id"]
    ##-- Patch project
    RestApi(ProjectInterface.endpoint, id=starting_point.project_id).patch(
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
            _check_project_path_status(
                starting_point.project_id, starting_point.id, starting_point.__class__.__name__
            )
            log.info("Waiting for the process to finish...")
            time.sleep(10)
    return destination_obj


def generate_surface_mesh(
    starting_point: AssetBase, params: SimulationParams, async_mode: bool = True
):
    """generate surface mesh from the geometry"""
    return _run(starting_point, params, SurfaceMesh, async_mode)


def generate_volume_mesh(
    starting_point: AssetBase, params: SimulationParams, async_mode: bool = True
):
    """generate volume mesh from the geometry"""
    return _run(starting_point, params, VolumeMesh, async_mode)


def run_case(starting_point: AssetBase, params: SimulationParams, async_mode: bool = True):
    """run case from the geometry"""
    return _run(starting_point, params, Case, async_mode)
