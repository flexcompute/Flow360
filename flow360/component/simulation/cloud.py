"""Cloud communication for simulation related tasks"""

from __future__ import annotations

import time

from flow360.cloud.rest_api import RestApi
from flow360.component.case import Case
from flow360.component.interfaces import ProjectInterface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.utils import _model_attribute_unlock
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.simulation.web.draft import (
    Draft,
    _get_simulation_json_from_cloud,
)
from flow360.component.surface_mesh import SurfaceMesh
from flow360.component.volume_mesh import VolumeMesh
from flow360.log import log

TIMEOUT_MINUTES = 60


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
    simulation_dict = _get_simulation_json_from_cloud(source_asset.project_id)

    if (
        "private_attribute_asset_cache" not in simulation_dict
        or "project_length_unit" not in simulation_dict["private_attribute_asset_cache"]
    ):
        raise KeyError(
            "[Internal] Could not find project length unit in the draft's simulation settings."
        )

    length_unit = simulation_dict["private_attribute_asset_cache"]["project_length_unit"]
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

    ##-- Store the entity info part for future retrival
    # pylint: disable=protected-access
    params = source_asset._inject_entity_info_to_params(params)

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
        while destination_obj._webapi.status.is_final() is False:
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
