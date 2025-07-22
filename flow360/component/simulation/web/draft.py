"""Draft for workbench realizations"""

from __future__ import annotations

import ast
import json
from typing import Literal, Union

from flow360.cloud.flow360_requests import (
    DraftCreateRequest,
    DraftRunRequest,
    ForceCreationConfig,
    IDStringType,
)
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import DraftInterface
from flow360.component.project_utils import formatting_validation_errors
from flow360.component.resource_base import (
    AssetMetaBaseModel,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.simulation.framework.updater_utils import deprecation_reminder
from flow360.component.utils import validate_type
from flow360.exceptions import Flow360RuntimeError, Flow360WebError
from flow360.log import log


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
        interpolation_volume_mesh_id: str,
        tags: list[str],
    ):
        self._request = DraftCreateRequest(
            name=name,
            project_id=project_id,
            source_item_id=source_item_id,
            source_item_type=source_item_type,
            solver_version=solver_version,
            fork_case=fork_case,
            interpolation_volume_mesh_id=interpolation_volume_mesh_id,
            interpolation_case_id=source_item_id if interpolation_volume_mesh_id else None,
            tags=tags,
        )
        ResourceDraft.__init__(self)

    def submit(self) -> Draft:
        """
        Submit draft to cloud and under a given project
        """
        draft_meta = RestApi(DraftInterface.endpoint).post(self._request.model_dump(by_alias=True))
        self._id = draft_meta["id"]
        return Draft.from_cloud(self._id)


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
        interpolation_volume_mesh_id: str = None,
        tags: list[str] = None,
    ) -> DraftDraft:
        """Create a new instance of DraftDraft"""
        return DraftDraft(
            name=name,
            project_id=project_id,
            source_item_id=source_item_id,
            source_item_type=source_item_type,
            solver_version=solver_version,
            fork_case=fork_case,
            interpolation_volume_mesh_id=interpolation_volume_mesh_id,
            tags=tags,
        )

    @classmethod
    def from_cloud(cls, draft_id: IDStringType) -> Draft:
        """Load draft from cloud"""
        return Draft(draft_id=draft_id)

    def update_simulation_params(self, params):
        """update the SimulationParams of the draft"""

        @deprecation_reminder(version="25.5.4")
        def remove_none_inflow_velocity_direction_for_forward_compatibility(params_dict):
            """
            If None `velocity_direction` is found in root level of `Inflow` then
            pop the key so that forward compatibility is kept within 25.5 release.
            """
            if params_dict.get("models"):
                for idx, model in enumerate(params_dict["models"]):
                    if model.get("type") != "Inflow":
                        continue
                    if "velocity_direction" in model.keys():
                        params_dict["models"][idx].pop("velocity_direction")
            return params_dict

        params_dict = params.model_dump(exclude_none=True)
        if params_dict.get("models"):
            # Remove hybrid_model:None to avoid triggering front end display activated toggle.
            for idx, model in enumerate(params_dict["models"]):
                if (
                    model.get("turbulence_model_solver") is not None
                    and model["turbulence_model_solver"].get("hybrid_model") is None
                ):
                    params_dict["models"][idx]["turbulence_model_solver"].pop("hybrid_model", None)
                    break
        params_dict = remove_none_inflow_velocity_direction_for_forward_compatibility(
            params_dict=params_dict
        )
        self.post(
            json={"data": json.dumps(params_dict), "type": "simulation", "version": ""},
            method="simulation/file",
        )

    def get_simulation_dict(self) -> dict:
        """retrieve the SimulationParams of the draft"""
        response = self.get(method="simulation/file", params={"type": "simulation"})
        return json.loads(response["simulationJson"])

    def run_up_to_target_asset(
        self,
        target_asset: type,
        use_beta_mesher: bool,
        use_geometry_AI: bool,  # pylint: disable=invalid-name
        source_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh", "Case"],
        start_from: Union[None, Literal["SurfaceMesh", "VolumeMesh", "Case"]],
    ) -> str:
        """run the draft up to the target asset"""

        try:
            # pylint: disable=protected-access
            if use_beta_mesher is True:
                log.info("Selecting beta/in-house mesher for possible meshing tasks.")
            if use_geometry_AI is True:
                log.info("Using the Geometry AI surface mesher.")
            if start_from:
                if start_from != target_asset._cloud_resource_type_name:
                    log.info(
                        f"Force creating new resource(s) from {start_from} "
                        + f"until {target_asset._cloud_resource_type_name}"
                    )
                else:
                    log.info(f"Force creating a new {target_asset._cloud_resource_type_name}.")
            force_creation_config = (
                ForceCreationConfig(start_from=start_from) if start_from else None
            )
            run_request = DraftRunRequest(
                source_item_type=source_item_type,
                up_to=target_asset._cloud_resource_type_name,
                use_in_house=use_beta_mesher,
                use_gai=use_geometry_AI,
                force_creation_config=force_creation_config,
            )
            run_response = self.post(
                run_request.model_dump(by_alias=True),
                method="run",
            )
            destination_id = run_response["id"]
            return destination_id
        except Flow360WebError as err:
            # Error found when translating/running the simulation
            log.error(">>Submission failed.<<")
            try:
                detailed_error = json.loads(err.auxiliary_json["detail"])["detail"]
                log.error(
                    f"Failure detail: {formatting_validation_errors(ast.literal_eval(detailed_error))}"
                )
            except (json.decoder.JSONDecodeError, TypeError):
                # No detail given.
                raise Flow360RuntimeError(
                    "An unexpected error has occurred. Please contact customer support."
                ) from None
        raise RuntimeError("Submission not successful.")
