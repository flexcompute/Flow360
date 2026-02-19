"""Draft for workbench realizations"""

from __future__ import annotations

import ast
import json
from functools import cached_property
from typing import TYPE_CHECKING, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from flow360.cloud.flow360_requests import (
    DraftCreateRequest,
    DraftRunRequest,
    ForceCreationConfig,
    IDStringType,
)
from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import DraftInterface
from flow360.component.resource_base import Flow360Resource, ResourceDraft
from flow360.component.simulation.framework.entity_selector import (
    collect_and_tokenize_selectors_in_place,
)
from flow360.component.simulation.services_utils import (
    strip_implicit_edge_split_layers_inplace,
)
from flow360.component.utils import formatting_validation_errors, validate_type
from flow360.environment import Env
from flow360.exceptions import Flow360RuntimeError, Flow360WebError
from flow360.log import log

if TYPE_CHECKING:
    from flow360.component.simulation.simulation_params import SimulationParams


class DraftMetaModel(BaseModel):
    """Draft metadata deserializer"""

    type: Literal["Draft"] = "Draft"
    name: str
    id: str
    project_id: str = Field(alias="projectId")
    solver_version: str = Field(alias="solverVersion")

    model_config = ConfigDict(extra="ignore")


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
            meta_class=DraftMetaModel,  # We do not have dedicated meta class for Draft
            id=draft_id,
        )

    @classmethod
    # pylint: disable=protected-access
    def _from_meta(cls, meta: DraftMetaModel):
        validate_type(meta, "meta", DraftMetaModel)
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

    def update_simulation_params(self, params: SimulationParams):
        """update the SimulationParams of the draft"""
        params_dict = params.model_dump(mode="json", exclude_none=True)
        params_dict = strip_implicit_edge_split_layers_inplace(params, params_dict)
        params_dict = collect_and_tokenize_selectors_in_place(params_dict)

        self.post(
            json={
                "data": json.dumps(params_dict),
                "type": "simulation",
                "version": "",
            },
            method="simulation/file",
        )

    def activate_dependencies(self, active_draft):
        """Enable dependency resources for the draft"""

        if active_draft is None:
            return

        geometry_dependencies = [geometry.id for geometry in active_draft.imported_geometries]

        surface_mesh_dependencies = [
            surface.surface_mesh_id for surface in active_draft.imported_surfaces
        ]

        self.put(
            json={
                "geometryDependencies": geometry_dependencies,
                "surfaceMeshDependencies": surface_mesh_dependencies,
            },
            method="dependency-resource",
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
        job_type: Optional[Literal["TIME_SHARED_VGPU", "FLEX_CREDIT"]] = None,
        priority: Optional[int] = None,
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
                job_type=job_type,
                priority=priority,
            )
            request_body = run_request.model_dump(by_alias=True)
            if request_body.get("jobType") is None:
                request_body.pop("jobType", None)
            if request_body.get("priority") is None:
                request_body.pop("priority", None)
            run_response = self.post(
                request_body,
                method="run",
            )
            destination_id = run_response["id"]
            return destination_id
        except Flow360WebError as err:
            # Error found when translating/running the simulation
            log.error(">>Submission error returned from cloud.<<")
            try:
                detailed_error = json.loads(err.auxiliary_json["detail"])["detail"]
                log.error(
                    f"Failure detail: {formatting_validation_errors(ast.literal_eval(detailed_error))}"
                )
            except SyntaxError:
                # Not validation errors, likely translation error
                detailed_error = json.loads(err.auxiliary_json["detail"])["detail"]
                log.error(f"Failure detail: {detailed_error}")
            except (json.decoder.JSONDecodeError, TypeError):
                # No detail given.
                raise Flow360RuntimeError(
                    "An unexpected error has occurred. Please contact customer support."
                ) from None
        raise RuntimeError("Submission not successful.")

    @cached_property
    def project_id(self) -> str:
        """Get the project ID of the draft"""
        return self.info.project_id

    @property
    def web_url(self) -> str:
        """Get the web URL of the draft"""

        return Env.current.web_url + f"/workbench/{self.project_id}?id={self.id}&type=Draft"
