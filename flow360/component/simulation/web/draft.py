"""Draft for workbench realizations"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Literal

import pydantic as pd

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import DraftInterface, ProjectInterface
from flow360.component.resource_base import (
    AssetMetaBaseModel,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.simulation.web.asset_base import AssetBase
from flow360.component.utils import is_valid_uuid, validate_type
from flow360.exceptions import Flow360WebError
from flow360.log import log


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

    def update_simulation_params(self, params):
        """update the SimulationParams of the draft"""

        self.post(
            json={"data": params.model_dump_json(), "type": "simulation", "version": ""},
            method="simulation/file",
        )

    def get_simulation_dict(self) -> dict:
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


def _get_simulation_json_from_cloud(project_id: str) -> dict:
    _resp = RestApi(ProjectInterface.endpoint, id=project_id).get()
    last_opened_draft_id = _resp["lastOpenDraftId"]
    assert last_opened_draft_id is not None
    _draft = Draft.from_cloud(last_opened_draft_id)

    simulation_dict = _draft.get_simulation_dict()
    return simulation_dict