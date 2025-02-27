"""Draft for workbench realizations"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Literal, Optional

import pydantic as pd

from flow360.cloud.rest_api import RestApi
from flow360.component.interfaces import DraftInterface
from flow360.component.resource_base import (
    AssetMetaBaseModel,
    Flow360Resource,
    ResourceDraft,
)
from flow360.component.utils import is_valid_uuid, validate_type
from flow360.exceptions import Flow360WebError
from flow360.log import log


def _valid_id_validator(input_id: str):
    is_valid_uuid(input_id)
    return input_id


IDStringType = Annotated[str, pd.AfterValidator(_valid_id_validator)]


class DraftPostRequest(pd.BaseModel):
    """Data model for draft post request"""

    name: Optional[str] = pd.Field(None)
    project_id: IDStringType = pd.Field(serialization_alias="projectId")
    source_item_id: IDStringType = pd.Field(serialization_alias="sourceItemId")
    source_item_type: Literal[
        "Project", "Folder", "Geometry", "SurfaceMesh", "VolumeMesh", "Case", "Draft"
    ] = pd.Field(serialization_alias="sourceItemType")
    solver_version: str = pd.Field(serialization_alias="solverVersion")
    fork_case: bool = pd.Field(serialization_alias="forkCase")

    @pd.field_validator("name", mode="after")
    @classmethod
    def _generate_default_name(cls, values):
        if values is None:
            values = "Draft " + datetime.now().strftime("%m-%d %H:%M:%S")
        return values


class ForceCreationConfig(pd.BaseModel):
    """Data model for force creation configuration"""

    start_from: Literal["SurfaceMesh", "VolumeMesh", "Case"] = pd.Field(
        None, serialization_alias="startFrom"
    )


class DraftRunRequest(pd.BaseModel):
    """Data model for draft run request"""

    up_to: Literal["SurfaceMesh", "VolumeMesh", "Case"] = pd.Field(serialization_alias="upTo")
    use_in_house: bool = pd.Field(serialization_alias="useInHouse")
    force_creation_config: Optional[ForceCreationConfig] = pd.Field(
        None, serialization_alias="forceCreationConfig"
    )
    source_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh", "Case"] = pd.Field(
        exclude=True
    )

    @pd.model_validator(mode="after")
    def _validate_force_creation_config(self):
        # pylint: disable=no-member
        if self.force_creation_config is None:
            return self
        if (
            self.source_item_type == "SurfaceMesh"
            and self.force_creation_config.start_from not in ["VolumeMesh", "Case"]
        ) or (
            self.source_item_type in ["VolumeMesh", "Case"]
            and self.force_creation_config.start_from != "Case"
        ):
            raise ValueError(
                f"Cannot force create {self.force_creation_config.start_from} "
                f"since the project starts from {self.source_item_type}."
            )
        if (
            self.up_to == "SurfaceMesh"
            and self.force_creation_config.start_from in ["VolumeMesh", "Case"]
        ) or (self.up_to == "VolumeMesh" and self.force_creation_config.start_from == "Case"):
            raise ValueError(
                f"Cannot force create {self.force_creation_config.start_from} "
                f"since the project only runs up to {self.up_to}."
            )
        return self


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
        response = self.get(method="simulation/file", params={"type": "simulation"})
        return json.loads(response["simulationJson"])

    def run_up_to_target_asset(
        self,
        target_asset: type,
        use_beta_mesher: bool,
        source_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh", "Case"],
        start_from: Literal["SurfaceMesh", "VolumeMesh", "Case"],
    ) -> str:
        """run the draft up to the target asset"""

        try:
            # pylint: disable=protected-access
            if use_beta_mesher is True:
                log.info("Selecting beta/in-house mesher for possible meshing tasks.")
            if start_from:
                if start_from != target_asset._cloud_resource_type_name:
                    log.info(
                        f"Force creating new resouces from {start_from} until {target_asset._cloud_resource_type_name}"
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
                log.error(f"Failure detail: {detailed_error}")
            except json.decoder.JSONDecodeError:
                # No detail given.
                log.error("An unexpected error has occurred. Please contact customer support.")
        raise RuntimeError("Submission not successful.")
