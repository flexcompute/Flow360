"""Draft for workbench realizations"""

from __future__ import annotations

import ast
import json
import time
from functools import cached_property
from typing import TYPE_CHECKING, Literal, Optional, Union

from flow360_schema.framework.entity.entity_selector import (
    collect_and_tokenize_selectors_in_place,
)
from flow360_schema.models.simulation.services_utils import (
    strip_implicit_edge_split_layers_inplace,
)
from flow360_schema.models.simulation.validation.validation_service import (
    _determine_validation_level,
)
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
from flow360.component.utils import formatting_validation_errors, validate_type
from flow360.environment import Env, current_environment
from flow360.exceptions import Flow360Error, Flow360WebError, Flow360WebTimeoutError
from flow360.log import log

if TYPE_CHECKING:
    from flow360_schema.models.simulation.simulation_params import SimulationParams


# Backend ErrorCode.FLOW360_VALIDATION_FAILS: the draft validation endpoint reports
# validation errors through this envelope code with HTTP 200 and the verdict in `detail`.
_VALIDATION_FAILS_CODE = "5000000107"

# How long to keep asking the cloud what became of a run whose acknowledgement was lost,
# and how often. Deliberately longer than the WebUI's two minutes: a script has nothing
# else to do while it waits, whereas a browser tab does.
_RUN_CONFIRMATION_TIMEOUT_SECONDS = 5 * 60
_RUN_CONFIRMATION_POLL_INTERVAL_SECONDS = 5

# Uploaded document size from which the cloud plausibly takes longer over a run request
# than the gateway in front of it waits. Same threshold as the WebUI's
# LARGE_SIMULATION_JSON_LENGTH, on the same measure — the uncompressed document.
_LARGE_SIMULATION_JSON_LENGTH = 50 * 1024 * 1024


def _error_is_relevant(error: dict, requested_levels: Optional[list]) -> bool:
    """Whether a validation error applies to the requested validation levels.

    The validation webservice always validates every level (its request carries
    no up_to), so the level gate local validation used to apply moves here:
    errors tagged with ``ctx.relevant_for`` are matched against the requested
    levels, untagged errors apply to every level.
    """
    if requested_levels is None:
        return True
    relevant_for = (error.get("ctx") or {}).get("relevant_for")
    if not relevant_for:
        return True
    return any(level in requested_levels for level in relevant_for)


class DraftMetaModel(BaseModel):
    """Draft metadata deserializer"""

    type: Literal["Draft"] = "Draft"
    name: str
    id: str
    project_id: str = Field(alias="projectId")
    solver_version: str = Field(alias="solverVersion")

    model_config = ConfigDict(extra="ignore")


def _get_run_response_target_id(run_response: dict) -> str:
    """Return the resource ID from a draft run response."""
    return run_response["id"]


def _log_run_rejection(err: Flow360WebError) -> None:
    """Report what the cloud said when it refused to run the draft."""
    # Error found when translating/running the simulation
    log.error(">>Submission error returned from cloud.<<")
    try:
        detailed_error = json.loads(err.auxiliary_json["detail"])["detail"]
        log.error(
            f"Failure detail: {formatting_validation_errors(ast.literal_eval(detailed_error))}"
        )
    except SyntaxError:
        detailed_error = json.loads(err.auxiliary_json["detail"])["detail"]
        log.error(f"Failure detail: {detailed_error}")
    except (json.decoder.JSONDecodeError, TypeError):
        # detail is not JSON — surface the raw server error
        if err.auxiliary_json:
            server_error = err.auxiliary_json.get("error", "")
            server_detail = err.auxiliary_json.get("detail", "")
            error_code = err.auxiliary_json.get("code", "")
            parts = [str(p) for p in [server_error, server_detail] if p]
            message = ": ".join(parts)
            if error_code:
                message += f" (code: {error_code})"
            log.error(f"Failure detail: {message}")


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
        draft_meta = RestApi(
            DraftInterface.endpoint, environment_provider=current_environment
        ).post(self._request.model_dump(by_alias=True))
        self._id = draft_meta["id"]
        return Draft.from_cloud(self._id)


class Draft(Flow360Resource):
    """Project Draft component"""

    # Size of the document last uploaded here. Only one large enough to outrun the gateway
    # earns the wait that resolves a lost run acknowledgement.
    _uploaded_simulation_json_length: int = 0

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
        document = json.dumps(params_dict)
        self._uploaded_simulation_json_length = len(document)

        self.post(
            json={
                "data": document,
                "type": "simulation",
                "version": "",
            },
            method="simulation/file",
            compress_when_larger_than_mb=5,
        )

    def validate_simulation_json(
        self,
        *,
        root_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh"],
        up_to: Literal["SurfaceMesh", "VolumeMesh", "Case"],
    ) -> tuple[Optional[list], list]:
        """Validate the uploaded simulation.json against this draft's solver version.

        Runs the deployed validation webservice of the draft's own solver version —
        the authoritative oracle for what the pipeline will later accept; the
        backend reads the already-uploaded document from storage, so nothing is
        re-uploaded. Returns ``(errors, warnings)`` with errors filtered to the
        requested validation levels; ``errors`` is None when the document is valid
        for those levels. Anything short of a verdict raises: the submit path
        requires the remote verdict, it never silently skips it.

        (Known server-side behavior: if the target version has no deployed
        webservice, the backend silently validates against the default solver
        version instead — invisible to this client.)
        """
        envelope = self.post_envelope(json={}, method="validation")
        data = envelope.get("data")
        if data is not None:
            return None, data.get("warnings") or []
        if envelope.get("code") != _VALIDATION_FAILS_CODE or not envelope.get("detail"):
            raise Flow360WebError(f"Draft validation did not return a verdict: {envelope}")
        verdict = json.loads(envelope["detail"])
        requested_levels = _determine_validation_level(up_to=up_to, root_item_type=root_item_type)
        errors = [
            error
            for error in verdict.get("errors") or []
            if _error_is_relevant(error, requested_levels)
        ]
        return (errors or None), verdict.get("warnings") or []

    def delete_draft(self):
        """Delete this draft.

        Used when the draft is known not to be runnable, so that iterating on
        rejected params does not leave one dead draft per attempt in the project.
        """
        self.delete()

    def exists_in_cloud(self) -> bool:
        """Whether this draft is still present in the cloud.

        Asks the lookup that answers "no such draft" with an empty result instead of
        a 404, so a missing draft is an answer rather than an error.
        """
        found = self.get(path=f"{DraftInterface.endpoint}/find", params={"id": self.id})
        return found is not None

    def confirm_run_started(self) -> bool:
        """Whether the run this draft asked for went through, when no answer ever came.

        The cloud can spend longer on a run request than the gateway in front of it is
        willing to wait, which leaves the client with nothing at all for a request the
        server may well have carried out. Only a document big enough for that to happen
        is worth waiting on — a smaller one stopped for some other reason, and the wait
        would buy the user five minutes of nothing.
        """
        if self._uploaded_simulation_json_length < _LARGE_SIMULATION_JSON_LENGTH:
            return False
        log.info(
            "The cloud has not acknowledged the submission yet. Checking whether it went "
            f"through anyway, for up to {_RUN_CONFIRMATION_TIMEOUT_SECONDS // 60} minutes."
        )
        return self.wait_until_deleted(
            timeout_seconds=_RUN_CONFIRMATION_TIMEOUT_SECONDS,
            poll_interval_seconds=_RUN_CONFIRMATION_POLL_INTERVAL_SECONDS,
        )

    def wait_until_deleted(
        self,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float,
    ) -> bool:
        """Poll until this draft is gone, returning whether it went away in time.

        The backend deletes a draft only after the run it asked for has been created,
        so a vanished draft proves the run went through — which is the only signal a
        client has when the acknowledgement of ``/run`` never arrives. A draft that is
        still there when the deadline passes means unresolved, not failed, and a lookup
        that errors out answers nothing, so polling continues either way.
        """
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                if not self.exists_in_cloud():
                    return True
            except Flow360Error as error:
                log.debug(f"Draft lookup gave no answer, still waiting: {error}")
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_interval_seconds)

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

    def run_up_to_target_asset(  # pylint:disable = too-many-locals
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
            destination_id = _get_run_response_target_id(run_response)
            return destination_id
        except Flow360WebTimeoutError:
            # No report came back from the cloud, so there is nothing to report onwards;
            # the run may well be underway. The caller resolves it against the cloud.
            raise
        except Flow360WebError as err:
            _log_run_rejection(err)
            raise

    @cached_property
    def project_id(self) -> str:
        """Get the project ID of the draft"""
        return self.info.project_id

    @property
    def web_url(self) -> str:
        """Get the web URL of the draft"""

        return Env.current.web_url + f"/workbench/{self.project_id}?id={self.id}&type=Draft"
