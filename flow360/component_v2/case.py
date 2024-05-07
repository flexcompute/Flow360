"""
Case component
"""

from __future__ import annotations

import json
import tempfile
import time
from typing import Any, Iterator, List, Union, Optional

import pydantic as pd

from flow360.component.simulation.simulation_params import (
    SimulationParams,
    UnvalidatedSimulationParams,
)

from .. import error_messages
from ..cloud.requests import MoveCaseItem, MoveToFolderRequest
from ..cloud.rest_api import RestApi

# from .results.case_results import (
#     ActuatorDiskResultCSVModel,
#     AeroacousticsResultCSVModel,
#     BETForcesResultCSVModel,
#     CaseDownloadable,
#     CFLResultCSVModel,
#     ForceDistributionResultCSVModel,
#     LinearResidualsResultCSVModel,
#     MaxResidualLocationResultCSVModel,
#     MinMaxStateResultCSVModel,
#     MonitorsResultModel,
#     NonlinearResidualsResultCSVModel,
#     ResultBaseModel,
#     ResultsDownloaderSettings,
#     ResultTarGZModel,
#     SurfaceForcesResultCSVModel,
#     SurfaceHeatTrasferResultCSVModel,
#     TotalForcesResultCSVModel,
#     UserDefinedDynamicsResultModel,
# )
from ..component.utils import (
    is_valid_uuid,
    shared_account_confirm_proceed,
    validate_type,
)
from ..exceptions import Flow360RuntimeError, Flow360ValidationError, Flow360ValueError
from ..log import log
from .folder import Folder
from .interfaces import CaseInterface, FolderInterface, VolumeMeshInterface
from .resource_base import (
    Flow360Resource,
    Flow360ResourceBaseModel,
    Flow360ResourceListBase,
    Flow360Status,
    ResourceDraft,
    before_submit_only,
    is_object_cloud_resource,
)
from .validator import Validator


class CaseBase:
    """
    Case Base component
    """

    def copy(
        self,
        name: str = None,
        params: SimulationParams = None,
        solver_version: str = None,
        tags: List[str] = None,
    ) -> CaseDraft:
        """
        Alias for retry case
        :param name:
        :param params:
        :param tags:
        :return:
        """

        return self.retry(name, params, solver_version=solver_version, tags=tags)

    # pylint: disable=no-member
    def retry(
        self,
        name: str = None,
        params: SimulationParams = None,
        solver_version: str = None,
        tags: List[str] = None,
    ) -> CaseDraft:
        """
        Retry case
        :param name:
        :param params:
        :param tags:
        :return:
        """

        name = name or self.name or self.info.name
        params = params or self.params.copy(deep=True)
        new_case = Case.create(
            name, params, other_case=self, solver_version=solver_version, tags=tags
        )
        return new_case

    def continuation(
        self, name: str = None, params: SimulationParams = None, tags: List[str] = None
    ) -> CaseDraft:
        """
        Alias for fork a case to continue simulation
        :param name:
        :param params:
        :param tags:
        :return:
        """

        return self.fork(name, params, tags)

    # pylint: disable=no-member
    def fork(
        self, name: str = None, params: SimulationParams = None, tags: List[str] = None
    ) -> CaseDraft:
        """
        Fork a case to continue simulation
        :param name:
        :param params:
        :param tags:
        :return:
        """

        name = name or self.name or self.info.name
        params = params or self.params.copy(deep=True)
        return Case.create(name, params, parent_case=self, tags=tags)


class CaseMeta(Flow360ResourceBaseModel):
    """
    CaseMeta data component
    """

    id: str = pd.Field(alias="caseId")
    case_mesh_id: str = pd.Field(alias="caseMeshId")
    parent_id: Union[str, None] = pd.Field(alias="parentId")
    status: Flow360Status = pd.Field()

    # pylint: disable=no-self-argument
    @pd.field_validator("status")
    def set_status_type(cls, value: Flow360Status):
        """set_status_type when case uploaded"""
        if value is Flow360Status.UPLOADED:
            return Flow360Status.CASE_UPLOADED
        return value

    def to_case(self) -> Case:
        """
        returns Case object from case meta info
        """
        return Case(self.id)


# pylint: disable=too-many-instance-attributes
class CaseDraft(CaseBase, ResourceDraft):
    """
    Case Draft component (before submission)
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        name: str,
        params: SimulationParams,
        volume_mesh_id: str = None,
        tags: List[str] = None,
        parent_id: str = None,
        other_case: Case = None,
        parent_case: Case = None,
        solver_version: str = None,
    ):
        self.name = name
        self.params = params
        self.volume_mesh_id = volume_mesh_id
        self.parent_case = parent_case
        self.parent_id = parent_id
        self.other_case = other_case
        self.tags = tags
        self.solver_version = solver_version
        self._id = None
        self._submitted_case = None
        ResourceDraft.__init__(self)

        # self.validate_case_inputs()

    def __str__(self):
        return self.params.__str__()

    @property
    def params(self) -> SimulationParams:
        """
        returns case params
        """
        return self._params

    @params.setter
    def params(self, value: SimulationParams):
        """
        sets case params (before submit only)
        """
        if not isinstance(value, SimulationParams) and not isinstance(
            value, UnvalidatedSimulationParams
        ):
            raise Flow360ValueError("params are not of type SimulationParams.")
        self._params = value

    @property
    def name(self) -> str:
        """
        returns case name
        """
        return self._name

    @name.setter
    def name(self, value) -> str:
        """
        sets case name
        """
        self._name = value

    @property
    def volume_mesh_id(self):
        """
        returns volume mesh id
        """
        return self._volume_mesh_id

    @volume_mesh_id.setter
    def volume_mesh_id(self, value):
        """
        sets volume mesh id
        """
        self._volume_mesh_id = value

    def to_case(self) -> Case:
        """Return Case from CaseDraft (must be after .submit())

        Returns
        -------
        Case
            Case representation

        Raises
        ------
        RuntimeError
            Raises error when case is before submission, i.e., is in draft state
        """
        if not self.is_cloud_resource():
            raise Flow360RuntimeError(
                f"Case name={self.name} is in draft state. Run .submit() before calling this function."
            )
        return Case(self.id)

    @before_submit_only
    def submit(self, force_submit: bool = False) -> Case:
        """
        submits case to cloud for running
        """
        assert self.name
        assert self.volume_mesh_id or self.other_case or self.parent_id or self.parent_case
        assert self.params

        # self.validate_case_inputs(pre_submit_checks=True)

        if not shared_account_confirm_proceed():
            raise Flow360ValueError("User aborted resource submit.")

        volume_mesh_id = self.volume_mesh_id
        parent_id = self.parent_id
        if parent_id is not None:
            self.parent_case = Case(self.parent_id)

        if isinstance(self.parent_case, CaseDraft):
            self.parent_case = self.parent_case.to_case()

        if isinstance(self.other_case, CaseDraft):
            self.other_case = self.other_case.to_case()

        if self.other_case is not None and self.other_case.has_parent():
            self.parent_case = self.other_case.parent

        if self.parent_case is not None:
            parent_id = self.parent_case.id
            volume_mesh_id = self.parent_case.volume_mesh_id

            if (
                self.solver_version is not None
                and self.parent_case.solver_version != self.solver_version
            ):
                raise Flow360RuntimeError(
                    error_messages.change_solver_version_error(
                        self.parent_case.solver_version, self.solver_version
                    )
                )
            self.solver_version = self.parent_case.solver_version

        volume_mesh_id = volume_mesh_id or self.other_case.volume_mesh_id

        if self.solver_version is None:
            volume_mesh_info = Flow360ResourceBaseModel(
                **RestApi(VolumeMeshInterface.endpoint, id=volume_mesh_id).get()
            )
            self.solver_version = volume_mesh_info.solver_version

        is_valid_uuid(volume_mesh_id)
        # self.validator_api(
        #     self.params,
        #     volume_mesh_id=volume_mesh_id,
        #     solver_version=self.solver_version,
        #     raise_on_error=(not force_submit),
        # )

        data = {
            "name": self.name,
            "meshId": volume_mesh_id,
            "runtimeParams": self.params.model_dump_json(),
            "tags": self.tags,
            "parentId": parent_id,
        }

        if self.solver_version is not None:
            data["solverVersion"] = self.solver_version

        # Note: So the SimulationParam -->  Flow360Param is supposed to happen in case editor service but we have not implement that yet. Now we just do the transformation manually
        from flow360.component.simulation.translator.translator import (
            convert_SimulationParams_to_Flow360Params,
        )

        data["runtimeParams"] = convert_SimulationParams_to_Flow360Params(self.params).json()
        ##:: Done

        resp = RestApi(CaseInterface.endpoint).post(
            json=data,
            path=f"volumemeshes/{volume_mesh_id}/case",
        )
        info = CaseMeta(**resp)
        self._id = info.id

        self._submitted_case = Case(self.id)
        log.info(f"Case successfully submitted: {self._submitted_case.short_description()}")
        return self._submitted_case

    def validate_case_inputs(self, pre_submit_checks=False):
        """
        validates case inputs (before submit only)
        """
        if self.volume_mesh_id is not None and self.other_case is not None:
            raise Flow360ValueError("You cannot specify both volume_mesh_id AND other_case.")

        if self.parent_id is not None and self.parent_case is not None:
            raise Flow360ValueError("You cannot specify both parent_id AND parent_case.")

        if self.parent_id is not None or self.parent_case is not None:
            if self.volume_mesh_id is not None or self.other_case is not None:
                raise Flow360ValueError(
                    "You cannot specify volume_mesh_id OR other_case when parent case provided."
                )

        is_valid_uuid(self.volume_mesh_id, allow_none=True)

        if pre_submit_checks:
            is_object_cloud_resource(self.other_case)
            is_object_cloud_resource(self.parent_case)

    @classmethod
    def validator_api(
        cls,
        params: SimulationParams,
        volume_mesh_id,
        solver_version: str = None,
        raise_on_error: bool = True,
    ):
        """
        validation api: validates case parameters before submitting
        """
        return Validator.CASE.validate(
            params,
            mesh_id=volume_mesh_id,
            solver_version=solver_version,
            raise_on_error=raise_on_error,
        )


# pylint: disable=too-many-instance-attributes
class Case(CaseBase, Flow360Resource):
    """
    Case component
    """

    case_draft: Optional[CaseDraft] = None

    # pylint: disable=redefined-builtin
    def __init__(
        self,
        id: str = None,
        name: str = None,
        params: SimulationParams = None,
        volume_mesh_id: str = None,
        tags: List[str] = None,
        parent_id: str = None,
        other_case: Case = None,
        parent_case: Case = None,
        solver_version: str = None,
    ):
        if id is not None:
            assert name is None
            super().__init__(
                interface=CaseInterface,
                info_type_class=CaseMeta,
                id=id,
            )

            self._params = None
            self._raw_params = None
        else:
            self.case_draft = CaseDraft(
                name=name,
                volume_mesh_id=volume_mesh_id,
                params=params.copy(),
                parent_id=parent_id,
                other_case=other_case,
                parent_case=parent_case,
                tags=tags,
                solver_version=solver_version,
            )

        # self._results = CaseResultsModel(case=self)

    @classmethod
    def _from_meta(cls, meta: CaseMeta):
        validate_type(meta, "meta", CaseMeta)
        case = cls(id=meta.id)
        case._set_meta(meta)
        return case

    @property
    def params(self) -> SimulationParams:
        """
        returns case params
        """
        if self._params is None:
            self._raw_params = json.loads(self.get(method="runtimeParams")["content"])
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as temp_file:
                    json.dump(self._raw_params, temp_file)

                self._params = SimulationParams(temp_file.name)
            except pd.ValidationError as err:
                raise Flow360ValidationError(error_messages.params_fetching_error(err)) from err

        return self._params

    @property
    def params_as_dict(self) -> dict:
        """
        returns case params as dictionary
        """
        if self._raw_params is None:
            self._raw_params = json.loads(self.get(method="runtimeParams")["content"])
        return self._raw_params

    def has_parent(self) -> bool:
        """Check if case has parent case

        Returns
        -------
        bool
            True when case has parent, False otherwise
        """
        return self.info.parent_id is not None

    @property
    def parent(self) -> Case:
        """parent case

        Returns
        -------
        Case
            parent case object

        Raises
        ------
        RuntimeError
            When case does not have parent
        """
        if self.has_parent():
            return Case(self.info.parent_id)
        raise Flow360RuntimeError("Case does not have parent case.")

    @property
    def info(self) -> CaseMeta:
        """
        returns metadata info for case
        """
        return super().info

    @property
    def volume_mesh_id(self):
        """
        returns volume mesh id
        """
        return self.info.case_mesh_id

    # @property
    # def results(self) -> CaseResultsModel:
    #     """
    #     returns results object to managing case results
    #     """
    #     return self._results

    def is_finished(self):
        """
        returns False when case is in running or preprocessing state
        """
        return self.status.is_final()

    def move_to_folder(self, folder: Folder):
        """
        Move the current case to the specified folder.

        Parameters
        ----------
        folder : Folder
            The destination folder where the item will be moved.

        Returns
        -------
        self
            Returns the modified item after it has been moved to the new folder.

        Notes
        -----
        This method sends a REST API request to move the current item to the specified folder.
        The `folder` parameter should be an instance of the `Folder` class with a valid ID.
        """
        RestApi(FolderInterface.endpoint).put(
            MoveToFolderRequest(dest_folder_id=folder.id, items=[MoveCaseItem(id=self.id)]).dict(),
            method="move",
        )
        return self

    @classmethod
    def _interface(cls):
        return CaseInterface

    @classmethod
    def _meta_class(cls):
        """
        returns case meta info class: CaseMeta
        """
        return CaseMeta

    @classmethod
    def _params_ancestor_id_name(cls):
        """
        returns volumeMeshId name
        """
        return "meshId"

    @classmethod
    def from_cloud(cls, case_id: str):
        """
        get case from cloud
        """
        return cls(case_id)

    # pylint: disable=too-many-arguments
    @classmethod
    def create(
        cls,
        name: str,
        params: SimulationParams,
        volume_mesh_id: str = None,
        tags: List[str] = None,
        parent_id: str = None,
        other_case: Case = None,
        parent_case: Case = None,
        solver_version: str = None,
    ):
        """
        Create new case
        :param name:
        :param params:
        :param volume_mesh_id:
        :param other_case:
        :param tags:
        :param parent_id:
        :param parent_case:
        :return:
        """

        assert name
        assert volume_mesh_id or other_case or parent_id or parent_case
        assert params

        if not isinstance(params, SimulationParams) and not isinstance(
            params, UnvalidatedSimulationParams
        ):
            raise Flow360ValueError("params are not of type SimulationParams.")

        return cls(
            name=name,
            volume_mesh_id=volume_mesh_id,
            params=params.copy(),
            parent_id=parent_id,
            other_case=other_case,
            parent_case=parent_case,
            tags=tags,
            solver_version=solver_version,
        )

    def wait(self, timeout_minutes=60):
        """Wait until the Case finishes processing, refresh periodically"""

        start_time = time.time()
        while self.is_finished() is False:
            if time.time() - start_time > timeout_minutes * 60:
                raise TimeoutError(
                    "Timeout: Process did not finish within the specified timeout period"
                )
            time.sleep(2)
