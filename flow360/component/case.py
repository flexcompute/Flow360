"""
Case component
"""

from __future__ import annotations

import json
import tempfile
import time
from typing import Any, Iterator, List, Union

import pydantic as pd

from .. import error_messages
from ..cloud.requests import MoveCaseItem, MoveToFolderRequest
from ..cloud.rest_api import RestApi
from ..exceptions import Flow360RuntimeError, Flow360ValidationError, Flow360ValueError
from ..log import log
from .flow360_params.flow360_params import Flow360Params, UnvalidatedFlow360Params
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
from .results.case_results import (
    ActuatorDiskResultCSVModel,
    AeroacousticsResultCSVModel,
    BETForcesResultCSVModel,
    CaseDownloadable,
    CFLResultCSVModel,
    ForceDistributionResultCSVModel,
    LinearResidualsResultCSVModel,
    MaxResidualLocationResultCSVModel,
    MinMaxStateResultCSVModel,
    MonitorsResultModel,
    NonlinearResidualsResultCSVModel,
    ResultBaseModel,
    ResultsDownloaderSettings,
    ResultTarGZModel,
    SurfaceForcesResultCSVModel,
    SurfaceHeatTrasferResultCSVModel,
    TotalForcesResultCSVModel,
    UserDefinedDynamicsResultModel,
)
from .utils import is_valid_uuid, shared_account_confirm_proceed, validate_type
from .validator import Validator


class CaseBase:
    """
    Case Base component
    """

    def copy(
        self,
        name: str = None,
        params: Flow360Params = None,
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
        params: Flow360Params = None,
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
        self, name: str = None, params: Flow360Params = None, tags: List[str] = None
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
        self, name: str = None, params: Flow360Params = None, tags: List[str] = None
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
    @pd.validator("status")
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
        params: Flow360Params,
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

        self.validate_case_inputs()

    def __str__(self):
        return self.params.__str__()

    @property
    def params(self) -> Flow360Params:
        """
        returns case params
        """
        return self._params

    @params.setter
    def params(self, value: Flow360Params):
        """
        sets case params (before submit only)
        """
        if not isinstance(value, Flow360Params) and not isinstance(value, UnvalidatedFlow360Params):
            raise Flow360ValueError("params are not of type Flow360Params.")
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

        self.validate_case_inputs(pre_submit_checks=True)

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
        self.validator_api(
            self.params,
            volume_mesh_id=volume_mesh_id,
            solver_version=self.solver_version,
            raise_on_error=(not force_submit),
        )

        data = {
            "name": self.name,
            "meshId": volume_mesh_id,
            "runtimeParams": self.params.flow360_json(),
            "tags": self.tags,
            "parentId": parent_id,
        }

        if self.solver_version is not None:
            data["solverVersion"] = self.solver_version

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
        params: Flow360Params,
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

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        super().__init__(
            interface=CaseInterface,
            info_type_class=CaseMeta,
            id=id,
        )

        self._params = None
        self._raw_params = None
        self._results = CaseResultsModel(case=self)

    @classmethod
    def _from_meta(cls, meta: CaseMeta):
        validate_type(meta, "meta", CaseMeta)
        case = cls(id=meta.id)
        case._set_meta(meta)
        return case

    @property
    def params(self) -> Flow360Params:
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

                self._params = Flow360Params(temp_file.name)
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

    @property
    def results(self) -> CaseResultsModel:
        """
        returns results object to managing case results
        """
        return self._results

    def is_steady(self):
        """
        returns True when case is steady state
        """
        return self.params.time_stepping.time_step_size == "inf"

    def has_actuator_disks(self):
        """
        returns True when case has actuator disk
        """
        if self.params.actuator_disks is not None:
            if len(self.params.actuator_disks) > 0:
                return True
        return False

    def has_bet_disks(self):
        """
        returns True when case has BET disk
        """
        if self.params.bet_disks is not None:
            if len(self.params.bet_disks) > 0:
                return True
        return False

    def has_isosurfaces(self):
        """
        returns True when case has isosurfaces
        """
        return self.params.iso_surface_output is not None

    def has_monitors(self):
        """
        returns True when case has monitors
        """
        return self.params.monitor_output is not None

    def has_aeroacoustics(self):
        """
        returns True when case has aeroacoustics
        """
        return self.params.aeroacoustic_output is not None

    def has_user_defined_dynamics(self):
        """
        returns True when case has user defined dynamics
        """
        return self.params.user_defined_dynamics is not None

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
        params: Flow360Params,
        volume_mesh_id: str = None,
        tags: List[str] = None,
        parent_id: str = None,
        other_case: Case = None,
        parent_case: Case = None,
        solver_version: str = None,
    ) -> CaseDraft:
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

        if not isinstance(params, Flow360Params) and not isinstance(
            params, UnvalidatedFlow360Params
        ):
            raise Flow360ValueError("params are not of type Flow360Params.")

        new_case = CaseDraft(
            name=name,
            volume_mesh_id=volume_mesh_id,
            params=params.copy(),
            parent_id=parent_id,
            other_case=other_case,
            parent_case=parent_case,
            tags=tags,
            solver_version=solver_version,
        )
        return new_case

    def wait(self, timeout_minutes=60):
        """Wait until the Case finishes processing, refresh periodically"""

        start_time = time.time()
        while self.is_finished() is False:
            if time.time() - start_time > timeout_minutes * 60:
                raise TimeoutError(
                    "Timeout: Process did not finish within the specified timeout period"
                )
            time.sleep(2)


# pylint: disable=unnecessary-lambda
class CaseResultsModel(pd.BaseModel):
    """
    Pydantic models for case results
    """

    case: Any = pd.Field()

    # tar.gz results:
    surfaces: ResultTarGZModel = pd.Field(
        default_factory=lambda: ResultTarGZModel(remote_file_name=CaseDownloadable.SURFACES.value)
    )
    volumes: ResultTarGZModel = pd.Field(
        default_factory=lambda: ResultTarGZModel(remote_file_name=CaseDownloadable.VOLUMES.value)
    )
    slices: ResultTarGZModel = pd.Field(
        default_factory=lambda: ResultTarGZModel(remote_file_name=CaseDownloadable.SLICES.value)
    )
    isosurfaces: ResultTarGZModel = pd.Field(
        default_factory=lambda: ResultTarGZModel(
            remote_file_name=CaseDownloadable.ISOSURFACES.value
        )
    )
    monitors: MonitorsResultModel = pd.Field(MonitorsResultModel())

    # convergence:
    nonlinear_residuals: NonlinearResidualsResultCSVModel = pd.Field(
        default_factory=lambda: NonlinearResidualsResultCSVModel()
    )
    linear_residuals: LinearResidualsResultCSVModel = pd.Field(
        default_factory=lambda: LinearResidualsResultCSVModel()
    )
    cfl: CFLResultCSVModel = pd.Field(default_factory=lambda: CFLResultCSVModel())
    minmax_state: MinMaxStateResultCSVModel = pd.Field(
        default_factory=lambda: MinMaxStateResultCSVModel()
    )
    max_residual_location: MaxResidualLocationResultCSVModel = pd.Field(
        default_factory=lambda: MaxResidualLocationResultCSVModel()
    )

    # forces
    total_forces: TotalForcesResultCSVModel = pd.Field(
        default_factory=lambda: TotalForcesResultCSVModel()
    )
    surface_forces: SurfaceForcesResultCSVModel = pd.Field(
        default_factory=lambda: SurfaceForcesResultCSVModel()
    )
    actuator_disks: ActuatorDiskResultCSVModel = pd.Field(
        default_factory=lambda: ActuatorDiskResultCSVModel()
    )
    bet_forces: BETForcesResultCSVModel = pd.Field(
        default_factory=lambda: BETForcesResultCSVModel()
    )
    force_distribution: ForceDistributionResultCSVModel = pd.Field(
        default_factory=lambda: ForceDistributionResultCSVModel()
    )

    # user defined:
    user_defined_dynamics: UserDefinedDynamicsResultModel = pd.Field(
        default_factory=lambda: UserDefinedDynamicsResultModel()
    )

    # others
    surface_heat_transfer: SurfaceHeatTrasferResultCSVModel = pd.Field(
        default_factory=lambda: SurfaceHeatTrasferResultCSVModel()
    )
    aeroacoustics: AeroacousticsResultCSVModel = pd.Field(
        default_factory=lambda: AeroacousticsResultCSVModel()
    )

    _downloader_settings: ResultsDownloaderSettings = pd.PrivateAttr(ResultsDownloaderSettings())

    # pylint: disable=no-self-argument, protected-access
    @pd.root_validator(pre=False)
    def pass_download_function(cls, values):
        """
        Pass download methods into fields of the case results
        """
        if "case" not in values:
            raise ValueError("case (type Case) is required")

        if not isinstance(values["case"], Case):
            raise TypeError("case must be of type Case")

        for field in cls.__fields__.values():
            if field.name in values.keys():
                value = values[field.name]
                if isinstance(value, ResultBaseModel):
                    value._download_method = values["case"]._download_file
                    value._get_params_method = lambda: values["case"].params

                    values[field.name] = value

        return values

    # pylint: disable=no-self-argument, protected-access
    @pd.validator("monitors", "user_defined_dynamics", always=True)
    def pass_get_files_function(cls, value, values):
        """
        Pass file getters into fields of the case results
        """
        value.get_download_file_list_method = values["case"].get_download_file_list
        return value

    # pylint: disable=no-self-argument, protected-access
    @pd.validator("bet_forces", always=True)
    def pass_has_bet_forces_function(cls, value, values):
        """
        Pass check to see if result is downloadable based on params
        """
        value._is_downloadable = values["case"].has_bet_disks
        return value

    # pylint: disable=no-self-argument, protected-access
    @pd.validator("actuator_disks", always=True)
    def pass_has_actuator_disks_function(cls, value, values):
        """
        Pass check to see if result is downloadable based on params
        """
        value._is_downloadable = values["case"].has_actuator_disks
        return value

    # pylint: disable=no-self-argument, protected-access
    @pd.validator("isosurfaces", always=True)
    def pass_has_isosurfaces_function(cls, value, values):
        """
        Pass check to see if result is downloadable based on params
        """
        value._is_downloadable = values["case"].has_isosurfaces
        return value

    # pylint: disable=no-self-argument, protected-access
    @pd.validator("monitors", always=True)
    def pass_has_monitors_function(cls, value, values):
        """
        Pass check to see if result is downloadable based on params
        """
        value._is_downloadable = values["case"].has_monitors
        return value

    # pylint: disable=no-self-argument, protected-access
    @pd.validator("aeroacoustics", always=True)
    def pass_has_aeroacoustics_function(cls, value, values):
        """
        Pass check to see if result is downloadable based on params
        """
        value._is_downloadable = values["case"].has_aeroacoustics
        return value

    # pylint: disable=no-self-argument, protected-access
    @pd.validator("user_defined_dynamics", always=True)
    def pass_has_user_defined_dynamics_function(cls, value, values):
        """
        Pass check to see if result is downloadable based on params
        """
        value._is_downloadable = values["case"].has_user_defined_dynamics
        return value

    def _execute_downloading(self):
        """
        Download all specified and available results for the case
        """
        for _, value in self.__dict__.items():
            if isinstance(value, ResultBaseModel):
                # we download if explicitly set set_downloader(<result_name>=True),
                # or all=True but only when is not result=False
                try_download = value.do_download is True
                if self._downloader_settings.all is True and value.do_download is not False:
                    try_download = value._is_downloadable() is True
                if try_download is True:
                    value.download(
                        to_folder=self._downloader_settings.destination,
                        overwrite=self._downloader_settings.overwrite,
                    )

    def set_destination(
        self, folder_name: str = None, use_case_name: bool = None, use_case_id: bool = None
    ):
        """
        Set the destination for downloading files.

        Parameters
        ----------
        folder_name : str, optional
            The name of the folder where files will be downloaded.
        use_case_name : bool, optional
            Whether to use the use case name for the destination.
        use_case_id : bool, optional
            Whether to use the use case ID for the destination.

        Raises
        ------
        ValueError
            If more than one argument is provided or if no arguments are provided.

        """
        # Check if only one argument is provided
        if sum(arg is not None for arg in [folder_name, use_case_name, use_case_id]) != 1:
            raise ValueError("Exactly one argument should be provided.")

        if folder_name is not None:
            self._downloader_settings.destination = folder_name
        if use_case_name is True:
            self._downloader_settings.destination = self.case.name
        if use_case_id is True:
            self._downloader_settings.destination = self.case.id

    # pylint: disable=too-many-arguments, too-many-locals, redefined-builtin
    def download(
        self,
        surface: bool = None,
        volume: bool = None,
        slices: bool = None,
        isosurfaces: bool = None,
        monitors: bool = None,
        nonlinear_residuals: bool = None,
        linear_residuals: bool = None,
        cfl: bool = None,
        minmax_state: bool = None,
        max_residual_location: bool = None,
        surface_forces: bool = None,
        total_forces: bool = None,
        bet_forces: bool = None,
        actuator_disks: bool = None,
        force_distribution: bool = None,
        user_defined_dynamics: bool = None,
        aeroacoustics: bool = None,
        surface_heat_transfer: bool = None,
        all: bool = None,
        overwrite: bool = False,
        destination: str = None,
    ):
        """
        Download result files associated with the case.

        Parameters
        ----------
        surface : bool, optional
            Download surface result file if True.
        volume : bool, optional
            Download volume result file if True.
        nonlinear_residuals : bool, optional
            Download nonlinear residuals file if True.
        linear_residuals : bool, optional
            Download linear residuals file if True.
        cfl : bool, optional
            Download CFL file if True.
        minmax_state : bool, optional
            Download minmax state file if True.
        surface_forces : bool, optional
            Download surface forces file if True.
        total_forces : bool, optional
            Download total forces file if True.
        bet_forces : bool, optional
            Download BET (Blade Element Theory) forces file if True.
        actuator_disk_output : bool, optional
            Download actuator disk output file if True.
        all : bool, optional
            Download all result files if True. Ignore file if explicitly set: <result_name>=False
        overwrite : bool, optional
            If True, overwrite existing files with the same name in the destination.
        destination : str, optional
            Location to save downloaded files. If None, files will be saved in the current directory under ID folder.
        """

        self.surfaces.do_download = surface
        self.volumes.do_download = volume
        self.slices.do_download = slices
        self.isosurfaces.do_download = isosurfaces
        self.monitors.do_download = monitors

        self.nonlinear_residuals.do_download = nonlinear_residuals
        self.linear_residuals.do_download = linear_residuals
        self.cfl.do_download = cfl
        self.minmax_state.do_download = minmax_state
        self.max_residual_location.do_download = max_residual_location

        self.surface_forces.do_download = surface_forces
        self.total_forces.do_download = total_forces
        self.bet_forces.do_download = bet_forces
        self.actuator_disks.do_download = actuator_disks
        self.force_distribution.do_download = force_distribution

        self.user_defined_dynamics.do_download = user_defined_dynamics
        self.aeroacoustics.do_download = aeroacoustics
        self.surface_heat_transfer.do_download = surface_heat_transfer

        self._downloader_settings.all = all
        self._downloader_settings.overwrite = overwrite
        if destination is not None:
            self.set_destination(folder_name=destination)

        self._execute_downloading()

    def download_file_by_name(self, file_name, to_file=None, to_folder=".", overwrite: bool = True):
        """
        Download file by name
        """
        return self.case._download_file(
            file_name=file_name, to_file=to_file, to_folder=to_folder, overwrite=overwrite
        )


class CaseList(Flow360ResourceListBase):
    """
    Case List component
    """

    def __init__(
        self, mesh_id: str = None, from_cloud: bool = True, include_deleted: bool = False, limit=100
    ):
        super().__init__(
            ancestor_id=mesh_id,
            from_cloud=from_cloud,
            include_deleted=include_deleted,
            limit=limit,
            resourceClass=Case,
        )

    def filter(self):
        """
        flitering list, not implemented yet
        """
        raise NotImplementedError("Filters are not implemented yet")
        # resp = list(filter(lambda i: i['caseStatus'] != 'deleted', resp))

    # pylint: disable=useless-parent-delegation
    def __getitem__(self, index) -> Case:
        """
        returns CaseMeta info item of the list
        """
        return super().__getitem__(index)

    def __iter__(self) -> Iterator[Case]:
        return super().__iter__()
