"""
Case component
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Iterator, List, Union

import pydantic as pd
from pylab import show, subplots

from .. import error_messages
from ..cloud.rest_api import RestApi
from ..cloud.s3_utils import CloudFileNotFoundError, S3TransferType
from ..exceptions import RuntimeError as FlRuntimeError
from ..exceptions import ValidationError
from ..exceptions import ValueError as FlValueError
from ..log import log
from .flow360_params.flow360_params import Flow360Params, UnvalidatedFlow360Params
from .resource_base import (
    Flow360Resource,
    Flow360ResourceBaseModel,
    Flow360ResourceListBase,
    Flow360Status,
    ResourceDraft,
    before_submit_only,
    is_object_cloud_resource,
)
from .utils import is_valid_uuid, validate_type
from .validator import Validator


class CaseBase:
    """
    Case Base component
    """

    _endpoint = "cases"

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
            raise FlValueError("params are not of type Flow360Params.")
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
            raise FlRuntimeError(
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
                raise FlRuntimeError(
                    error_messages.change_solver_version_error(
                        self.parent_case.solver_version, self.solver_version
                    )
                )

        volume_mesh_id = volume_mesh_id or self.other_case.volume_mesh_id

        is_valid_uuid(volume_mesh_id)
        is_valid_uuid(parent_id, ignore_none=True)
        self.validator_api(
            self.params,
            volume_mesh_id=volume_mesh_id,
            solver_version=self.solver_version,
            raise_on_error=(not force_submit),
        )

        data = {
            "name": self.name,
            "meshId": volume_mesh_id,
            "runtimeParams": self.params.to_flow360_json(),
            "tags": self.tags,
            "parentId": parent_id,
        }

        if self.solver_version is not None:
            data["solverVersion"] = self.solver_version

        resp = RestApi(self._endpoint).post(
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
            raise FlValueError("You cannot specify both volume_mesh_id AND other_case.")

        if self.parent_id is not None and self.parent_case is not None:
            raise FlValueError("You cannot specify both parent_id AND parent_case.")

        if self.parent_id is not None or self.parent_case is not None:
            if self.volume_mesh_id is not None or self.other_case is not None:
                raise FlValueError(
                    "You cannot specify volume_mesh_id OR other_case when parent case provided."
                )

        is_valid_uuid(self.volume_mesh_id, ignore_none=True)

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
            resource_type="Case",
            info_type_class=CaseMeta,
            s3_transfer_method=S3TransferType.CASE,
            endpoint=self._endpoint,
            id=id,
        )

        self._params = None
        self._results = CaseResults(self)

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
            raw_params = json.loads(self.get(method="runtimeParams")["content"])
            try:
                self._params = Flow360Params(**raw_params)
            except pd.ValidationError as err:
                if self.status is Flow360Status.ERROR:
                    log.error(f"{err}")
                    self._params = raw_params
                else:
                    raise ValidationError(f"{err}") from err

        return self._params

    def has_parent(self) -> bool:
        """Check if case has parent case

        Returns
        -------
        bool
            True when case has parent, False otherwise
        """
        print(f"has_parent {self.info.parent_id} (type={type(self.info.parent_id)})")
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
        raise FlRuntimeError("Case does not have parent case.")

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
    def results(self) -> CaseResults:
        """
        returns results object to managing case results
        """
        return self._results

    def download_log(self, log_file, to_file=".", keep_folder: bool = True):
        """
        Download log
        :param log_file:
        :param to_file: file name on local disk, could be either folder or file name.
        :param keep_folder: If true, the downloaded file will be put in the same folder as the file on cloud. Only work
        when file_name is a folder name.
        :return:
        """

        self.download_file(f"logs/{log_file.value}", to_file, keep_folder)

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

    def is_finished(self):
        """
        returns False when case is in running or preprocessing state
        """
        return self.status.is_final()

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
            raise FlValueError("params are not of type Flow360Params.")

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


class CaseResultType(Enum):
    """
    Case results types
    """

    NONLINEAR_RESIDUALS = "nonlinear_residual_v2"
    TOTAL_FORCES = "total_forces_v2"
    LINEAR_RESIDUALS = "linear_residual_v2"
    MINMAX_STATE = "minmax_state_v2"
    CFL = "cfl_v2"


class CaseDownloadable(Enum):
    """
    Case results filenames
    """

    VOLUME = "volumes.tar.gz"
    SURFACE = "surfaces.tar.gz"
    NONLINEAR_RESIDUALS = "nonlinear_residual_v2.csv"
    LINEAR_RESIDUALS = "linear_residual_v2.csv"
    CFL = "cfl_v2.csv"
    MINMAX_STATE = "minmax_state_v2.csv"
    SURFACE_FORCES = "surface_forces_v2.csv"
    TOTAL_FORCES = "total_forces_v2.csv"
    BET_FORCES = "bet_forces_v2.csv"
    ACTUATOR_DISK_OUTPUT = "actuatorDisk_output_v2.csv"


class CacheableData:
    """
    Cacheable data class, fetches data from server only once and stores in memory unless "force" is used
    """

    def __init__(self, get_method, path):
        self.get_method = get_method
        self.path = path
        self.data = None

    def get(self, force=False) -> CacheableData:
        """fetch data from server

        Parameters
        ----------
        force : bool, optional
            whether to ignore cached data and force fetching data from server, by default False

        Returns
        -------
        CacheableData
            self
        """
        if self.data is None or force:
            self.data = self.get_method(method=self.path)["csvOutput"]
        return self

    @property
    def raw(self):
        """
        returns data in raw format: dictionary of lists
        """
        return self.data

    def plot(self):
        """
        plotting not implemented yet
        """
        raise NotImplementedError("Plotting is not implemented yet.")

    def to_csv(self):
        """
        saving to csv not implemented yet
        """
        raise NotImplementedError("Saving to CSV is not implemented yet.")


class ResultsPloter:
    """
    Collection of plotting functions for case results
    """

    def __init__(self, results: CaseResults):
        self.results = results

    # pylint: disable=protected-access
    def total_forces(self):
        """plot total forces"""
        steady = self.results._case.is_steady()
        forces = self.results.total_forces.raw
        if steady:
            _, ax1 = subplots()

            ax2 = ax1.twinx()
            ax1.plot(forces["pseudo_step"], forces["CL"], "-g")
            ax2.plot(forces["pseudo_step"], forces["CD"], "-b")
            ax1.set_xlabel("pseudo step")
            ax1.set_ylabel("CL", color="g")
            ax2.set_ylabel("CD", color="b")
            show()
        else:
            raise NotImplementedError("Plotting unsteady not supported yet.")

    def residuals(self):
        """plot residuals"""
        raise NotImplementedError("Plotting residuals not supported yet.")


class CaseResults:
    """
    Case results class for managing results: viewing, downloading, postprocessing
    """

    def __init__(self, case: Case):
        self._case = case
        self._residuals = CacheableData(
            self._case.get, self._get_result_path(CaseResultType.NONLINEAR_RESIDUALS)
        )
        self._total_forces = CacheableData(
            self._case.get, self._get_result_path(CaseResultType.TOTAL_FORCES)
        )
        self._linear_residuals = CacheableData(
            self._case.get, self._get_result_path(CaseResultType.LINEAR_RESIDUALS)
        )
        self._minmax_state = CacheableData(
            self._case.get, self._get_result_path(CaseResultType.MINMAX_STATE)
        )
        self._cfl = CacheableData(self._case.get, self._get_result_path(CaseResultType.CFL))
        self._plotter = ResultsPloter(self)

    def _get_result_path(self, result_type: CaseResultType):
        return f"results/v2/{result_type.value}.csv"

    def get_residuals(self, force: bool = False):
        """
        Returns residuals
        :param force: when True, fetches data from server, otherwise uses cached data if exist
        """
        return self._residuals.get(force=force)

    @property
    def residuals(self):
        """
        Returns residuals
        """
        return self.get_residuals()

    def get_total_forces(self, force: bool = False):
        """
        Returns total forces
        :param force: when True, fetches data from server, otherwise uses cached data if exist
        """
        return self._total_forces.get(force=force)

    @property
    def total_forces(self):
        """
        Returns total forces
        """
        return self.get_total_forces()

    def get_linear_residuals(self, force: bool = False):
        """
        Returns linear residuals
        :param force: when True, fetches data from server, otherwise uses cached data if exist
        """
        return self._linear_residuals.get(force=force)

    @property
    def linear_residuals(self):
        """
        Returns linear residuals
        """
        return self.get_linear_residuals()

    def get_minmax_state(self, force: bool = False):
        """
        Returns min/max state: min pressure, min density and max velocity magnitude
        :param force: when True, fetches data from server, otherwise uses cached data if exist
        """
        return self._minmax_state.get(force=force)

    @property
    def minmax_state(self):
        """
        Returns min/max state: min pressure, min density and max velocity magnitude
        """
        return self.get_minmax_state()

    def get_cfl(self, force: bool = False):
        """
        Returns cfl
        :param force: when True, fetches data from server, otherwise uses cached data if exist
        """
        return self._cfl.get(force=force)

    @property
    def cfl(self):
        """
        Returns cfl
        """
        return self.get_cfl()

    @property
    def plot(self):
        """
        plotter which manages plotting functions
        """
        return self._plotter

    def download_file(self, downloadable: CaseDownloadable, overwrite: bool = True, **kwargs):
        """
        download specific file by filename
        :param downloadable: filename to download
        :param overwrite: when True, overwrites existing file, otherwise skip
        """
        return self._case.download_file(
            f"results/{downloadable.value}", overwrite=overwrite, **kwargs
        )

    def download_volumetric(self):
        """
        download volumetric results data
        """
        self.download_file(CaseDownloadable.VOLUME)

    def download_surface(self):
        """
        download surface results data
        """
        self.download_file(CaseDownloadable.SURFACE)

    # pylint: disable=redefined-builtin,too-many-locals,too-many-arguments
    def download_manager(
        self,
        surface: bool = False,
        volume: bool = False,
        nonlinear_residuals: bool = False,
        linear_residuals: bool = False,
        cfl: bool = False,
        minmax_state: bool = False,
        surface_forces: bool = False,
        total_forces: bool = False,
        bet_forces: bool = False,
        actuator_disk_output: bool = False,
        all: bool = False,
        overwrite: bool = False,
    ):
        """download manager for downloading many files at once

        Parameters
        ----------
        surface : bool, optional
            _description_, by default False
        volume : bool, optional
            _description_, by default False
        nonlinear_residuals : bool, optional
            _description_, by default False
        linear_residuals : bool, optional
            _description_, by default False
        cfl : bool, optional
            _description_, by default False
        minmax_state : bool, optional
            _description_, by default False
        surface_forces : bool, optional
            _description_, by default False
        total_forces : bool, optional
            _description_, by default False
        bet_forces : bool, optional
            _description_, by default False
        actuator_disk_output : bool, optional
            _description_, by default False
        all : bool, optional
            _description_, by default False
        overwrite : bool, optional
            _description_, by default False

        Raises
        ------
        e
            _description_
        e
            _description_
        """

        download_map = [
            (surface, CaseDownloadable.SURFACE),
            (volume, CaseDownloadable.VOLUME),
            (nonlinear_residuals, CaseDownloadable.NONLINEAR_RESIDUALS),
            (linear_residuals, CaseDownloadable.LINEAR_RESIDUALS),
            (cfl, CaseDownloadable.CFL),
            (minmax_state, CaseDownloadable.MINMAX_STATE),
            (surface_forces, CaseDownloadable.SURFACE_FORCES),
            (total_forces, CaseDownloadable.TOTAL_FORCES),
        ]

        for do_download, filename in download_map:
            if do_download or all:
                self.download_file(filename, overwrite=overwrite)

        if bet_forces or all:
            try:
                self.download_file(
                    CaseDownloadable.BET_FORCES, overwrite=overwrite, log_error=False
                )
            except CloudFileNotFoundError as err:
                if not self._case.has_bet_disks():
                    if bet_forces:
                        log.warning("Case does not have any BET disks.")
                else:
                    log.error(
                        f"A problem occured when trying to download bet disk forces: {CaseDownloadable.BET_FORCES}"
                    )
                    raise err

        if actuator_disk_output or all:
            try:
                self.download_file(
                    CaseDownloadable.ACTUATOR_DISK_OUTPUT, overwrite=overwrite, log_error=False
                )
            except CloudFileNotFoundError as err:
                if not self._case.has_actuator_disks():
                    if actuator_disk_output:
                        log.warning("Case does not have any actuator disks.")
                else:
                    log.error(
                        (
                            "A problem occured when trying to download actuator disk results:"
                            f"{CaseDownloadable.ACTUATOR_DISK_OUTPUT}"
                        )
                    )
                    raise err


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
