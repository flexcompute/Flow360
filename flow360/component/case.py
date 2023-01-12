"""
Case component
"""
from __future__ import annotations
import json
from enum import Enum
from pydantic import Extra, Field
from pylab import show, subplots

from ..cloud.s3_utils import S3TransferType, CloudFileNotFoundError
from ..cloud.rest_api import RestApi
from .flow360_base_model import (
    Flow360BaseModel,
    Flow360Resource,
    before_submit_only,
    on_cloud_resource_only,
    is_object_cloud_resource,
)
from .flow360_solver_params import Flow360Params
from .utils import is_valid_uuid


class CaseMeta(Flow360BaseModel, extra=Extra.allow):
    """
    Case component
    """

    id: str = Field(alias="caseId")
    case_mesh_id: str = Field(alias="caseMeshId")
    status: str = Field(alias="caseStatus")
    parent_id: str = Field(alias="parentId")

    def to_case(self) -> Case:
        """
        returns Case object from case meta info
        """
        return Case(self.id)


# pylint: disable=too-many-instance-attributes
class Case(Flow360Resource):
    """
    Case component
    """

    def __init__(self, case_id: str = None):
        super().__init__(
            resource_type="Case",
            info_type_class=CaseMeta,
            s3_transfer_method=S3TransferType.CASE,
            endpoint="case",
            id=case_id,
        )
        if case_id is not None:
            self.get_info()
            self._params = Flow360Params(**json.loads(self.get(method="runtimeParams")["content"]))

        self.other_case = None
        self.parent_case = None
        self.parent_id = None
        self.tags = None
        self._results = CaseResults(self)

    def __str__(self):
        if self._info is not None:
            return self.info.__str__()
        return "Case is not yet submitted"

    @property
    def params(self):
        """
        returns case params
        """
        return self._params

    @params.setter
    @before_submit_only
    def params(self, value):
        """
        sets case params (before submit only)
        """
        if not isinstance(value, Flow360Params):
            raise ValueError("params are not of type Flow360Params.")
        self._params = value

    @property
    def name(self):
        """
        returns case name
        """
        if self.is_cloud_resource():
            return self.info.name
        return self._name

    @name.setter
    @before_submit_only
    def name(self, value):
        """
        sets case name (before submit only)
        """
        self._name = value

    @property
    def volume_mesh_id(self):
        """
        returns volume mesh id (before submit only)
        """
        if self.is_cloud_resource():
            return self.info.case_mesh_id
        return self._volume_mesh_id

    @volume_mesh_id.setter
    @before_submit_only
    def volume_mesh_id(self, value):
        """
        sets volume mesh id (before submit only)
        """
        self._volume_mesh_id = value

    @property
    @on_cloud_resource_only
    def results(self) -> CaseResults:
        """
        returns results object to managing case results
        """
        return self._results

    @before_submit_only
    def submit(self):
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

        if self.parent_case is not None:
            parent_id = self.parent_case.id
            volume_mesh_id = self.parent_case.volume_mesh_id

        volume_mesh_id = volume_mesh_id or self.other_case.volume_mesh_id

        is_valid_uuid(volume_mesh_id)
        is_valid_uuid(parent_id, ignore_none=True)

        resp = self.post(
            json={
                "name": self.name,
                "meshId": volume_mesh_id,
                "runtimeParams": self.params.json(),
                "tags": self.tags,
                "parentId": parent_id,
            },
            path=f"volumemeshes/{volume_mesh_id}/case",
        )
        self._info = CaseMeta(**resp)
        self.init_id(self._info.id)

    @on_cloud_resource_only
    def download_log(self, log, to_file=".", keep_folder: bool = True):
        """
        Download log
        :param log:
        :param to_file: file name on local disk, could be either folder or file name.
        :param keep_folder: If true, the downloaded file will be put in the same folder as the file on cloud. Only work
        when file_name is a folder name.
        :return:
        """

        self.download_file(f"logs/{log.value}", to_file, keep_folder)

    @on_cloud_resource_only
    def is_case_steady(self):
        """
        returns True when case is steady state
        """
        return self.params.time_stepping.time_step_size == "inf"

    @on_cloud_resource_only
    def has_actuator_disks(self):
        """
        returns True when case has actuator disk
        """
        if self.params.actuator_disks is not None:
            if len(self.params.actuator_disks) > 0:
                return True
        return False

    @on_cloud_resource_only
    def has_bet_disks(self):
        """
        returns True when case has BET disk
        """
        if self.params.bet_disks is not None:
            if len(self.params.bet_disks) > 0:
                return True
        return False

    def copy(self, name: str = None, params: Flow360Params = None, tags: [str] = None) -> Case:
        """
        Alias for retry case
        :param name:
        :param params:
        :param tags:
        :return:
        """

        return self.retry(name, params, tags)

    def retry(self, name: str = None, params: Flow360Params = None, tags: [str] = None) -> Case:
        """
        Retry case
        :param name:
        :param params:
        :param tags:
        :return:
        """

        name = name or self.name or self.info.name
        params = params or self.params.copy(deep=True)
        new_case = Case.new(name, params, other_case=self, tags=tags)
        return new_case

    def continuation(
        self, name: str = None, params: Flow360Params = None, tags: [str] = None
    ) -> Case:
        """
        Alias for fork a case to continue simulation
        :param name:
        :param params:
        :param tags:
        :return:
        """

        return self.fork(name, params, tags)

    def fork(self, name: str = None, params: Flow360Params = None, tags: [str] = None) -> Case:
        """
        Fork a case to continue simulation
        :param name:
        :param params:
        :param tags:
        :return:
        """

        name = name or self.name or self.info.name
        params = params or self.params.copy(deep=True)
        return Case.new(name, params, parent_case=self, tags=tags)

    @before_submit_only
    def validate_case_inputs(self, pre_submit_checks=False):
        """
        validates case inputs (before submit only)
        """
        if self.volume_mesh_id is not None and self.other_case is not None:
            raise ValueError("You cannot specify both volume_mesh_id AND other_case.")

        if self.parent_id is not None and self.parent_case is not None:
            raise ValueError("You cannot specify both parent_id AND parent_case.")

        if self.parent_id is not None or self.parent_case is not None:
            if self.volume_mesh_id is not None or self.other_case is not None:
                raise ValueError(
                    "You cannot specify volume_mesh_id OR other_case when parent case provided."
                )

        is_valid_uuid(self.volume_mesh_id, ignore_none=True)

        if pre_submit_checks:
            is_object_cloud_resource(self.other_case)
            is_object_cloud_resource(self.parent_case)

    @classmethod
    def from_cloud(cls, case_id: str):
        """
        get case from cloud
        """
        return cls(case_id)

    # pylint: disable=too-many-arguments
    @classmethod
    def new(
        cls,
        name: str,
        params: Flow360Params,
        volume_mesh_id: str = None,
        tags: [str] = None,
        parent_id=None,
        other_case: Case = None,
        parent_case: Case = None,
    ) -> Case:
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

        if not isinstance(params, Flow360Params):
            raise ValueError("params are not of type Flow360Params.")

        new_case = cls()
        new_case.name = name
        new_case.volume_mesh_id = volume_mesh_id
        new_case.other_case = other_case
        new_case.params = params.copy(deep=True)
        new_case.tags = tags
        new_case.parent_id = parent_id
        new_case.parent_case = parent_case

        new_case.validate_case_inputs()

        return new_case

    # pylint: disable=too-many-arguments
    # @classmethod
    # def submit_multiple_phases(
    #     cls,
    #     name: str,
    #     volume_mesh_id: str,
    #     params: Flow360Params,
    #     tags: [str] = None,
    #     phase_steps=1,
    # ):
    #     """
    #     Create multiple cases from volume mesh
    #     :param name:
    #     :param volume_mesh_id:
    #     :param params:
    #     :param tags:
    #     :param parent_id:
    #     :param phase_steps:
    #     :return:
    #     """

    #     assert name
    #     assert volume_mesh_id
    #     assert params
    #     assert phase_steps >= 1

    #     result = []

    #     total_steps = (
    #         params.time_stepping.max_physical_steps
    #         if params.time_stepping and params.time_stepping.max_physical_steps
    #         else 1
    #     )

    #     num_cases = math.ceil(total_steps / phase_steps)
    #     for i in range(1, num_cases + 1):
    #         parent_id = result[-1].case_id if result else None
    #         case = http.post(
    #             f"volumemeshes/{volume_mesh_id}/case",
    #             json={
    #                 "name": f"{name}_{i}",
    #                 "meshId": volume_mesh_id,
    #                 "runtimeParams": params.json(),
    #                 "tags": tags,
    #                 "parentId": parent_id,
    #             },
    #         )

    #         result.append(cls(**case))

    #     return result


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
            self.data = self.get_method(method=self.path)
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
        steady = self.results._case.is_case_steady()
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
        return f"result/{result_type.value}"

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

    # pylint: disable=redefined-builtin,too-many-locals
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
                self.download_file(CaseDownloadable.BET_FORCES, overwrite=overwrite)
            except CloudFileNotFoundError as err:
                if not self._case.has_bet_disks():
                    if bet_forces:
                        print("Case does not have any BET disks.")
                else:
                    print("A problem occured when trying to download bet disk forces.")
                    raise err

        if actuator_disk_output or all:
            try:
                self.download_file(CaseDownloadable.ACTUATOR_DISK_OUTPUT, overwrite=overwrite)
            except CloudFileNotFoundError as err:
                if not self._case.has_actuator_disks():
                    if actuator_disk_output:
                        print("Case does not have any actuator disks.")
                else:
                    print("A problem occured when trying to download actuator disk results")
                    raise err


class CaseList(list, RestApi):
    """
    Case List component
    """

    def __init__(self, mesh_id: str = None, from_cloud: bool = True, include_deleted: bool = False):
        if mesh_id is not None:
            RestApi.__init__(self, endpoint=f"volumemeshes/{mesh_id}/cases")
        else:
            RestApi.__init__(self, endpoint="cases")

        if from_cloud:
            resp = self.get(params={"includeDeleted": include_deleted})
            list.__init__(self, [CaseMeta(**item) for item in resp])

    def filter(self):
        """
        flitering list, not implemented yet
        """
        raise NotImplementedError("Filters are not implemented yet")
        # resp = list(filter(lambda i: i['caseStatus'] != 'deleted', resp))

    def __getitem__(self, index) -> CaseMeta:
        """
        returns CaseMeta info item of the list
        """
        return super().__getitem__(index)

    @classmethod
    def from_cloud(cls, mesh_id: str = None):
        """
        get Case List from cloud
        """
        return cls(mesh_id=mesh_id, from_cloud=True)
