"""
Case component
"""

# pylint: disable=too-many-lines
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Iterator, List, Optional, Union

import pydantic as pd
import pydantic.v1 as pd_v1

from .. import error_messages
from ..cloud.flow360_requests import MoveCaseItem, MoveToFolderRequest
from ..cloud.rest_api import RestApi
from ..cloud.s3_utils import CloudFileNotFoundError
from ..exceptions import Flow360RuntimeError, Flow360ValidationError, Flow360ValueError
from ..log import log
from .folder import Folder
from .interfaces import (
    CaseInterface,
    CaseInterfaceV2,
    FolderInterface,
    VolumeMeshInterface,
)
from .resource_base import (
    AssetMetaBaseModel,
    AssetMetaBaseModelV2,
    Flow360Resource,
    Flow360ResourceListBase,
    Flow360Status,
    ResourceDraft,
    before_submit_only,
    is_object_cloud_resource,
)
from .results.case_results import (
    ActuatorDiskResultCSVModel,
    AeroacousticsResultCSVModel,
    BETForcesRadialDistributionResultCSVModel,
    BETForcesResultCSVModel,
    CaseDownloadable,
    CFLResultCSVModel,
    LegacyForceDistributionResultCSVModel,
    LinearResidualsResultCSVModel,
    MaxResidualLocationResultCSVModel,
    MinMaxStateResultCSVModel,
    MonitorsResultModel,
    NonlinearResidualsResultCSVModel,
    ResultBaseModel,
    ResultsDownloaderSettings,
    ResultTarGZModel,
    SurfaceForcesResultCSVModel,
    SurfaceHeatTransferResultCSVModel,
    TotalForcesResultCSVModel,
    UserDefinedDynamicsResultModel,
    XSlicingForceDistributionResultCSVModel,
    YSlicingForceDistributionResultCSVModel,
)
from .simulation import services
from .simulation.simulation_params import SimulationParams
from .utils import (
    _local_download_overwrite,
    is_valid_uuid,
    shared_account_confirm_proceed,
    validate_type,
)
from .v1.flow360_params import Flow360Params, UnvalidatedFlow360Params
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


class CaseMeta(AssetMetaBaseModel):
    """
    CaseMeta data component
    """

    id: str = pd_v1.Field(alias="caseId")
    case_mesh_id: str = pd_v1.Field(alias="caseMeshId")
    parent_id: Union[str, None] = pd_v1.Field(alias="parentId")
    status: Flow360Status = pd_v1.Field()

    # Resource status change, revisit when updating the case class
    @pd_v1.validator("status")
    @classmethod
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


class CaseMetaV2(AssetMetaBaseModelV2):
    """
    CaseMetaV2 component
    """

    id: str = pd.Field(alias="caseId")
    case_mesh_id: str = pd.Field(alias="caseMeshId")
    status: Flow360Status = pd.Field()

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
            volume_mesh_info = AssetMetaBaseModel(
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
            "runtimeParams": self.params.json(),
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
        # setting _id will disable "WARNING: You have not submitted..." warning message
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


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class Case(CaseBase, Flow360Resource):
    """
    Case component
    """

    _manifest_path = "visualize/manifest/manifest.json"
    _cloud_resource_type_name = "Case"
    _web_api_v2_class = Flow360Resource

    # pylint: disable=redefined-builtin
    def __init__(self, id: str):
        super().__init__(
            interface=CaseInterface,
            meta_class=CaseMeta,
            id=id,
        )

        self._params = None
        self._raw_params = None
        self._results = CaseResultsModel(case=self)
        self._manifest = None
        # _web_api_v2 handles all V2 communications for Case
        self._web_api_v2 = self._web_api_v2_class(
            interface=CaseInterfaceV2,
            meta_class=CaseMetaV2,
            id=id,
        )

    @classmethod
    def _from_meta(cls, meta: CaseMeta):
        validate_type(meta, "meta", CaseMeta)
        case = cls(id=meta.id)
        case._set_meta(meta)
        return case

    def get_simulation_params(self):
        """
        returns simulation params
        """

        try:
            params_as_dict = self._parse_json_from_cloud("simulation.json")
        except CloudFileNotFoundError as err:
            raise Flow360ValueError(
                "Simulation params not found for this case. It is likely it was created with old interface"
            ) from err

        # if the params come from GUI, it can contain data that is not conformal with SimulationParams thus cleaning
        param, errors, _ = services.validate_model(
            params_as_dict=params_as_dict,
            validated_by=services.ValidationCalledBy.LOCAL,
            root_item_type=None,
            validation_level=None,
        )

        if errors is not None:
            raise Flow360ValidationError(
                f"Error found in simulation params. The param may be created by an incompatible version. {errors}",
            )

        return param

    @property
    def params(self) -> Union[Flow360Params, SimulationParams]:
        """
        returns case params
        """
        if self._params is None:
            try:
                self._params = self.get_simulation_params()
                return self._params
            except Flow360ValueError:
                pass

            self._raw_params = json.loads(self.get(method="runtimeParams")["content"])
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as temp_file:
                    json.dump(self._raw_params, temp_file)

                self._params = Flow360Params(temp_file.name)
            except pd_v1.ValidationError as err:
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
    def info(self) -> CaseMetaV2:
        """
        returns metadata info for case
        """
        return super().info

    @property
    def project_id(self) -> Optional[str]:
        """Returns the project id of the case if case was run with V2 interface."""
        if isinstance(self.info, CaseMeta):
            return self.info.projectId
        if isinstance(self.info, CaseMetaV2):
            return self.info.project_id
        raise ValueError("Case info is not of type CaseMeta or CaseMetaV2")

    @property
    def volume_mesh(self) -> "VolumeMeshV2":
        """
        returns volume mesh
        """
        from_cache = self.local_resource_cache[self.volume_mesh_id]
        if from_cache is not None:
            return from_cache
        # pylint: disable=import-outside-toplevel,cyclic-import
        from .volume_mesh import VolumeMeshV2

        return VolumeMeshV2(self.volume_mesh_id)

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

    # pylint: disable=no-member
    def is_steady(self):
        """
        returns True when case is steady state
        """
        if isinstance(self.params, Flow360Params):
            return self.params.time_stepping.time_step_size == "inf"
        return self.params.is_steady()

    def has_actuator_disks(self):
        """
        returns True when case has actuator disk
        """
        if isinstance(self.params, Flow360Params):
            return self.params.actuator_disks is not None and len(self.params.actuator_disks) > 0
        return self.params.has_actuator_disks()

    def has_bet_disks(self):
        """
        returns True when case has BET disk
        """
        if isinstance(self.params, Flow360Params):
            return self.params.bet_disks is not None and len(self.params.bet_disks) > 0
        return self.params.has_bet_disks()

    def has_isosurfaces(self):
        """
        returns True when case has isosurfaces
        """
        if isinstance(self.params, Flow360Params):
            return self.params.iso_surface_output is not None
        return self.params.has_isosurfaces()

    def has_monitors(self):
        """
        returns True when case has monitors
        """
        if isinstance(self.params, Flow360Params):
            return self.params.monitor_output is not None
        return self.params.has_monitors()

    def has_volume_output(self):
        """
        returns True when case has volume output
        """
        if isinstance(self.params, Flow360Params):
            return self.params.volume_output is not None
        return self.params.has_volume_output()

    def has_aeroacoustics(self):
        """
        returns True when case has aeroacoustics
        """
        if isinstance(self.params, Flow360Params):
            return self.params.aeroacoustic_output is not None
        return self.params.has_aeroacoustics()

    def has_user_defined_dynamics(self):
        """
        returns True when case has user defined dynamics
        """
        if isinstance(self.params, Flow360Params):
            return self.params.user_defined_dynamics is not None
        return self.params.has_user_defined_dynamics()

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

    @classmethod
    def from_local_storage(cls, local_storage_path, meta_data: CaseMeta) -> Case:
        """
        Create a `Case` instance from local storage.

        Parameters
        ----------
        local_storage_path : str
            The path to the local storage directory.
        meta_data : CaseMeta
            case metadata such as:
            id : str
                The unique identifier for the case.
            name : str
                The name of the case.
            user_id : str
                The user ID associated with the case, can be "local".

        Returns
        -------
        Case
            An instance of `Case` with data loaded from local storage.
        """
        _local_download_file = _local_download_overwrite(local_storage_path, cls.__name__)
        case = cls._from_meta(meta_data)
        case._download_file = _local_download_file
        case._results = CaseResultsModel(case=case)
        case.results.set_local_storage(local_storage_path, keep_remote_structure=True)
        return case

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
    bet_forces_radial_distribution: BETForcesRadialDistributionResultCSVModel = pd.Field(
        default_factory=lambda: BETForcesRadialDistributionResultCSVModel()
    )
    legacy_force_distribution: LegacyForceDistributionResultCSVModel = pd.Field(
        default_factory=lambda: LegacyForceDistributionResultCSVModel()
    )
    x_slicing_force_distribution: XSlicingForceDistributionResultCSVModel = pd.Field(
        default_factory=lambda: XSlicingForceDistributionResultCSVModel()
    )
    y_slicing_force_distribution: YSlicingForceDistributionResultCSVModel = pd.Field(
        default_factory=lambda: YSlicingForceDistributionResultCSVModel()
    )

    # user defined:
    user_defined_dynamics: UserDefinedDynamicsResultModel = pd.Field(
        default_factory=lambda: UserDefinedDynamicsResultModel()
    )

    # others
    surface_heat_transfer: SurfaceHeatTransferResultCSVModel = pd.Field(
        default_factory=lambda: SurfaceHeatTransferResultCSVModel()
    )
    aeroacoustics: AeroacousticsResultCSVModel = pd.Field(
        default_factory=lambda: AeroacousticsResultCSVModel()
    )

    local_storage: Optional[str] = None

    _downloader_settings: ResultsDownloaderSettings = pd.PrivateAttr(ResultsDownloaderSettings())

    @pd.model_validator(mode="after")
    def pass_download_function(self):
        """
        Pass download methods into fields of the case results.
        """
        if self.case is None:
            raise ValueError("case (type Case) is required")

        if not isinstance(self.case, Case):
            raise TypeError("case must be of type Case")

        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, ResultBaseModel):
                # pylint: disable=protected-access,no-member
                value._download_method = self.case._download_file
                # pylint: disable=protected-access,no-member
                value._get_params_method = lambda: self.case.params
                value.local_storage = self.local_storage

        return self

    @pd.model_validator(mode="after")
    def pass_get_files_function(self):
        """
        Pass file getters into fields of the case results
        """
        # pylint: disable=no-member,assigning-non-slot
        self.monitors.get_download_file_list_method = self.case.get_download_file_list
        return self

    # pylint: disable=protected-access,no-member
    @pd.model_validator(mode="after")
    def pass_has_functions(self):
        """
        Pass check to see if result is downloadable based on params
        """

        has_function_map = {
            "actuator_disks": self.case.has_actuator_disks,
            "bet_forces": self.case.has_bet_disks,
            "bet_forces_radial_distribution": self.case.has_bet_disks,
            "isosurfaces": self.case.has_isosurfaces,
            "monitors": self.case.has_monitors,
            "volumes": self.case.has_volume_output,
            "aeroacoustics": self.case.has_aeroacoustics,
            "user_defined_dynamics": self.case.has_user_defined_dynamics,
        }

        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, ResultBaseModel):
                function = has_function_map.get(field_name, lambda: True)
                value._is_downloadable = function  # pylint: disable=protected-access

        return self

    def _execute_downloading(self):
        """
        Download all specified and available results for the case
        """
        for _, value in self.__dict__.items():
            if isinstance(value, ResultBaseModel):
                # we download if explicitly set set_downloader(<result_name>=True),
                # or all=True but only when is not result=False
                # pylint: disable=no-member
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

        # pylint: disable=assigning-non-slot
        if folder_name is not None:
            self._downloader_settings.destination = folder_name
        # pylint: disable=no-member
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
        bet_forces_radial_distribution: bool = None,
        actuator_disks: bool = None,
        legacy_force_distribution: bool = None,
        x_slicing_force_distribution: bool = None,
        y_slicing_force_distribution: bool = None,
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
        bet_forces_radial_distribution : bool, optional
            Download BET (Blade Element Theory) forces radial distribution file if True.
        actuator_disk_output : bool, optional
            Download actuator disk output file if True.
        all : bool, optional
            Download all result files if True. Ignore file if explicitly set: <result_name>=False
        overwrite : bool, optional
            If True, overwrite existing files with the same name in the destination.
        destination : str, optional
            Location to save downloaded files. If None, files will be saved in the current directory under ID folder.
        """
        # pylint: disable=assigning-non-slot
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
        self.bet_forces_radial_distribution.do_download = bet_forces_radial_distribution
        self.actuator_disks.do_download = actuator_disks
        self.legacy_force_distribution.do_download = legacy_force_distribution
        self.x_slicing_force_distribution.do_download = x_slicing_force_distribution
        self.y_slicing_force_distribution.do_download = y_slicing_force_distribution

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
        # pylint: disable=no-member
        return self.case._download_file(
            file_name=file_name, to_file=to_file, to_folder=to_folder, overwrite=overwrite
        )

    def set_local_storage(self, local_storage: str, keep_remote_structure: bool = False):
        """
        Set local storage for fetching data from. Used with Case.from_local_storage(...)

        Parameters
        ----------
        local_storage : str
            Path to local folder
        keep_remote_structure : bool, optional
            When true, remote folder structure is assumed to be preserved, otherwise flat structure, by default False
        """
        for field_name in self.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, ResultBaseModel):
                if keep_remote_structure is True:
                    # pylint: disable=protected-access,no-member
                    value.local_storage = os.path.dirname(
                        os.path.join(local_storage, value._remote_path())
                    )
                else:
                    value.local_storage = local_storage


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
