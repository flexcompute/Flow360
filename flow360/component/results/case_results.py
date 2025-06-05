"""Case results module"""

from __future__ import annotations

import re
from collections import defaultdict
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np
import pydantic as pd

from flow360.cloud.s3_utils import CloudFileNotFoundError
from flow360.component.results.base_results import (
    _PHYSICAL_STEP,
    _PSEUDO_STEP,
    PerEntityResultCSVModel,
    ResultBaseModel,
    ResultCSVModel,
    ResultTarGZModel,
)
from flow360.component.simulation.conversion import unit_converter as unit_converter_v2
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.models.surface_models import BoundaryBase
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import (
    Flow360UnitSystem,
    ForceType,
    MomentType,
    PowerType,
    is_flow360_unit,
)
from flow360.component.v1.conversions import unit_converter as unit_converter_v1
from flow360.component.v1.flow360_params import Flow360Params
from flow360.exceptions import Flow360ValueError
from flow360.log import log

# pylint:disable=invalid-name
_CL = "CL"
_CD = "CD"
_CFx = "CFx"
_CFy = "CFy"
_CFz = "CFz"
_CMx = "CMx"
_CMy = "CMy"
_CMz = "CMz"
_CL_PRESSURE = "CLPressure"
_CD_PRESSURE = "CDPressure"
_CFx_PRESSURE = "CFxPressure"
_CFy_PRESSURE = "CFyPressure"
_CFz_PRESSURE = "CFzPressure"
_CMx_PRESSURE = "CMxPressure"
_CMy_PRESSURE = "CMyPressure"
_CMz_PRESSURE = "CMzPressure"
_CL_VISCOUS = "CLViscous"
_CD_VISCOUS = "CDViscous"
_CFx_VISCOUS = "CFxViscous"
_CFy_VISCOUS = "CFyViscous"
_CFz_VISCOUS = "CFzViscous"
_CMx_VISCOUS = "CMxViscous"
_CMy_VISCOUS = "CMyViscous"
_CMz_VISCOUS = "CMzViscous"
_HEAT_TRANSFER = "HeatTransfer"
_HEAT_FLUX = "HeatFlux"
_X = "X"
_Y = "Y"
_STRIDE = "stride"
_CUMULATIVE_CD_CURVE = "Cumulative_CD_Curve"
_CD_PER_STRIP = "CD_per_strip"
_CFx_PER_SPAN = "CFx_per_span"
_CFz_PER_SPAN = "CFz_per_span"
_CMy_PER_SPAN = "CMy_per_span"


class CaseDownloadable(Enum):
    """
    Case results filenames
    """

    # tar.gz
    SURFACES = "surfaces.tar.gz"
    VOLUMES = "volumes.tar.gz"
    SLICES = "slices.tar.gz"
    ISOSURFACES = "isosurfaces.tar.gz"
    MONITORS_ALL = "monitors.tar.gz"

    # convergence:
    NONLINEAR_RESIDUALS = "nonlinear_residual_v2.csv"
    LINEAR_RESIDUALS = "linear_residual_v2.csv"
    CFL = "cfl_v2.csv"
    MINMAX_STATE = "minmax_state_v2.csv"
    MAX_RESIDUAL_LOCATION = "max_residual_location_v2.csv"

    # forces:
    SURFACE_FORCES = "surface_forces_v2.csv"
    TOTAL_FORCES = "total_forces_v2.csv"
    BET_FORCES = "bet_forces_v2.csv"
    BET_FORCES_RADIAL_DISTRIBUTION = "bet_forces_radial_distribution_v2.csv"
    ACTUATOR_DISKS = "actuatorDisk_output_v2.csv"
    LEGACY_FORCE_DISTRIBUTION = "postprocess/forceDistribution.csv"
    Y_SLICING_FORCE_DISTRIBUTION = "Y_slicing_forceDistribution.csv"
    X_SLICING_FORCE_DISTRIBUTION = "X_slicing_forceDistribution.csv"

    # user defined:
    MONITOR_PATTERN = r"monitor_(.+)_v2.csv"
    USER_DEFINED_DYNAMICS_PATTERN = r"udd_(.+)_v2.csv"

    # others:
    AEROACOUSTICS = "total_acoustics_v3.csv"
    SURFACE_HEAT_TRANSFER = "surface_heat_transfer_v2.csv"


class ResultsDownloaderSettings(pd.BaseModel):
    """
    Settings for the results downloader.

    Parameters
    ----------
    all : bool, optional (default False)
        Flag indicating whether to download all available results.
    overwrite : bool, optional (default False)
        Flag indicating whether to overwrite existing files during download.
    destination : str, optional (default ".")
        The destination directory where the results will be downloaded.
    """

    all: Optional[bool] = pd.Field(False)
    overwrite: Optional[bool] = pd.Field(False)
    destination: Optional[str] = pd.Field(".")


class TimeSeriesResultCSVModel(ResultCSVModel):
    """Base CSV model for time series results"""

    _x_columns: List[str] = [_PHYSICAL_STEP, _PSEUDO_STEP]

    @property
    def x_columns(self):
        """Get x column"""
        return self._x_columns


# separate classes used to further customise give resutls, for example nonlinear_residuals.plot()
class NonlinearResidualsResultCSVModel(TimeSeriesResultCSVModel):
    """NonlinearResidualsResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.NONLINEAR_RESIDUALS.value, frozen=True)


class LinearResidualsResultCSVModel(TimeSeriesResultCSVModel):
    """LinearResidualsResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.LINEAR_RESIDUALS.value, frozen=True)


class CFLResultCSVModel(TimeSeriesResultCSVModel):
    """CFLResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.CFL.value, frozen=True)


class MinMaxStateResultCSVModel(TimeSeriesResultCSVModel):
    """CFLResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.MINMAX_STATE.value, frozen=True)


class MaxResidualLocationResultCSVModel(TimeSeriesResultCSVModel):
    """MaxResidualLocationResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.MAX_RESIDUAL_LOCATION.value, frozen=True)


class TotalForcesResultCSVModel(TimeSeriesResultCSVModel):
    """TotalForcesResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.TOTAL_FORCES.value, frozen=True)


class SurfaceForcesResultCSVModel(PerEntityResultCSVModel, TimeSeriesResultCSVModel):
    """SurfaceForcesResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.SURFACE_FORCES.value, frozen=True)

    _variables: List[str] = [
        _CL,
        _CD,
        _CFx,
        _CFy,
        _CFz,
        _CMx,
        _CMy,
        _CMz,
        _CL_PRESSURE,
        _CD_PRESSURE,
        _CFx_PRESSURE,
        _CFy_PRESSURE,
        _CFz_PRESSURE,
        _CMx_PRESSURE,
        _CMy_PRESSURE,
        _CMz_PRESSURE,
        _CL_VISCOUS,
        _CD_VISCOUS,
        _CFx_VISCOUS,
        _CFy_VISCOUS,
        _CFz_VISCOUS,
        _CMx_VISCOUS,
        _CMy_VISCOUS,
        _CMz_VISCOUS,
        _HEAT_TRANSFER,
    ]

    def _preprocess(self, filter_physical_steps_only: bool = True, include_time: bool = True):
        """
        run some processing after data is loaded
        """
        super()._preprocess(
            filter_physical_steps_only=filter_physical_steps_only, include_time=include_time
        )

    def reload_data(self, filter_physical_steps_only: bool = True, include_time: bool = True):
        return super().reload_data(filter_physical_steps_only, include_time)

    def _create_surface_forces_group(
        self, entity_groups: Dict[str, List[str]]
    ) -> SurfaceForcesGroupResultCSVModel:
        """
        Create the SurfaceForcesGroupResultCSVModel for the given entity groups.
        """
        raw_values = {}
        for x_column in self._x_columns:
            raw_values[x_column] = np.array(self.raw_values[x_column])
        for name, entities in entity_groups.items():
            self.filter(include=entities)
            for variable in self._variables:
                if f"{name}_{variable}" not in raw_values:
                    raw_values[f"{name}_{variable}"] = np.array(self.values[f"total{variable}"])
                    continue
                raw_values[f"{name}_{variable}"] += np.array(self.values[f"total{variable}"])

        raw_values = {key: val.tolist() for key, val in raw_values.items()}
        entity_groups = {key: sorted(val) for key, val in entity_groups.items()}

        return SurfaceForcesGroupResultCSVModel.from_dict(data=raw_values, group=entity_groups)

    def by_boundary_condition(self, params: SimulationParams) -> SurfaceForcesGroupResultCSVModel:
        """
        Group entities by boundary condition's name and create a
        SurfaceForcesGroupResultCSVModel.
        Forces from different boundaries but with the same type and name will be summed together.
        """

        entity_groups = defaultdict(list)
        for model in params.models:
            if not isinstance(model, BoundaryBase):
                continue
            boundary_name = model.name if model.name is not None else model.type
            entity_groups[boundary_name].extend(
                [entity.name for entity in model.entities.stored_entities]
            )
        return self._create_surface_forces_group(entity_groups=entity_groups)

    def by_body_group(self, params: SimulationParams) -> SurfaceForcesGroupResultCSVModel:
        """
        Group entities by body group's name and create a
        SurfaceForcesGroupResultCSVModel
        """
        if not isinstance(
            params.private_attribute_asset_cache.project_entity_info, GeometryEntityInfo
        ):
            raise Flow360ValueError(
                "Group surface forces by body group is only supported for case starting from geometry."
            )
        entity_info = params.private_attribute_asset_cache.project_entity_info
        if (
            not hasattr(entity_info, "body_attribute_names")
            or "groupByBodyId" not in entity_info.face_attribute_names
        ):
            raise Flow360ValueError(
                "The geometry in this case does not contain the necessary body group information, "
                "please upgrade the project to the latest version and re-run the case."
            )
        entity_groups = entity_info.get_body_group_to_face_group_name_map()
        return self._create_surface_forces_group(entity_groups=entity_groups)


class SurfaceForcesGroupResultCSVModel(SurfaceForcesResultCSVModel):
    """SurfaceForcesGroupResultCSVModel"""

    remote_file_name: str = pd.Field(None, frozen=True)  # Unused dummy field
    _entity_groups: dict = pd.PrivateAttr()

    @classmethod
    # pylint: disable=arguments-differ
    def from_dict(cls, data: dict, group: dict):
        obj = super().from_dict(data)
        # pylint: disable=protected-access
        obj._entity_groups = group
        return obj


class LegacyForceDistributionResultCSVModel(ResultCSVModel):
    """ForceDistributionResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.LEGACY_FORCE_DISTRIBUTION.value, frozen=True)


class XSlicingForceDistributionResultCSVModel(PerEntityResultCSVModel):
    """ForceDistributionResultCSVModel"""

    remote_file_name: str = pd.Field(
        CaseDownloadable.X_SLICING_FORCE_DISTRIBUTION.value, frozen=True
    )

    _variables: List[str] = [_CUMULATIVE_CD_CURVE, _CD_PER_STRIP]
    _filter_when_zero = [_CD_PER_STRIP]
    _x_columns: List[str] = [_X]

    def _preprocess(self, filter_physical_steps_only: bool = False, include_time: bool = False):
        """
        add _CD_PER_STRIP for filtering purpose and preprocess
        """
        for entity in self.entities:
            header = f"{entity}_{_CUMULATIVE_CD_CURVE}"
            cumulative_cd = np.array(self._values[header])
            cd_per_strip = np.insert(np.diff(cumulative_cd), 0, cumulative_cd[0])
            header_to_add = f"{entity}_{_CD_PER_STRIP}"
            self._values[header_to_add] = cd_per_strip.tolist()

        super()._preprocess(
            filter_physical_steps_only=filter_physical_steps_only, include_time=include_time
        )


class YSlicingForceDistributionResultCSVModel(PerEntityResultCSVModel):
    """ForceDistributionResultCSVModel"""

    remote_file_name: str = pd.Field(
        CaseDownloadable.Y_SLICING_FORCE_DISTRIBUTION.value, frozen=True
    )

    _variables: List[str] = [_CFx_PER_SPAN, _CFz_PER_SPAN, _CMy_PER_SPAN]
    _filter_when_zero = [_CFx_PER_SPAN, _CFz_PER_SPAN, _CMy_PER_SPAN]
    _x_columns: List[str] = [_Y, _STRIDE]


class SurfaceHeatTransferResultCSVModel(PerEntityResultCSVModel, TimeSeriesResultCSVModel):
    """SurfaceHeatTransferResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.SURFACE_HEAT_TRANSFER.value, frozen=True)
    _variables: List[str] = [_HEAT_FLUX]
    _filter_when_zero = []


class AeroacousticsResultCSVModel(TimeSeriesResultCSVModel):
    """AeroacousticsResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.AEROACOUSTICS.value, frozen=True)


MonitorCSVModel = ResultCSVModel


class MonitorsResultModel(ResultTarGZModel):
    """
    Model for handling results of monitors in TAR GZ and CSV formats.

    Inherits from ResultTarGZModel.
    """

    remote_file_name: str = pd.Field(CaseDownloadable.MONITORS_ALL.value, frozen=True)
    get_download_file_list_method: Optional[Callable] = pd.Field(lambda: None)

    _monitor_names: List[str] = pd.PrivateAttr([])
    _monitors: Dict[str, MonitorCSVModel] = pd.PrivateAttr({})

    @property
    def monitor_names(self):
        """
        Get the list of monitor names.

        Returns
        -------
        list of str
            List of monitor names.
        """

        if len(self._monitor_names) == 0:
            pattern = CaseDownloadable.MONITOR_PATTERN.value
            file_list = [
                file["fileName"]
                for file in self.get_download_file_list_method()  # pylint:disable=not-callable
            ]
            for filename in file_list:
                if filename.startswith("results/"):
                    filename = filename.split("results/")[1]
                    match = re.match(pattern, filename)
                    if match:
                        name = match.group(1)
                        self._monitor_names.append(name)
                        self._monitors[name] = MonitorCSVModel(remote_file_name=filename)
                        # pylint: disable=protected-access
                        self._monitors[name]._download_method = (
                            self._download_method
                        )  # pylint: disable=protected-access

        return self._monitor_names

    def get_monitor_by_name(self, name: str) -> MonitorCSVModel:
        """
        Get a monitor by name.

        Parameters
        ----------
        name : str
            The name of the monitor.

        Returns
        -------
        MonitorCSVModel
            The MonitorCSVModel corresponding to the given name.

        Raises
        ------
        Flow360ValueError
            If the monitor with the provided name is not found.
        """

        if name not in self.monitor_names:
            raise Flow360ValueError(
                f"Cannot find monitor with provided name={name}, available monitors: {self.monitor_names}"
            )
        return self._monitors[name]

    def __getitem__(self, name: str) -> MonitorCSVModel:
        """
        Get a monitor by name (supporting [] access).
        """

        return self.get_monitor_by_name(name)


UserDefinedDynamicsCSVModel = TimeSeriesResultCSVModel


class UserDefinedDynamicsResultModel(ResultBaseModel):
    """
    Model for handling results of user-defined dynamics.

    Inherits from ResultBaseModel.
    """

    remote_file_name: str = pd.Field(None, frozen=True)
    get_download_file_list_method: Optional[Callable] = pd.Field(lambda: None)

    _udd_names: List[str] = pd.PrivateAttr([])
    _udds: Dict[str, UserDefinedDynamicsCSVModel] = pd.PrivateAttr({})

    @property
    def udd_names(self):
        """
        Get the list of user-defined dynamics names.

        Returns
        -------
        list of str
            List of user-defined dynamics names.
        """

        if len(self._udd_names) == 0:
            pattern = CaseDownloadable.USER_DEFINED_DYNAMICS_PATTERN.value
            file_list = [
                file["fileName"]
                for file in self.get_download_file_list_method()  # pylint:disable=not-callable
            ]
            for filename in file_list:
                if filename.startswith("results/"):
                    filename = filename.split("results/")[1]
                    match = re.match(pattern, filename)
                    if match:
                        name = match.group(1)
                        self._udd_names.append(name)
                        self._udds[name] = UserDefinedDynamicsCSVModel(remote_file_name=filename)
                        # pylint: disable=protected-access
                        self._udds[name]._download_method = self._download_method

        return self._udd_names

    def download(
        self, to_folder: str = ".", overwrite: bool = False
    ):  # pylint:disable=arguments-differ
        """
        Download all udd files to the specified location.

        Parameters
        ----------
        to_folder : str, optional
            The folder where the file will be downloaded.
        overwrite : bool, optional
            Flag indicating whether to overwrite existing files.
        """

        for udd in self._udds.values():
            udd.download(to_folder=to_folder, overwrite=overwrite)

    def get_udd_by_name(self, name: str) -> UserDefinedDynamicsCSVModel:
        """
        Get user-defined dynamics by name.

        Parameters
        ----------
        name : str
            The name of the user-defined dynamics.

        Returns
        -------
        UserDefinedDynamicsCSVModel
            The UserDefinedDynamicsCSVModel corresponding to the given name.

        Raises
        ------
        Flow360ValueError
            If the user-defined dynamics with the provided name is not found.
        """

        if name not in self.udd_names:
            raise Flow360ValueError(
                f"Cannot find user defined dynamics with provided name={name}, "
                f"available user defined dynamics: {self.udd_names}"
            )
        return self._udds[name]

    def __getitem__(self, name: str) -> UserDefinedDynamicsCSVModel:
        """
        Get a UUD by name (supporting [] access).
        """

        return self.get_udd_by_name(name)


class _DimensionedCSVResultModel(pd.BaseModel):
    """
    Base model for handling dimensioned CSV results.

    Attributes
    ----------
    _name : str
        Name of the dimensioned CSV result.
    """

    _name: str

    def _in_base_component(self, base, component, component_name, params):
        log.debug(f"   -> need conversion for: {component_name} = {component}")

        if isinstance(params, SimulationParams):
            flow360_conv_system = unit_converter_v2(
                component.units.dimensions,
                params=params,
                required_by=[self._name, component_name],
            )
        elif isinstance(params, Flow360Params):
            flow360_conv_system = unit_converter_v1(
                component.units.dimensions,
                params=params,
                required_by=[self._name, component_name],
            )
        else:
            raise Flow360ValueError(
                f"Unknown type of params: {type(params)=}, expected one of (Flow360Params, SimulationParams)"
            )

        if is_flow360_unit(component):
            converted = component.in_base(base, flow360_conv_system)
        else:
            component.units.registry = flow360_conv_system.registry  # pylint:disable=no-member
            converted = component.in_base(unit_system=base)
        log.debug(f"      converted to: {converted}")
        return converted


class _ActuatorDiskResults(_DimensionedCSVResultModel):
    """
    Model for handling results of actuator disks.

    Inherits from _DimensionedCSVResultModel.

    Attributes
    ----------
    power : PowerType.Array
        Array of power values.
    force : ForceType.Array
        Array of force values.
    moment : MomentType.Array
        Array of moment values.

    Methods
    -------
    to_base(base: Any, params: Any)
        Convert the results to the specified base system.
    """

    power: PowerType.Array = pd.Field()
    force: ForceType.Array = pd.Field()
    moment: MomentType.Array = pd.Field()
    _name = "actuator_disks"

    def to_base(self, base: str, params: Flow360Params):
        """
        Convert the results to the specified base system.

        Parameters
        ----------
        base : str
            The base system to convert the results to, for example SI.
        params : Flow360Params
            Case parameters for the conversion.
        """

        self.power = self._in_base_component(base, self.power, "power", params)
        self.force = self._in_base_component(base, self.force, "force", params)
        self.moment = self._in_base_component(base, self.moment, "moment", params)


class OptionallyDownloadableResultCSVModel(ResultCSVModel):
    """
    Model for handling optionally downloadable CSV results.

    Inherits from ResultCSVModel.
    """

    _err_msg = "Case does not produced these results."

    def download(
        self, to_file: str = None, to_folder: str = ".", overwrite: bool = False, **kwargs
    ):
        """
        Download the results to the specified file or folder.

        Parameters
        ----------
        to_file : str, optional
            The file path where the results will be saved.
        to_folder : str, optional
            The folder path where the results will be saved.
        overwrite : bool, optional
            Whether to overwrite existing files with the same name.

        Raises
        ------
        CloudFileNotFoundError
            If the cloud file for the results is not found.
        """

        try:
            super().download(
                to_file=to_file, to_folder=to_folder, overwrite=overwrite, log_error=False, **kwargs
            )
        except CloudFileNotFoundError as err:
            if self._is_downloadable() is False:  # pylint:disable=not-callable
                log.warning(self._err_msg)
            else:
                log.error(
                    "A problem occured when trying to download results:" f"{self.remote_file_name}"
                )
                raise err


class ActuatorDiskResultCSVModel(OptionallyDownloadableResultCSVModel):
    """
    Model for handling actuator disk CSV results.

    Inherits from OptionallyDownloadableResultCSVModel.

    Methods
    -------
    to_base(base, params=None)
        Convert the results to the specified base system.

    Notes
    -----
    This class provides methods to handle actuator disk CSV results and convert them to the specified base system.
    """

    remote_file_name: str = pd.Field(CaseDownloadable.ACTUATOR_DISKS.value, frozen=True)
    _err_msg = "Case does not have any actuator disks."

    def to_base(self, base: str, params: Flow360Params = None):
        """
        Convert the results to the specified base system.

        Parameters
        ----------
        base : str
            The base system to convert the results to. For example SI.
        params : Flow360Params, optional
            Case parameters for the conversion.
        """

        if params is None:
            params = self._get_params_method()  # pylint:disable=not-callable
        disk_names = np.unique(
            [v.split("_")[0] for v in self.values.keys() if v.startswith("Disk")]
        )
        with Flow360UnitSystem(verbose=False):
            for disk_name in disk_names:
                disk = _ActuatorDiskResults(
                    power=self.values[f"{disk_name}_Power"],
                    force=self.values[f"{disk_name}_Force"],
                    moment=self.values[f"{disk_name}_Moment"],
                )
                disk.to_base(base, params)
                self.values[f"{disk_name}_Power"] = disk.power
                self.values[f"{disk_name}_Force"] = disk.force
                self.values[f"{disk_name}_Moment"] = disk.moment

                self.values["PowerUnits"] = disk.power.units
                self.values["ForceUnits"] = disk.force.units
                self.values["MomentUnits"] = disk.moment.units


class _BETDiskResults(_DimensionedCSVResultModel):
    """
    Model for handling BET disk results.

    Inherits from _DimensionedCSVResultModel.

    Attributes
    ----------
    force_x : ForceType.Array
        Array of force values along the x-axis.
    force_y : ForceType.Array
        Array of force values along the y-axis.
    force_z : ForceType.Array
        Array of force values along the z-axis.
    moment_x : MomentType.Array
        Array of moment values about the x-axis.
    moment_y : MomentType.Array
        Array of moment values about the y-axis.
    moment_z : MomentType.Array
        Array of moment values about the z-axis.
    _name : str
        Name of the BET forces result.

    Methods
    -------
    to_base(base, params)
        Convert the results to the specified base system.
    """

    force_x: ForceType.Array = pd.Field()
    force_y: ForceType.Array = pd.Field()
    force_z: ForceType.Array = pd.Field()
    moment_x: MomentType.Array = pd.Field()
    moment_y: MomentType.Array = pd.Field()
    moment_z: MomentType.Array = pd.Field()

    _name = "bet_forces"

    def to_base(self, base: str, params: Flow360Params):
        """
        Convert the results to the specified base system.

        Parameters
        ----------
        base : str
            The base system to convert the results to, for example SI.
        params : Flow360Params
            Case parameters for the conversion.
        """

        self.force_x = self._in_base_component(base, self.force_x, "force_x", params)
        self.force_y = self._in_base_component(base, self.force_y, "force_y", params)
        self.force_z = self._in_base_component(base, self.force_z, "force_z", params)
        self.moment_x = self._in_base_component(base, self.moment_x, "moment_x", params)
        self.moment_y = self._in_base_component(base, self.moment_y, "moment_y", params)
        self.moment_z = self._in_base_component(base, self.moment_z, "moment_z", params)


class BETForcesResultCSVModel(OptionallyDownloadableResultCSVModel):
    """
    Model for handling BET forces CSV results.

    Inherits from OptionallyDownloadableResultCSVModel.

    Methods
    -------
    to_base(base, params=None)
        Convert the results to the specified base system.
    """

    remote_file_name: str = pd.Field(CaseDownloadable.BET_FORCES.value, frozen=True)
    _err_msg = "Case does not have any BET disks."

    def to_base(self, base: str, params: Flow360Params = None):
        """
        Convert the results to the specified base system.

        Parameters
        ----------
        base : str
            The base system to convert the results to. For example SI.
        params : Flow360Params, optional
            Case parameters for the conversion.
        """

        if params is None:
            params = self._get_params_method()  # pylint:disable=not-callable
        disk_names = np.unique(
            [v.split("_")[0] for v in self.values.keys() if v.startswith("Disk")]
        )
        with Flow360UnitSystem(verbose=False):
            for disk_name in disk_names:
                bet = _BETDiskResults(
                    force_x=self.values[f"{disk_name}_Force_x"],
                    force_y=self.values[f"{disk_name}_Force_y"],
                    force_z=self.values[f"{disk_name}_Force_z"],
                    moment_x=self.values[f"{disk_name}_Moment_x"],
                    moment_y=self.values[f"{disk_name}_Moment_y"],
                    moment_z=self.values[f"{disk_name}_Moment_z"],
                )
                bet.to_base(base, params)

                self.values[f"{disk_name}_Force_x"] = bet.force_x
                self.values[f"{disk_name}_Force_y"] = bet.force_y
                self.values[f"{disk_name}_Force_z"] = bet.force_z
                self.values[f"{disk_name}_Moment_x"] = bet.moment_x
                self.values[f"{disk_name}_Moment_y"] = bet.moment_y
                self.values[f"{disk_name}_Moment_z"] = bet.moment_z

                self.values["ForceUnits"] = bet.force_x.units
                self.values["MomentUnits"] = bet.moment_x.units


class BETForcesRadialDistributionResultCSVModel(OptionallyDownloadableResultCSVModel):
    """
    Model for handling BET forces radial distribution CSV results.

    Inherits from OptionallyDownloadableResultCSVModel.
    """

    remote_file_name: str = pd.Field(
        CaseDownloadable.BET_FORCES_RADIAL_DISTRIBUTION.value, frozen=True
    )
    _err_msg = "Case does not have any BET disks."
