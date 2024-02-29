import os
import re
import shutil
import tempfile
import time
from enum import Enum
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas
import pydantic as pd

from ...cloud.s3_utils import (
    CloudFileNotFoundError,
    get_local_filename_and_create_folders,
)
from ...exceptions import Flow360ValueError
from ...log import log
from ..flow360_params.conversions import unit_converter
from ..flow360_params.flow360_params import Flow360Params
from ..flow360_params.unit_system import (
    Flow360UnitSystem,
    ForceType,
    MomentType,
    PowerType,
    is_flow360_unit,
)


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
    ACTUATOR_DISKS = "actuatorDisk_output_v2.csv"
    FORCE_DISTRIBUTION = "postprocess/forceDistribution.csv"

    # user defined:
    MONITOR_PATTERN = r"monitor_(.+)_v2.csv"
    USER_DEFINED_DYNAMICS_PATTERN = r"udd_(.+)_v2.csv"

    # others:
    AEROACOUSTICS = "total_acoustics_v3.csv"
    SURFACE_HEAT_TRANSFER = "surface_heat_transfer_v2.csv"


class ResultsDownloaderSettings(pd.BaseModel):

    all: bool = pd.Field(False)
    overwrite: bool = pd.Field(False)
    destination: str = "."


class ResultBaseModel(pd.BaseModel):
    remote_file_name: str = pd.Field()
    local_file_name: str = pd.Field(None)
    do_download: Optional[bool] = pd.Field(None)
    _download_method: Optional[Callable] = pd.PrivateAttr()
    _get_params_method: Optional[Callable] = pd.PrivateAttr()
    _is_downloadable: Callable = pd.PrivateAttr(lambda: True)

    def download(self, to_file: str = None, to_folder: str = ".", overwrite: bool = False):
        self._download_method(
            self._remote_path(), to_file=to_file, to_folder=to_folder, overwrite=overwrite
        )

    def _remote_path(self):
        return f"results/{self.remote_file_name}"


class ResultCSVModel(ResultBaseModel):
    temp_file: str = pd.Field(
        const=True,
        default_factory=lambda: tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name,
    )
    _values: Optional[Dict] = pd.PrivateAttr(None)
    _raw_values: Optional[Dict] = pd.PrivateAttr(None)

    def _read_csv_file(self, filename: str):
        df = pandas.read_csv(filename, skipinitialspace=True)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        return df.to_dict("list")

    @property
    def raw_values(self):
        if self._raw_values is None:
            self.load_from_remote()
        return self._raw_values

    def load_from_local(self, filename: str):
        self._raw_values = self._read_csv_file(filename)
        self.local_file_name = filename

    def load_from_remote(self, **kwargs_download):
        self.download(to_file=self.temp_file, overwrite=True, **kwargs_download)
        self._raw_values = self._read_csv_file(self.temp_file)
        self.local_file_name = self.temp_file

    def download(
        self, to_file: str = None, to_folder: str = ".", overwrite: bool = False, **kwargs
    ):
        local_file_path = get_local_filename_and_create_folders(
            self._remote_path(), to_file, to_folder
        )
        if os.path.exists(local_file_path) and not overwrite:
            log.info(
                f"Skipping downloading {self.remote_file_name}, local file {local_file_path} exists."
            )

        else:
            if overwrite is True or self.local_file_name is None:
                self._download_method(
                    self._remote_path(),
                    to_file=to_file,
                    to_folder=to_folder,
                    overwrite=overwrite,
                    **kwargs,
                )
            else:
                shutil.copy(self.temp_file, local_file_path)
                log.info(f"Saved to {local_file_path}")

    @property
    def values(self):
        if self._values is None:
            self._values = self.raw_values
        return self._values

    def to_base(self, base):
        """values in base system"""
        pass

    def to_file(self, filename: str = None):
        self.as_dataframe().to_csv(filename, index=False)
        log.info(f"Saved to {filename}")

    def as_dict(self):
        return self.values

    def as_numpy(self):
        return self.as_dataframe().to_numpy()

    def as_dataframe(self):
        return pandas.DataFrame(self.values)


class ResultTarGZModel(ResultBaseModel):
    def to_file(self, filename, overwrite: bool = False):
        self.download(to_file=filename, overwrite=overwrite)


# separate classes used to further customise give resutls, for example nonlinear_residuals.plot()
class NonlinearResidualsResultCSVModel(ResultCSVModel):
    remote_file_name: str = pd.Field(CaseDownloadable.NONLINEAR_RESIDUALS.value, const=True)


class LinearResidualsResultCSVModel(ResultCSVModel):
    remote_file_name: str = pd.Field(CaseDownloadable.LINEAR_RESIDUALS.value, const=True)


class CFLResultCSVModel(ResultCSVModel):
    remote_file_name: str = pd.Field(CaseDownloadable.CFL.value, const=True)


class MinMaxStateResultCSVModel(ResultCSVModel):
    remote_file_name: str = pd.Field(CaseDownloadable.MINMAX_STATE.value, const=True)


class MaxResidualLocationResultCSVModel(ResultCSVModel):
    remote_file_name: str = pd.Field(CaseDownloadable.MAX_RESIDUAL_LOCATION.value, const=True)


class TotalForcesResultCSVModel(ResultCSVModel):
    remote_file_name: str = pd.Field(CaseDownloadable.TOTAL_FORCES.value, const=True)


class SurfaceForcesResultCSVModel(ResultCSVModel):
    remote_file_name: str = pd.Field(CaseDownloadable.SURFACE_FORCES.value, const=True)


class ForceDistributionResultCSVModel(ResultCSVModel):
    remote_file_name: str = pd.Field(CaseDownloadable.FORCE_DISTRIBUTION.value, const=True)

    def wait(self, refresh_rate_seconds=2, timeout_minutes=60):
        """Wait until the Case finishes processing, refresh periodically"""
        
        start_time = time.time()
        while time.time() - start_time < timeout_minutes * 60:
            try:
                self.load_from_remote(log_error=False)
                return None
            except CloudFileNotFoundError:
                pass
            time.sleep(refresh_rate_seconds)

        raise TimeoutError("Timeout: post-processing did not finish within the specified timeout period.")            



class SurfaceHeatTrasferResultCSVModel(ResultCSVModel):
    remote_file_name: str = pd.Field(CaseDownloadable.SURFACE_HEAT_TRANSFER.value, const=True)


class AeroacousticsResultCSVModel(ResultCSVModel):
    remote_file_name: str = pd.Field(CaseDownloadable.AEROACOUSTICS.value, const=True)


MonitorCSVModel = ResultCSVModel


class MonitorsResultModel(ResultTarGZModel):
    remote_file_name: str = pd.Field(CaseDownloadable.MONITORS_ALL.value, const=True)
    get_download_file_list_method: Optional[Callable] = pd.Field()

    _monitor_names: List[str] = pd.PrivateAttr([])
    _monitors: Dict[str, MonitorCSVModel] = pd.PrivateAttr({})

    @property
    def monitor_names(self):
        if len(self._monitor_names) == 0:
            pattern = CaseDownloadable.MONITOR_PATTERN.value
            file_list = [file["fileName"] for file in self.get_download_file_list_method()]
            for filename in file_list:
                if filename.startswith("results/"):
                    filename = filename.split("results/")[1]
                    match = re.match(pattern, filename)
                    if match:
                        name = match.group(1)
                        self._monitor_names.append(name)
                        self._monitors[name] = MonitorCSVModel(remote_file_name=filename)
                        self._monitors[name]._download_method = self._download_method

        return self._monitor_names

    def get_monitor_by_name(self, name: str) -> MonitorCSVModel:
        if name not in self.monitor_names:
            raise Flow360ValueError(
                f"Cannot find monitor with provided name={name}, available monitors: {self.monitor_names}"
            )
        return self._monitors[name]

    def __getitem__(self, name: str) -> MonitorCSVModel:
        """to support [] access"""
        return self.get_monitor_by_name(name)


UserDefinedDynamicsCSVModel = ResultCSVModel


class UserDefinedDynamicsResultModel(ResultBaseModel):
    remote_file_name: str = pd.Field(None, const=True)
    get_download_file_list_method: Optional[Callable] = pd.Field()

    _udd_names: List[str] = pd.PrivateAttr([])
    _udds: Dict[str, UserDefinedDynamicsCSVModel] = pd.PrivateAttr({})

    @property
    def udd_names(self):
        if len(self._udd_names) == 0:
            pattern = CaseDownloadable.USER_DEFINED_DYNAMICS_PATTERN.value
            file_list = [file["fileName"] for file in self.get_download_file_list_method()]
            for filename in file_list:
                if filename.startswith("results/"):
                    filename = filename.split("results/")[1]
                    match = re.match(pattern, filename)
                    if match:
                        name = match.group(1)
                        self._udd_names.append(name)
                        self._udds[name] = UserDefinedDynamicsCSVModel(remote_file_name=filename)
                        self._udds[name]._download_method = self._download_method

        return self._udd_names

    def get_udd_by_name(self, name: str) -> UserDefinedDynamicsCSVModel:
        if name not in self.udd_names:
            raise Flow360ValueError(
                f"Cannot find user defined dynamics with provided name={name}, available user defined dynamics: {self.udd_names}"
            )
        return self._udds[name]

    def __getitem__(self, name: str) -> UserDefinedDynamicsCSVModel:
        """to support [] access"""
        return self.get_udd_by_name(name)


class _DimensionedCSVResultModel(pd.BaseModel):
    _name: str

    def _in_base_component(self, base, component, component_name, params):
        log.debug(f"   -> need conversion for: {component_name} = {component}")

        flow360_conv_system = unit_converter(
            component.units.dimensions,
            params=params,
            required_by=[self._name, component_name],
        )

        if is_flow360_unit(component):
            converted = component.in_base(base, flow360_conv_system)
        else:
            component.units.registry = flow360_conv_system.registry
            converted = component.in_base(unit_system=base)
        log.debug(f"      converted to: {converted}")
        return converted


class _ActuatorDiskResults(_DimensionedCSVResultModel):
    power: PowerType.Array = pd.Field()
    force: ForceType.Array = pd.Field()
    moment: MomentType.Array = pd.Field()
    _name = "actuator_disks"

    def to_base(self, base, params):
        self.power = self._in_base_component(base, self.power, "power", params)
        self.force = self._in_base_component(base, self.force, "force", params)
        self.moment = self._in_base_component(base, self.moment, "moment", params)


class OptionallyDownloadableResultCSVModel(ResultCSVModel):
    _err_msg = "Case does not produced these results."

    def download(self, to_file: str = None, to_folder: str = ".", overwrite: bool = False):
        try:
            super().download(
                to_file=to_file, to_folder=to_folder, overwrite=overwrite, log_error=False
            )
        except CloudFileNotFoundError as err:
            if self._is_downloadable() is False:
                log.warning(self._err_msg)
            else:
                log.error(
                    (
                        "A problem occured when trying to download results:"
                        f"{self.remote_file_name}"
                    )
                )
                raise err


class ActuatorDiskResultCSVModel(OptionallyDownloadableResultCSVModel):
    remote_file_name: str = pd.Field(CaseDownloadable.ACTUATOR_DISKS.value, const=True)
    _err_msg = "Case does not have any actuator disks."

    def to_base(self, base, params: Flow360Params = None):
        """values in base system"""
        if params is None:
            params = self._get_params_method()
        disk_names = np.unique(
            [v.split("_")[0] for v in self.values.keys() if v.startswith("Disk")]
        )
        with Flow360UnitSystem(verbose=False):
            for disk_name in disk_names:
                ad = _ActuatorDiskResults(
                    power=self.values[f"{disk_name}_Power"],
                    force=self.values[f"{disk_name}_Force"],
                    moment=self.values[f"{disk_name}_Moment"],
                )
                ad.to_base(base, params)
                self.values[f"{disk_name}_Power"] = ad.power
                self.values[f"{disk_name}_Force"] = ad.force
                self.values[f"{disk_name}_Moment"] = ad.moment

                self.values[f"PowerUnits"] = ad.power.units
                self.values[f"ForceUnits"] = ad.force.units
                self.values[f"MomentUnits"] = ad.moment.units


class _BETDiskResults(_DimensionedCSVResultModel):
    force_x: ForceType.Array = pd.Field()
    force_y: ForceType.Array = pd.Field()
    force_z: ForceType.Array = pd.Field()
    moment_x: MomentType.Array = pd.Field()
    moment_y: MomentType.Array = pd.Field()
    moment_z: MomentType.Array = pd.Field()

    _name = "bet_forces"

    def to_base(self, base, params):
        self.force_x = self._in_base_component(base, self.force_x, "force_x", params)
        self.force_y = self._in_base_component(base, self.force_y, "force_y", params)
        self.force_z = self._in_base_component(base, self.force_z, "force_z", params)
        self.moment_x = self._in_base_component(base, self.moment_x, "moment_x", params)
        self.moment_y = self._in_base_component(base, self.moment_y, "moment_y", params)
        self.moment_z = self._in_base_component(base, self.moment_z, "moment_z", params)


class BETForcesResultCSVModel(OptionallyDownloadableResultCSVModel):
    remote_file_name: str = pd.Field(CaseDownloadable.BET_FORCES.value, const=True)
    _err_msg = "Case does not have any BET disks."

    def to_base(self, base, params: Flow360Params = None):
        """values in base system"""
        if params is None:
            params = self._get_params_method()
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

                self.values[f"ForceUnits"] = bet.force_x.units
                self.values[f"MomentUnits"] = bet.moment_x.units
