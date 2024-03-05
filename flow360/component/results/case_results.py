""" Case results module"""

import os
import re
import shutil
import tempfile
import time
import uuid
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

# pylint: disable=consider-using-with
TMP_DIR = tempfile.TemporaryDirectory()


def _temp_file_generator(suffix: str = ""):
    random_name = str(uuid.uuid4()) + suffix
    file_path = os.path.join(TMP_DIR.name, random_name)
    return file_path


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


class ResultBaseModel(pd.BaseModel):
    """
    Base model for handling results.

    Parameters
    ----------
    remote_file_name : str
        The name of the file stored remotely.
    local_file_name : str, optional
        The name of the file stored locally.
    do_download : bool, optional
        Flag indicating whether to perform the download.
    _download_method : Callable, optional
        The method responsible for downloading the file.
    _get_params_method : Callable, optional
        The method to get Case parameters.
    _is_downloadable : Callable, optional
        Function to determine if the file is downloadable.

    Methods
    -------
    download(to_file: str = None, to_folder: str = ".", overwrite: bool = False)
        Download the file to the specified location.

    """

    remote_file_name: str = pd.Field()
    local_file_name: str = pd.Field(None)
    do_download: Optional[bool] = pd.Field(None)
    _download_method: Optional[Callable] = pd.PrivateAttr()
    _get_params_method: Optional[Callable] = pd.PrivateAttr()
    _is_downloadable: Callable = pd.PrivateAttr(lambda: True)

    def download(self, to_file: str = None, to_folder: str = ".", overwrite: bool = False):
        """
        Download the file to the specified location.

        Parameters
        ----------
        to_file : str, optional
            The name of the file after downloading.
        to_folder : str, optional
            The folder where the file will be downloaded.
        overwrite : bool, optional
            Flag indicating whether to overwrite existing files.
        """

        self._download_method(
            self._remote_path(), to_file=to_file, to_folder=to_folder, overwrite=overwrite
        )

    def _remote_path(self):
        return f"results/{self.remote_file_name}"


class ResultCSVModel(ResultBaseModel):
    """
    Model for handling CSV results.

    Parameters
    ----------
    temp_file : str
        Path to the temporary CSV file.
    _values : dict, optional
        Internal storage for the CSV data.
    _raw_values : dict, optional
        Internal storage for the raw CSV data.

    Methods
    -------
    load_from_local(filename: str)
        Load CSV data from a local file.
    load_from_remote(**kwargs_download)
        Load CSV data from a remote source.
    download(to_file: str = None, to_folder: str = ".", overwrite: bool = False, **kwargs)
        Download the CSV file.
    values
        Get the CSV data.
    to_base(base)
        Convert the CSV data to a different base system.
    to_file(filename: str = None)
        Save the data to a CSV file.
    as_dict()
        Convert the data to a dictionary.
    as_numpy()
        Convert the data to a NumPy array.
    as_dataframe()
        Convert the data to a Pandas DataFrame.
    """

    temp_file: str = pd.Field(
        const=True, default_factory=lambda: _temp_file_generator(suffix=".csv")
    )
    _values: Optional[Dict] = pd.PrivateAttr(None)
    _raw_values: Optional[Dict] = pd.PrivateAttr(None)

    def _read_csv_file(self, filename: str):
        dataframe = pandas.read_csv(filename, skipinitialspace=True)
        dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
        return dataframe.to_dict("list")

    @property
    def raw_values(self):
        """
        Get the raw CSV data.

        Returns
        -------
        dict
            Dictionary containing the raw CSV data.
        """

        if self._raw_values is None:
            self.load_from_remote()
        return self._raw_values

    def load_from_local(self, filename: str):
        """
        Load CSV data from a local file.

        Parameters
        ----------
        filename : str
            Path to the local CSV file.
        """

        self._raw_values = self._read_csv_file(filename)
        self.local_file_name = filename

    def load_from_remote(self, **kwargs_download):
        """
        Load CSV data from a remote source.
        """

        self.download(to_file=self.temp_file, overwrite=True, **kwargs_download)
        self._raw_values = self._read_csv_file(self.temp_file)
        self.local_file_name = self.temp_file

    def download(
        self, to_file: str = None, to_folder: str = ".", overwrite: bool = False, **kwargs
    ):
        """
        Download the CSV file.

        Parameters
        ----------
        to_file : str, optional
            The name of the file after downloading.
        to_folder : str, optional
            The folder where the file will be downloaded.
        overwrite : bool, optional
            Flag indicating whether to overwrite existing files.
        """

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

    def __str__(self):
        res_str = self.as_dataframe().__str__()
        res_str += "\nif you want to get access to data, use one of the data format functions:"
        res_str += "\n .as_dataframe()\n .as_dict()\n .as_numpy()"
        return res_str

    def __repr__(self):
        res_str = self.as_dataframe().__repr__()
        res_str += "\nif you want to get access to data, use one of the data format functions:"
        res_str += "\n .as_dataframe()\n .as_dict()\n .as_numpy()"
        return res_str

    @property
    def values(self):
        """
        Get the current data.

        Returns
        -------
        dict
            Dictionary containing the current data.
        """

        if self._values is None:
            self._values = self.raw_values
        return self._values

    def to_base(self, base: str):
        """
        Convert the CSV data to a different base system.

        Parameters
        ----------
        base : str
            The base system to which the CSV data will be converted, for example SI
        """
        raise ValueError(f"You cannot convert these results to {base}, the method is not defined.")

    def to_file(self, filename: str = None):
        """
        Save the data to a CSV file.

        Parameters
        ----------
        filename : str, optional
            The name of the file to save the CSV data.
        """

        self.as_dataframe().to_csv(filename, index=False)
        log.info(f"Saved to {filename}")

    def as_dict(self):
        """
        Convert the data to a dictionary.

        Returns
        -------
        dict
            Dictionary containing the data.
        """

        return self.values

    def as_numpy(self):
        """
        Convert the data to a NumPy array.

        Returns
        -------
        numpy.ndarray
            NumPy array containing the data.
        """

        return self.as_dataframe().to_numpy()

    def as_dataframe(self):
        """
        Convert the data to a Pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the data.
        """

        return pandas.DataFrame(self.values)


class ResultTarGZModel(ResultBaseModel):
    """
    Model for handling TAR GZ results.

    Inherits from ResultBaseModel.

    Methods
    -------
    to_file(filename: str, overwrite: bool = False)
        Save the TAR GZ file.

    """

    def to_file(self, filename, overwrite: bool = False):
        """
        Save the TAR GZ file.

        Parameters
        ----------
        filename : str
            The name of the file to save the TAR GZ data.
        overwrite : bool, optional
            Flag indicating whether to overwrite existing files.
        """

        self.download(to_file=filename, overwrite=overwrite)


# separate classes used to further customise give resutls, for example nonlinear_residuals.plot()
class NonlinearResidualsResultCSVModel(ResultCSVModel):
    """NonlinearResidualsResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.NONLINEAR_RESIDUALS.value, const=True)


class LinearResidualsResultCSVModel(ResultCSVModel):
    """LinearResidualsResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.LINEAR_RESIDUALS.value, const=True)


class CFLResultCSVModel(ResultCSVModel):
    """CFLResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.CFL.value, const=True)


class MinMaxStateResultCSVModel(ResultCSVModel):
    """CFLResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.MINMAX_STATE.value, const=True)


class MaxResidualLocationResultCSVModel(ResultCSVModel):
    """MaxResidualLocationResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.MAX_RESIDUAL_LOCATION.value, const=True)


class TotalForcesResultCSVModel(ResultCSVModel):
    """TotalForcesResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.TOTAL_FORCES.value, const=True)


class SurfaceForcesResultCSVModel(ResultCSVModel):
    """SurfaceForcesResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.SURFACE_FORCES.value, const=True)


class ForceDistributionResultCSVModel(ResultCSVModel):
    """ForceDistributionResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.FORCE_DISTRIBUTION.value, const=True)

    def wait(self, timeout_minutes=60):
        """Wait until the Case finishes processing, refresh periodically"""

        start_time = time.time()
        while time.time() - start_time < timeout_minutes * 60:
            try:
                self.load_from_remote(log_error=False)
                return None
            except CloudFileNotFoundError:
                pass
            time.sleep(2)

        raise TimeoutError(
            "Timeout: post-processing did not finish within the specified timeout period."
        )


class SurfaceHeatTrasferResultCSVModel(ResultCSVModel):
    """SurfaceHeatTrasferResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.SURFACE_HEAT_TRANSFER.value, const=True)


class AeroacousticsResultCSVModel(ResultCSVModel):
    """AeroacousticsResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.AEROACOUSTICS.value, const=True)


MonitorCSVModel = ResultCSVModel


class MonitorsResultModel(ResultTarGZModel):
    """
    Model for handling results of monitors in TAR GZ and CSV formats.

    Inherits from ResultTarGZModel.
    """

    remote_file_name: str = pd.Field(CaseDownloadable.MONITORS_ALL.value, const=True)
    get_download_file_list_method: Optional[Callable] = pd.Field()

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
            file_list = [file["fileName"] for file in self.get_download_file_list_method()]
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


UserDefinedDynamicsCSVModel = ResultCSVModel


class UserDefinedDynamicsResultModel(ResultBaseModel):
    """
    Model for handling results of user-defined dynamics.

    Inherits from ResultBaseModel.
    """

    remote_file_name: str = pd.Field(None, const=True)
    get_download_file_list_method: Optional[Callable] = pd.Field()

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
            file_list = [file["fileName"] for file in self.get_download_file_list_method()]
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

    remote_file_name: str = pd.Field(CaseDownloadable.ACTUATOR_DISKS.value, const=True)
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
            params = self._get_params_method()
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

    remote_file_name: str = pd.Field(CaseDownloadable.BET_FORCES.value, const=True)
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

                self.values["ForceUnits"] = bet.force_x.units
                self.values["MomentUnits"] = bet.moment_x.units
