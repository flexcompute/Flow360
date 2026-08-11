"""Case results module"""

# pylint: disable=too-many-lines

from __future__ import annotations

import os
import re
from collections import defaultdict
from enum import Enum
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Tuple, get_args

import numpy as np
import pydantic as pd
from flow360_schema.framework.physical_dimensions import Force, Moment, Power
from flow360_schema.models.simulation.outputs.output_fields import (
    _CD_PER_STRIP,
    _CUMULATIVE_CD_CURVE,
    _HEAT_FLUX,
    _NORMAL_DIRECTION,
    _X,
    _Y,
    ForceOutputCoefficientNames,
    _CFx_CUMULATIVE,
    _CFx_PER_SPAN,
    _CFy_CUMULATIVE,
    _CFy_PER_SPAN,
    _CFz_CUMULATIVE,
    _CFz_PER_SPAN,
    _CMx_CUMULATIVE,
    _CMx_PER_SPAN,
    _CMy_CUMULATIVE,
    _CMy_PER_SPAN,
    _CMz_CUMULATIVE,
    _CMz_PER_SPAN,
)
from flow360_schema.models.simulation.simulation_params import SimulationParams

from flow360.component.results.base_results import (
    _PHYSICAL_STEP,
    _PSEUDO_STEP,
    _TIME,
    LocalResultCSVModel,
    NamedResultsCollectionModel,
    PerEntityResultCSVModel,
    ResultBaseModel,
    ResultCSVModel,
)
from flow360.component.results.results_utils import (
    BETDiskCSVHeaderOperation,
    DiskCoefficientsComputation,
    PorousMediumCoefficientsComputation,
)
from flow360.component.v1.flow360_params import Flow360Params
from flow360.exceptions import Flow360NotImplementedError, Flow360ValueError
from flow360.log import log


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
    POROUS_MEDIA = "porous_media_output_v2.csv"
    LEGACY_FORCE_DISTRIBUTION = "postprocess/forceDistribution.csv"
    Y_SLICING_FORCE_DISTRIBUTION = "Y_slicing_forceDistribution.csv"
    X_SLICING_FORCE_DISTRIBUTION = "X_slicing_forceDistribution.csv"

    # user defined:
    MONITOR_PATTERN = r"monitor_(.+)_v2.csv"
    USER_DEFINED_DYNAMICS_PATTERN = r"udd_(.+)_v2.csv"
    CUSTOM_FORCE_PATTERN = r"force_output_(.+)_v2.csv"
    FORCE_DISTRIBUTION_PATTERN = (
        r"(?![XY]_slicing_forceDistribution\.csv$)(.+)_forceDistribution\.csv"
    )

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

    _variables: List[str] = list(get_args(ForceOutputCoefficientNames))

    def _preprocess(self, filter_physical_steps_only: bool = True, include_time: bool = True):
        """
        run some processing after data is loaded
        """
        super()._preprocess(
            filter_physical_steps_only=filter_physical_steps_only, include_time=include_time
        )

    def reload_data(self, filter_physical_steps_only: bool = True, include_time: bool = True):
        return super().reload_data(filter_physical_steps_only, include_time)


class SurfaceForcesGroupResultCSVModel(SurfaceForcesResultCSVModel):
    """SurfaceForcesGroupResultCSVModel"""

    remote_file_name: str = pd.Field(None, frozen=True)  # Unused dummy field


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
    _x_columns: List[str] = [_Y]


class ForceDistributionCSVModel(PerEntityResultCSVModel):
    """CustomForceDistributionResultCSVModel"""

    _VARIABLES_INCREMENTAL = (
        _CFx_PER_SPAN,
        _CFy_PER_SPAN,
        _CFz_PER_SPAN,
        _CMx_PER_SPAN,
        _CMy_PER_SPAN,
        _CMz_PER_SPAN,
    )
    _VARIABLES_CUMULATIVE = (
        _CFx_CUMULATIVE,
        _CFy_CUMULATIVE,
        _CFz_CUMULATIVE,
        _CMx_CUMULATIVE,
        _CMy_CUMULATIVE,
        _CMz_CUMULATIVE,
    )
    _filter_when_zero: List[str] = []
    _variables: List[str] = []
    _x_columns: List[str] = [_NORMAL_DIRECTION]

    def _preprocess(self, filter_physical_steps_only: bool = False, include_time: bool = False):
        """
        Detect whether the data contains incremental or cumulative variables
        based on column headers, then delegate to the parent preprocessor.
        """
        headers = set(self._values.keys()) if self._values else set()
        if all(
            h in self._x_columns or any(h.endswith(suffix) for suffix in self._VARIABLES_CUMULATIVE)
            for h in headers
        ):
            self._variables = list(self._VARIABLES_CUMULATIVE)
            self._filter_when_zero = list(self._VARIABLES_CUMULATIVE)
        elif all(
            h in self._x_columns
            or any(h.endswith(suffix) for suffix in self._VARIABLES_INCREMENTAL)
            for h in headers
        ):
            self._variables = list(self._VARIABLES_INCREMENTAL)
            self._filter_when_zero = list(self._VARIABLES_INCREMENTAL)
        else:
            raise Flow360NotImplementedError(f"Unknown type of data: {headers}")

        super()._preprocess(
            filter_physical_steps_only=filter_physical_steps_only, include_time=include_time
        )


class SurfaceHeatTransferResultCSVModel(PerEntityResultCSVModel, TimeSeriesResultCSVModel):
    """SurfaceHeatTransferResultCSVModel"""

    remote_file_name: str = pd.Field(CaseDownloadable.SURFACE_HEAT_TRANSFER.value, frozen=True)
    _variables: List[str] = [_HEAT_FLUX]
    _filter_when_zero = []


class AeroacousticsResultCSVModel(TimeSeriesResultCSVModel):
    """AeroacousticsResultCSVModel"""

    _x_columns: List[str] = [_PHYSICAL_STEP, _TIME]
    remote_file_name: str = pd.Field(CaseDownloadable.AEROACOUSTICS.value, frozen=True)


MonitorCSVModel = ResultCSVModel


class MonitorsResultModel(NamedResultsCollectionModel):
    """
    Model for handling results of monitors in TAR GZ and CSV formats.

    Inherits from NamedResultsCollectionModel.
    """

    remote_file_name: str = pd.Field(CaseDownloadable.MONITORS_ALL.value, frozen=True)
    _file_name_pattern: str = CaseDownloadable.MONITOR_PATTERN.value
    _result_model_class: type = MonitorCSVModel

    def download(  # pylint:disable=arguments-differ,arguments-renamed
        self, to_file: str = None, to_folder: str = ".", overwrite: bool = False
    ):
        """
        Download the monitors TAR GZ file to the specified location.

        Parameters
        ----------
        to_file : str, optional
            The name of the file after downloading.
        to_folder : str, optional
            The folder where the file will be downloaded.
        overwrite : bool, optional
            Flag indicating whether to overwrite existing files.
        """
        ResultBaseModel.download(self, to_file=to_file, to_folder=to_folder, overwrite=overwrite)

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

    @property
    def monitor_names(self):
        """
        Get the list of monitor names.

        Returns
        -------
        list of str
            List of monitor names.
        """
        return self.names

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
        return self.get_result_by_name(name)


UserDefinedDynamicsCSVModel = TimeSeriesResultCSVModel


class UserDefinedDynamicsResultModel(NamedResultsCollectionModel):
    """
    Model for handling results of user-defined dynamics.

    Inherits from NamedResultsCollectionModel.
    """

    _file_name_pattern: str = CaseDownloadable.USER_DEFINED_DYNAMICS_PATTERN.value
    _result_model_class: type = UserDefinedDynamicsCSVModel

    @property
    def udd_names(self):
        """
        Get the list of user-defined dynamics names.

        Returns
        -------
        list of str
            List of user-defined dynamics names.
        """
        return self.names

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
        return self.get_result_by_name(name)


CustomForceCSVModel = ResultCSVModel


class CustomForceResultModel(NamedResultsCollectionModel):
    """
    Model for handling results of custom force outputs.

    Inherits from NamedResultsCollectionModel.
    """

    _file_name_pattern: str = CaseDownloadable.CUSTOM_FORCE_PATTERN.value
    _result_model_class: type = CustomForceCSVModel

    @property
    def custom_force_names(self):
        """
        Get the list of custom force output names.

        Returns
        -------
        list of str
            List of custom force output names.
        """
        return self.names

    def get_custom_force_by_name(self, name: str) -> CustomForceCSVModel:
        """
        Get custom force output by name.

        Parameters
        ----------
        name : str
            The name of the custom force output.

        Returns
        -------
        CustomForceCSVModel
            The CustomForceCSVModel corresponding to the given name.

        Raises
        ------
        Flow360ValueError
            If the custom force output with the provided name is not found.
        """
        return self.get_result_by_name(name)


class ForceDistributionsResultModel(NamedResultsCollectionModel):
    """
    Model for handling results of force distributions.
    """

    _file_name_pattern: str = CaseDownloadable.FORCE_DISTRIBUTION_PATTERN.value
    _result_model_class: type = ForceDistributionCSVModel


class _DimensionedCSVResultModel(pd.BaseModel):
    """
    Base model for handling dimensioned CSV results.

    Attributes
    ----------
    _name : str
        Name of the dimensioned CSV result.
    """

    _name: str

    @staticmethod
    def _build_flow360_unit_system(params):
        """Build a schema UnitSystem from V1 or V2 params for unit inference context."""
        # pylint: disable=import-outside-toplevel
        from flow360_schema.framework.unit_system import create_flow360_unit_system

        if isinstance(params, SimulationParams):
            return create_flow360_unit_system(
                length=params.base_length,
                velocity=params.base_velocity,
                density=params.base_density,
                temperature=params.base_temperature,
            )
        if isinstance(params, Flow360Params):
            fp = params.fluid_properties.to_fluid_properties()
            return create_flow360_unit_system(
                length=params.geometry.mesh_unit.to("m"),
                velocity=params.fluid_properties.speed_of_sound().to("m/s"),
                density=fp.density.to("kg/m**3"),
                temperature=fp.temperature.to("K"),
            )
        raise Flow360ValueError(
            f"Unknown type of params: {type(params)=}, "
            "expected one of (Flow360Params, SimulationParams)"
        )

    def _in_base_component(self, base, component, component_name):
        log.debug(f"   -> need conversion for: {component_name} = {component}")
        converted = component.in_base(unit_system=base)
        log.debug(f"      converted to: {converted}")
        return converted


class _ActuatorDiskResults(_DimensionedCSVResultModel):
    """
    Model for handling results of actuator disks.

    Inherits from _DimensionedCSVResultModel.

    Attributes
    ----------
    power : Power.Array
        Array of power values.
    force : Force.Array
        Array of force values.
    moment : Moment.Array
        Array of moment values.

    Methods
    -------
    to_base(base: Any, params: Any)
        Convert the results to the specified base system.
    """

    power: Power.Array = pd.Field()
    force: Force.Array = pd.Field()
    moment: Moment.Array = pd.Field()
    _name = "actuator_disks"

    def to_base(self, base: str):
        """
        Convert the results to the specified base system.

        Parameters
        ----------
        base : str
            The base system to convert the results to, for example SI.
        """

        self.power = self._in_base_component(base, self.power, "power")
        self.force = self._in_base_component(base, self.force, "force")
        self.moment = self._in_base_component(base, self.moment, "moment")


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
        # pylint: disable=import-outside-toplevel
        from botocore.exceptions import ClientError as CloudFileNotFoundError

        try:
            super().download(
                to_file=to_file, to_folder=to_folder, overwrite=overwrite, log_error=False, **kwargs
            )
        except CloudFileNotFoundError as err:
            if self._is_downloadable() is False:  # pylint:disable=not-callable
                log.warning(self._err_msg)
            else:
                log.error(
                    "A problem occurred when trying to download results:" f"{self.remote_file_name}"
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

    def to_base(self, base: str, params: Flow360Params | SimulationParams | None = None):
        """
        Convert the results to the specified base system.

        Parameters
        ----------
        base : str
            The base system to convert the results to. For example SI.
        params : Flow360Params | SimulationParams, optional
            Case parameters for the conversion.
        """

        if params is None:
            params = self._get_params_method()  # pylint:disable=not-callable
        disk_names = np.unique(
            [v.split("_")[0] for v in self.values.keys() if v.startswith("Disk")]
        )
        with _ActuatorDiskResults._build_flow360_unit_system(  # pylint:disable=protected-access
            params
        ):
            for disk_name in disk_names:
                disk = _ActuatorDiskResults(
                    power=self.values[f"{disk_name}_Power"],
                    force=self.values[f"{disk_name}_Force"],
                    moment=self.values[f"{disk_name}_Moment"],
                )
                disk.to_base(base)
                self.values[f"{disk_name}_Power"] = disk.power
                self.values[f"{disk_name}_Force"] = disk.force
                self.values[f"{disk_name}_Moment"] = disk.moment

                self.values["PowerUnits"] = disk.power.units
                self.values["ForceUnits"] = disk.force.units
                self.values["MomentUnits"] = disk.moment.units

    def compute_coefficients(self, params: SimulationParams) -> ActuatorDiskCoefficientsCSVModel:
        """
        Compute disk coefficients from actuator disk forces and moments.

        Parameters
        ----------
        params : SimulationParams
            Simulation parameters

        Returns
        -------
        ActuatorDiskCoefficientsCSVModel
            Model containing computed coefficients
        """
        return DiskCoefficientsComputation.compute_coefficients_static(
            params=params,
            values=self.as_dict(),
            disk_model_type="ActuatorDisk",
            iterate_step_values_func=self._iterate_step_values,
            coefficients_model_class=ActuatorDiskCoefficientsCSVModel,
        )

    @staticmethod
    def _iterate_step_values(disk_name, disk_ctx, env, values):
        # pylint:disable=too-many-locals, protected-access
        force_mag_series = values.get(f"{disk_name}_Force", [])
        moment_mag_series = values.get(f"{disk_name}_Moment", [])
        for force_mag, moment_mag in zip(force_mag_series, moment_mag_series):
            axis = disk_ctx["axis"]
            center = disk_ctx["center"]

            force_vec = force_mag * axis
            r_vec = center - env["moment_center_global"]
            moment_global = moment_mag * axis + np.cross(r_vec, force_vec)

            dp_area = env["dynamic_pressure"] * env["area"]
            denom_force = dp_area
            denom_moment = dp_area * env["moment_length_vec"]

            # pylint:disable=invalid-name
            CF_vec = force_vec / denom_force
            CM_vec = np.divide(
                moment_global, denom_moment, out=np.zeros(3), where=denom_moment != 0
            )

            CD_val = float(np.dot(force_vec, env["drag_dir"]) / denom_force)
            CL_val = float(np.dot(force_vec, env["lift_dir"]) / denom_force)
            yield CF_vec, CM_vec, CL_val, CD_val


class ActuatorDiskCoefficientsCSVModel(ResultCSVModel):
    """CSV model for actuator disk coefficients output."""

    remote_file_name: str = pd.Field("actuatorDisk_force_coefficients_v2.csv", frozen=True)


class _BETDiskResults(_DimensionedCSVResultModel):
    """
    Model for handling BET disk results.

    Inherits from _DimensionedCSVResultModel.

    Attributes
    ----------
    force_x : Force.Array
        Array of force values along the x-axis.
    force_y : Force.Array
        Array of force values along the y-axis.
    force_z : Force.Array
        Array of force values along the z-axis.
    moment_x : Moment.Array
        Array of moment values about the x-axis.
    moment_y : Moment.Array
        Array of moment values about the y-axis.
    moment_z : Moment.Array
        Array of moment values about the z-axis.
    _name : str
        Name of the BET forces result.

    Methods
    -------
    to_base(base, params)
        Convert the results to the specified base system.
    """

    force_x: Force.Array = pd.Field()
    force_y: Force.Array = pd.Field()
    force_z: Force.Array = pd.Field()
    moment_x: Moment.Array = pd.Field()
    moment_y: Moment.Array = pd.Field()
    moment_z: Moment.Array = pd.Field()

    _name = "bet_forces"

    def to_base(self, base: str):
        """
        Convert the results to the specified base system.

        Parameters
        ----------
        base : str
            The base system to convert the results to, for example SI.
        """

        self.force_x = self._in_base_component(base, self.force_x, "force_x")
        self.force_y = self._in_base_component(base, self.force_y, "force_y")
        self.force_z = self._in_base_component(base, self.force_z, "force_z")
        self.moment_x = self._in_base_component(base, self.moment_x, "moment_x")
        self.moment_y = self._in_base_component(base, self.moment_y, "moment_y")
        self.moment_z = self._in_base_component(base, self.moment_z, "moment_z")


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

    def to_base(self, base: str, params: Flow360Params | SimulationParams | None = None):
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
        with _BETDiskResults._build_flow360_unit_system(params):  # pylint:disable=protected-access
            for disk_name in disk_names:
                bet = _BETDiskResults(
                    force_x=self.values[f"{disk_name}_Force_x"],
                    force_y=self.values[f"{disk_name}_Force_y"],
                    force_z=self.values[f"{disk_name}_Force_z"],
                    moment_x=self.values[f"{disk_name}_Moment_x"],
                    moment_y=self.values[f"{disk_name}_Moment_y"],
                    moment_z=self.values[f"{disk_name}_Moment_z"],
                )
                bet.to_base(base)

                self.values[f"{disk_name}_Force_x"] = bet.force_x
                self.values[f"{disk_name}_Force_y"] = bet.force_y
                self.values[f"{disk_name}_Force_z"] = bet.force_z
                self.values[f"{disk_name}_Moment_x"] = bet.moment_x
                self.values[f"{disk_name}_Moment_y"] = bet.moment_y
                self.values[f"{disk_name}_Moment_z"] = bet.moment_z

                self.values["ForceUnits"] = bet.force_x.units
                self.values["MomentUnits"] = bet.moment_x.units

    def format_headers(
        self, params: SimulationParams, pattern: str = "$BETName_$CylinderName"
    ) -> LocalResultCSVModel:
        """
        Renames the header entries from Disk{i}_ to based on an input user pattern
        such as $BETName_$CylinderName

        Parameters
        ----------
        params : SimulationParams
            Simulation parameters
        pattern : str
            Pattern string to rename header entries. Available patterns
            [$BETName, $CylinderName, $DiskLocalIndex, $DiskGlobalIndex]

        Returns
        -------
        LocalResultCSVModel
            Model containing csv with updated header
        """
        return BETDiskCSVHeaderOperation.format_headers(self, params, pattern)

    def compute_coefficients(self, params: SimulationParams) -> BETDiskCoefficientsCSVModel:
        """
        Compute disk coefficients from BET disk forces and moments.

        Parameters
        ----------
        params : SimulationParams
            Simulation parameters

        Returns
        -------
        BETDiskCoefficientsCSVModel
            Model containing computed coefficients
        """
        return DiskCoefficientsComputation.compute_coefficients_static(
            params=params,
            values=self.as_dict(),
            disk_model_type="BETDisk",
            iterate_step_values_func=self._iterate_step_values,
            coefficients_model_class=BETDiskCoefficientsCSVModel,
        )

    @staticmethod
    def _iterate_step_values(disk_name, disk_ctx, env, values):
        # pylint:disable=protected-access, too-many-locals
        fx_series = values.get(f"{disk_name}_Force_x", [])
        fy_series = values.get(f"{disk_name}_Force_y", [])
        fz_series = values.get(f"{disk_name}_Force_z", [])
        mx_series = values.get(f"{disk_name}_Moment_x", [])
        my_series = values.get(f"{disk_name}_Moment_y", [])
        mz_series = values.get(f"{disk_name}_Moment_z", [])

        for fx, fy, fz, mx, my, mz in zip(
            fx_series, fy_series, fz_series, mx_series, my_series, mz_series
        ):

            center = disk_ctx["center"]
            force_vec = np.array([fx, fy, fz], dtype=float)
            moment_vec = np.array([mx, my, mz], dtype=float)
            r_vec = center - env["moment_center_global"]
            moment_global = moment_vec + np.cross(r_vec, force_vec)

            dp_area = env["dynamic_pressure"] * env["area"]
            denom_force = dp_area
            denom_moment = dp_area * env["moment_length_vec"]

            # pylint:disable=invalid-name
            CF_vec = force_vec / denom_force
            CM_vec = np.divide(
                moment_global, denom_moment, out=np.zeros(3), where=denom_moment != 0
            )

            CD_val = float(np.dot(force_vec, env["drag_dir"]) / denom_force)
            CL_val = float(np.dot(force_vec, env["lift_dir"]) / denom_force)
            yield CF_vec, CM_vec, CL_val, CD_val


class BETDiskCoefficientsCSVModel(ResultCSVModel):
    """CSV model for BET disk coefficients output."""

    remote_file_name: str = pd.Field("bet_force_coefficients_v2.csv", frozen=True)

    def format_headers(
        self, params: SimulationParams, pattern: str = "$BETName_$CylinderName"
    ) -> LocalResultCSVModel:
        """
        Renames the header entries from Disk{i}_ to based on an input user pattern
        such as $BETName_$CylinderName

        Parameters
        ----------
        params : SimulationParams
            Simulation parameters
        pattern : str
            Pattern string to rename header entries. Available patterns
            [$BETName, $CylinderName, $DiskLocalIndex, $DiskGlobalIndex]

        Returns
        -------
        LocalResultCSVModel
            Model containing csv with updated header
        """
        return BETDiskCSVHeaderOperation.format_headers(self, params, pattern)


class PorousMediumResultCSVModel(OptionallyDownloadableResultCSVModel):
    """Model for handling porous medium CSV results."""

    remote_file_name: str = pd.Field(CaseDownloadable.POROUS_MEDIA.value, frozen=True)
    _err_msg = "Case does not have any porous media zones."

    def compute_coefficients(self, params: SimulationParams) -> PorousMediumCoefficientsCSVModel:
        """
        Compute porous medium coefficients from forces and moments.

        Parameters
        ----------
        params : SimulationParams
            Simulation parameters

        Returns
        -------
        PorousMediumCoefficientsCSVModel
            Model containing computed coefficients
        """
        return PorousMediumCoefficientsComputation.compute_coefficients_static(
            params=params,
            values=self.as_dict(),
            iterate_step_values_func=self._iterate_step_values,
            coefficients_model_class=PorousMediumCoefficientsCSVModel,
        )

    @staticmethod
    def _iterate_step_values(zone_name, _, env, values):
        # pylint:disable=protected-access, too-many-locals
        fx_series = values.get(f"{zone_name}_Force_x", [])
        fy_series = values.get(f"{zone_name}_Force_y", [])
        fz_series = values.get(f"{zone_name}_Force_z", [])
        mx_series = values.get(f"{zone_name}_Moment_x", [])
        my_series = values.get(f"{zone_name}_Moment_y", [])
        mz_series = values.get(f"{zone_name}_Moment_z", [])

        for fx, fy, fz, mx, my, mz in zip(
            fx_series, fy_series, fz_series, mx_series, my_series, mz_series
        ):

            force_vec = np.array([fx, fy, fz], dtype=float)
            moment_vec = np.array([mx, my, mz], dtype=float)
            # Note: moment is already relative to global moment center from solver

            dp_area = env["dynamic_pressure"] * env["area"]
            denom_force = dp_area
            denom_moment = dp_area * env["moment_length_vec"]

            # pylint:disable=invalid-name
            CF_vec = force_vec / denom_force
            CM_vec = np.divide(moment_vec, denom_moment, out=np.zeros(3), where=denom_moment != 0)

            CD_val = float(np.dot(force_vec, env["drag_dir"]) / denom_force)
            CL_val = float(np.dot(force_vec, env["lift_dir"]) / denom_force)
            yield CF_vec, CM_vec, CL_val, CD_val


class PorousMediumCoefficientsCSVModel(ResultCSVModel):
    """CSV model for porous medium coefficients output."""

    remote_file_name: str = pd.Field("porous_media_force_coefficients_v2.csv", frozen=True)


class BETForcesRadialDistributionResultCSVModel(OptionallyDownloadableResultCSVModel):
    """
    Model for handling BET forces radial distribution CSV results.

    Inherits from OptionallyDownloadableResultCSVModel.
    """

    remote_file_name: str = pd.Field(
        CaseDownloadable.BET_FORCES_RADIAL_DISTRIBUTION.value, frozen=True
    )
    _err_msg = "Case does not have any BET disks."

    def format_headers(
        self, params: SimulationParams, pattern: str = "$BETName_$CylinderName"
    ) -> LocalResultCSVModel:
        """
        Renames the header entries from Disk{i}_ to based on an input user pattern
        such as $BETName_$CylinderName

        Parameters
        ----------
        params : SimulationParams
            Simulation parameters
        pattern : str
            Pattern string to rename header entries. Available patterns
            [$BETName, $CylinderName, $DiskLocalIndex, $DiskGlobalIndex]

        Returns
        -------
        LocalResultCSVModel
            Model containing csv with updated header
        """
        return BETDiskCSVHeaderOperation.format_headers(self, params, pattern)


# Render outputs are written to the visualization path rather than `results/`,
# so they cannot reuse the `results/<file>.tar.gz` downloaders.
_RENDERS_REMOTE_DIR = "visualize/renders"

# Frame PNGs in `mode="frames"` are named `<name>_<index>.png` with a zero-padded
# index of at least 4 digits (see mergeRenderOutput._MIN_FRAME_PAD), captured here
# as `<name>`. A PNG only counts as a frame when it shares its base with other
# frames (see `_populate`), so a `mode="video"` single still named e.g.
# `run_2024.png` is not mis-grouped as frame 2024 of `run`.
_RENDER_FRAME_RE = re.compile(r"^(?P<name>.+)_\d{4,}$")


def _sanitize_render_name(name: str) -> str:
    """Mirror the merger's ``sanitize_id``: render artifact filenames keep
    alphanumerics and underscores and map whitespace to ``_``, dropping anything
    else. Used so a lookup with the configured ``RenderOutput.name`` (which may
    contain spaces) resolves to the sanitized on-disk stem."""
    sanitized = ""
    for char in name:
        if char.isalnum() or char == "_":
            sanitized += char
        elif char.isspace():
            sanitized += "_"
    return sanitized


class RenderResultFileModel(ResultBaseModel):
    """
    A single render output produced under ``visualize/renders/``.

    Depending on the ``RenderOutput`` ``mode``, a render output is either a
    single MP4 (``mode="video"``) or a sequence of full-fidelity per-frame PNGs
    (``mode="frames"``). Both are represented as a list of remote files that are
    downloaded together.
    """

    remote_file_name: Optional[str] = pd.Field(None, frozen=True)
    _remote_files: List[str] = pd.PrivateAttr(default_factory=list)

    @property
    def remote_files(self) -> List[str]:
        """Remote path(s) of the file(s) backing this render output."""
        return list(self._remote_files)

    def download(  # pylint: disable=arguments-differ
        self, to_folder: str = ".", overwrite: bool = False
    ):
        """
        Download the file(s) for this render output into ``to_folder``.

        For ``mode="video"`` this is a single MP4; for ``mode="frames"`` it is
        every per-frame PNG belonging to this render output.

        Parameters
        ----------
        to_folder : str, optional
            The folder where the file(s) will be downloaded.
        overwrite : bool, optional
            Flag indicating whether to overwrite existing files.
        """
        for remote_path in self._remote_files:
            # pylint: disable=not-callable
            self._download_method(
                remote_path, to_file=None, to_folder=to_folder, overwrite=overwrite
            )


class RendersResultModel(NamedResultsCollectionModel):
    """
    Collection of render outputs stored under ``visualize/renders/``.

    Unlike the CSV/tar.gz result collections (which live under ``results/``),
    render outputs are written to the visualization path. Each render output is
    exposed by name and is either an MP4 (``mode="video"``) or a set of
    per-frame PNGs (``mode="frames"``).
    """

    _result_model_class: type = RenderResultFileModel

    def _populate(self):
        """Group the rendered artifacts under ``visualize/renders/`` by render.

        The simulation params are authoritative: they give each
        ``RenderOutput``'s name and ``mode``, so we know whether to expect an MP4
        / single still (``video``) or a PNG sequence (``frames``) and can key the
        collection by the configured name. Filenames alone are ambiguous — a
        single-frame ``frames`` render (``<name>_0000.png``) is indistinguishable
        from a ``video`` still named ``<name>_0000`` — so we only fall back to a
        filename heuristic when params are unavailable."""
        files = [
            file["fileName"]
            for file in self.get_download_file_list_method()  # pylint: disable=not-callable
            # Cloud keys are POSIX; parse them as such so the directory match
            # holds on Windows too (where `Path` would use backslashes).
            if PurePosixPath(file["fileName"]).parent.as_posix() == _RENDERS_REMOTE_DIR
        ]
        render_outputs = self._render_outputs_from_params()
        grouped = (
            self._group_by_params(render_outputs, files)
            if render_outputs
            else self._group_by_filename(files)
        )
        for name in sorted(grouped):
            self._add_render(name, grouped[name])

    def _render_outputs_from_params(self) -> List[Tuple[str, str]]:
        """Return ``[(name, mode), ...]`` for the case's RenderOutputs, or an
        empty list when params are unavailable (callers fall back to filenames)."""
        get_params = self._get_params_method
        if get_params is None:
            return []
        try:
            params = get_params()  # pylint: disable=not-callable
        # pylint: disable=broad-except
        except Exception as err:  # params may be missing/invalid; degrade gracefully
            log.debug(f"Could not read render outputs from params: {err}")
            return []
        render_outputs = []
        for output in getattr(params, "outputs", None) or []:
            if getattr(output, "output_type", None) == "RenderOutput":
                render_outputs.append((output.name, getattr(output, "mode", "video")))
        return render_outputs

    @staticmethod
    def _group_by_params(
        render_outputs: List[Tuple[str, str]], files: List[str]
    ) -> Dict[str, List[str]]:
        """Map each configured render to its artifact(s), keyed by the real
        ``RenderOutput.name``. ``frames`` renders match ``<stem>_<index>.png``
        (one or many); ``video`` renders match ``<stem>.mp4`` or ``<stem>.png``."""
        names_on_disk = [(PurePosixPath(f).name, f) for f in files]
        grouped: Dict[str, List[str]] = {}
        for name, mode in render_outputs:
            stem = _sanitize_render_name(name)
            if mode == "frames":
                frame_re = re.compile(r"^" + re.escape(stem) + r"_\d+\.png$")
                matches = [f for fname, f in names_on_disk if frame_re.match(fname)]
            else:
                wanted = {stem + ".mp4", stem + ".png"}
                matches = [f for fname, f in names_on_disk if fname in wanted]
            if matches:
                grouped[name] = matches
        return grouped

    @staticmethod
    def _group_by_filename(files: List[str]) -> Dict[str, List[str]]:
        """Fallback used only when params are unavailable. An MP4, or a PNG that
        is the only one for its base name, is keyed by its full stem; PNGs that
        share a ``<base>_<index>`` base across 2+ files are a frame sequence
        keyed by ``<base>`` (so a lone ``run_2024.png`` stays ``run_2024``)."""
        mp4_stems: Dict[str, List[str]] = {}
        png_files: List[Tuple[str, str]] = []
        for filepath in files:
            stem, ext = os.path.splitext(PurePosixPath(filepath).name)
            if ext == ".mp4":
                mp4_stems.setdefault(stem, []).append(filepath)
            elif ext == ".png":
                png_files.append((stem, filepath))

        frame_base_counts: Dict[str, int] = defaultdict(int)
        for stem, _ in png_files:
            match = _RENDER_FRAME_RE.match(stem)
            if match:
                frame_base_counts[match.group("name")] += 1

        grouped: Dict[str, List[str]] = defaultdict(list)
        for stem, filepath in png_files:
            match = _RENDER_FRAME_RE.match(stem)
            if match and frame_base_counts[match.group("name")] >= 2:
                grouped[match.group("name")].append(filepath)
            else:
                grouped[stem].append(filepath)
        for stem, paths in mp4_stems.items():
            grouped[stem].extend(paths)
        return grouped

    def _add_render(self, name: str, paths: List[str]):
        """Register one render output, wiring the download/params methods that
        Case.results injected into this collection."""
        result = self._result_model_class()
        # pylint: disable=protected-access
        result._remote_files = sorted(paths)
        result._download_method = self._download_method
        result._get_params_method = self._get_params_method
        self._names.append(name)
        self._result_collection[name] = result

    @property
    def render_names(self) -> List[str]:
        """
        Get the list of render output names.

        Returns
        -------
        list of str
            List of render output names.
        """
        return self.names

    def get_result_by_name(self, name: str) -> RenderResultFileModel:
        """Look up a render by name, tolerating the configured
        ``RenderOutput.name`` (which may contain spaces) in addition to the
        sanitized on-disk stem (e.g. ``"Volumetric wake"`` or
        ``"Volumetric_wake"``).

        The collection may be keyed by the configured name (params-driven
        grouping) or by the sanitized stem (filename fallback), so resolve a
        miss by comparing the sanitized form of the query against the sanitized
        form of each key — that matches either spelling regardless of which
        grouping path produced the keys."""
        if name not in self.names:
            sanitized = _sanitize_render_name(name)
            name = next(
                (k for k in self.names if _sanitize_render_name(k) == sanitized),
                name,
            )
        return super().get_result_by_name(name)

    def get_render_by_name(self, name: str) -> RenderResultFileModel:
        """
        Get a render output by name.

        Accepts either the configured ``RenderOutput.name`` (e.g.
        ``"Volumetric wake"``) or the sanitized on-disk stem
        (``"Volumetric_wake"``).

        Parameters
        ----------
        name : str
            The name of the render output.

        Returns
        -------
        RenderResultFileModel
            The render output corresponding to the given name.

        Raises
        ------
        Flow360ValueError
            If the render output with the provided name is not found.
        """
        return self.get_result_by_name(name)
