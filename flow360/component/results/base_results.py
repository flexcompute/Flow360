"""Base results module"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
import uuid
from itertools import chain, product
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas
import pydantic as pd

from flow360.cloud.s3_utils import (
    CloudFileNotFoundError,
    get_local_filename_and_create_folders,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.v1.flow360_params import Flow360Params
from flow360.log import log

# pylint: disable=consider-using-with
TMP_DIR = tempfile.TemporaryDirectory()

_PSEUDO_STEP = "pseudo_step"
_PHYSICAL_STEP = "physical_step"
_TIME = "time"
_TIME_UNITS = "time_units"


def _temp_file_generator(suffix: str = ""):
    random_name = str(uuid.uuid4()) + suffix
    file_path = os.path.join(TMP_DIR.name, random_name)
    return file_path


def _find_by_pattern(all_items: list, pattern):

    matched_items = []
    if pattern is not None and "*" in pattern:
        regex_pattern = pattern.replace("*", ".*")
    else:
        regex_pattern = f"^{pattern}$"  # Exact match if no '*'

    regex = re.compile(regex_pattern)
    matched_items.extend(filter(regex.match, all_items))
    return matched_items


def _filter_headers_by_prefix(
    headers: List[str],
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    suffixes: Optional[List[str]] = None,
) -> List[str]:
    """
    Filter a list of header strings based on inclusion and exclusion criteria,
    optionally enforcing that headers follow a "<prefix>_<suffix>" pattern.

    When a list of valid suffixes is provided, each header is expected to match the
    pattern "<prefix>_<suffix>", where the suffix must be one of the specified valid suffixes.
    In this case, the function extracts the prefix (the substring before the last underscore)
    and includes the header only if:
      - The extracted prefix is in the `include` list (if specified), and
      - The extracted prefix is not in the `exclude` list (if specified).

    If no suffixes are provided, the entire header is treated as the prefix and the inclusion
    and exclusion checks are performed directly on the header.

    Parameters
    ----------
    headers : List[str]
        The list of header strings to be filtered.
    include : Optional[List[str]], default=None
        A list of prefixes to include. If provided, only headers whose prefix is in this list are retained.
    exclude : Optional[List[str]], default=None
        A list of prefixes to exclude. If provided, headers whose prefix is in this list are omitted.
    suffixes : Optional[List[str]], default=None
        A list of valid suffixes. When provided, only headers that match the "<prefix>_<suffix>" pattern,
        with a suffix in this list, will be considered.

    Returns
    -------
    List[str]
        A list of headers that satisfy the inclusion/exclusion criteria.
    """

    if suffixes is None:
        pattern = re.compile(r"(.*)$")
    else:
        pattern = re.compile(r"(.*)_(" + "|".join(suffixes) + r")$")

    filtered = []
    for header in headers:
        m = pattern.match(header)
        if not m:
            continue
        prefix = m.group(1)
        if include and prefix not in include:
            continue
        if exclude and prefix in exclude:
            continue
        filtered.append(header)
    return filtered


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
    local_storage: Optional[str] = pd.Field(None)
    _download_method: Optional[Callable] = pd.PrivateAttr()
    _get_params_method: Optional[Callable] = pd.PrivateAttr()
    _is_downloadable: Optional[Callable] = pd.PrivateAttr()

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
        # pylint:disable=not-callable
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
        frozen=True, default_factory=lambda: _temp_file_generator(suffix=".csv")
    )
    _values: Optional[Dict] = pd.PrivateAttr(None)
    _raw_values: Optional[Dict] = pd.PrivateAttr(None)
    _averages: Optional[Dict] = pd.PrivateAttr(None)

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

    @classmethod
    def from_dict(cls, data: dict):
        """Load from data dictionary"""
        obj = cls()
        obj._raw_values = data
        return obj

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

        if self.local_storage is not None:
            self.download(to_folder=self.local_storage, overwrite=True, **kwargs_download)
        else:
            self.download(to_file=self.temp_file, overwrite=True, **kwargs_download)
        self._raw_values = self._read_csv_file(self.local_file_name)

    def _preprocess(self, filter_physical_steps_only: bool = False, include_time: bool = False):
        """
        run some processing after data is loaded
        """
        if self._is_physical_time_series_data() is True:
            if filter_physical_steps_only is True:
                self.filter_physical_steps_only()
            if include_time is True:
                self.include_time()

    def reload_data(self, filter_physical_steps_only: bool = False, include_time: bool = False):
        """
        Change default behavior of data loader, reload
        """
        self._values = self.raw_values
        self._preprocess(
            filter_physical_steps_only=filter_physical_steps_only, include_time=include_time
        )

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
                self._download_method(  # pylint:disable = not-callable
                    self._remote_path(),
                    to_file=to_file,
                    to_folder=to_folder,
                    overwrite=overwrite,
                    **kwargs,
                )
                self.local_file_name = local_file_path
            else:
                shutil.copy(self.local_file_name, local_file_path)
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

    def update(self, df: pandas.DataFrame):
        """Update containing value to the given DataFrame"""
        self._values = df.to_dict("list")

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
            self._preprocess()
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

    def wait(self, timeout_minutes=60):
        """
        Wait until the Case finishes processing, refresh periodically. Useful for postprocessing, eg sectional data
        """

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

    def _is_physical_time_series_data(self):
        try:
            physical_step = self.values[_PHYSICAL_STEP]
            return physical_step[-1] - physical_step[0] > 0
        except KeyError:
            return False

    def include_time(self):
        """Set the option to include time in the data"""
        if self._is_physical_time_series_data() is False:
            raise ValueError(
                "Physical time can be included only for physical time series data (unsteady simulations)"
            )

        params = self._get_params_method()  # pylint:disable = not-callable

        if isinstance(params, SimulationParams):
            try:
                step_size = params.time_stepping.step_size
            except KeyError:
                raise ValueError(  # pylint:disable=raise-missing-from
                    "Cannot find time step size for this simulation. Check simulation.json."
                )
        elif isinstance(params, Flow360Params):
            try:
                step_size = params.time_stepping.time_step_size
            except KeyError:
                raise ValueError(  # pylint:disable=raise-missing-from
                    "Cannot find time step size for this simulation. Check flow360.json file."
                )
        else:
            raise ValueError(
                f"Unknown params model: {params}, allowed (Flow360Params, SimulationParams)"
            )

        physical_step = self.as_dataframe()[_PHYSICAL_STEP]
        self.values[_TIME] = (physical_step - physical_step[0]) * step_size
        self.values[_TIME_UNITS] = step_size.units

    def filter_physical_steps_only(self):
        """
        filters data to contain only last pseudo step data for every physical step
        """
        if self._is_physical_time_series_data() is False:
            log.warning(
                "Filtering out physical steps only but there is only one step in this simulation."
            )

        df = self.as_dataframe()
        _, last_iter_mask = self._pseudo_step_masks(df)
        self.update(df[last_iter_mask])

    @classmethod
    def _pseudo_step_masks(cls, df):
        try:
            physical_step = df[_PHYSICAL_STEP]
        except KeyError:
            raise ValueError(  # pylint:disable=raise-missing-from
                "Filtering physical steps is only available for results with physical_step column."
            )
        iter_mask = np.diff(physical_step)
        first_iter_mask = np.array([1, *iter_mask]) != 0
        last_iter_mask = np.array([*iter_mask, 1]) != 0
        return first_iter_mask, last_iter_mask

    @classmethod
    def _average_last_fraction(cls, df, average_fraction):
        columns_filtered = [
            col
            for col in df.columns
            if col not in [_PSEUDO_STEP, _PHYSICAL_STEP, _TIME, _TIME_UNITS]
        ]
        selected_fraction = df[columns_filtered].tail(int(len(df) * average_fraction))
        return selected_fraction.mean()

    def get_averages(self, average_fraction):
        """Computes the average of data"""
        df = self.as_dataframe()
        return self._average_last_fraction(df, average_fraction)

    @property
    def averages(self):
        """
        Get average data over last 10%

        Returns
        -------
        dict
            Dictionary containing CL, CD, CFx/y/z, CMx/y/z and other columns available in data
        """

        if self._averages is None:
            self._averages = self.get_averages(0.1).to_dict()
        return self._averages


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


class PerEntityResultCSVModel(ResultCSVModel):
    """CSV base model for data associated with entities"""

    _variables: List[str] = []
    _x_columns: List[str] = []
    _filter_when_zero = []
    _entities: List[str] = None

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
            _ = super().values
            self._filtered_sum()
        return super().values

    @property
    def entities(self):
        """
        Returns list of entities (boundary names) available for this result
        """
        if self._entities is None:
            pattern = re.compile(rf"(.*)_({'|'.join(self._variables)})$")
            prefixes = {
                match.group(1) for col in self.as_dict().keys() if (match := pattern.match(col))
            }
            self._entities = sorted(prefixes)
        return self._entities

    def filter(self, include: list = None, exclude: list = None):
        """
        Filters entities based on include and exclude lists.

        Parameters
        ----------
        include : list or single item, optional
            List of patterns or single pattern to include.
        exclude : list or single item, optional
            List of patterns or single pattern to exclude.
        """
        self.reload_data()
        include = (
            [include] if include is not None and not isinstance(include, list) else include or []
        )
        exclude = (
            [exclude] if exclude is not None and not isinstance(exclude, list) else exclude or []
        )

        include_resolved = list(
            chain.from_iterable(_find_by_pattern(self.entities, inc) for inc in include)
        )
        exclude_resolved = list(
            chain.from_iterable(_find_by_pattern(self.entities, exc) for exc in exclude)
        )

        headers = _filter_headers_by_prefix(
            self.raw_values.keys(), include_resolved, exclude_resolved, suffixes=self._variables
        )
        self._values = {
            key: val for key, val in self.as_dict().items() if key in [*headers, *self._x_columns]
        }
        self._filtered_sum()
        self._averages = None

    def _remove_zero_rows(self, df: pandas.DataFrame) -> pandas.DataFrame:
        headers = [
            f"{x}_{y}"
            for x, y in product(self.entities, self._filter_when_zero)
            if f"{x}_{y}" in df.keys()
        ]
        if len(headers) > 0:
            df = df[df[headers].apply(lambda row: not all(row == 0), axis=1)]
        return df

    def _filtered_sum(self):
        df = self.as_dataframe()
        df = self._remove_zero_rows(df)
        if self._variables is not None:
            for variable in self._variables:
                new_col_name = "total" + variable
                regex_pattern = rf"^(?!total).*{variable}$"
                df[new_col_name] = list(df.filter(regex=regex_pattern).sum(axis=1))
        self.update(df)
