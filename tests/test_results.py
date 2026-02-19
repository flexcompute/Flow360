import os
import tempfile
from copy import deepcopy
from itertools import product
from typing import List

import numpy as np
import pandas
import pytest

import flow360.component.v1.units as u1
import flow360.v1 as fl
from flow360 import log
from flow360.component.results.base_results import (
    _PHYSICAL_STEP,
    _PSEUDO_STEP,
    _TIME,
    _TIME_UNITS,
    _filter_headers_by_prefix,
)
from flow360.component.results.case_results import PerEntityResultCSVModel
from flow360.component.simulation import units as u2
from flow360.component.simulation.framework.entity_expansion_utils import (
    expand_entity_list_in_context,
    get_entity_info_and_registry_from_dict,
)
from flow360.component.simulation.framework.entity_selector import (
    EntitySelector,
    expand_entity_list_selectors,
)
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.models.surface_models import BoundaryBase
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.utils import model_attribute_unlock

# log.set_logging_level("DEBUG")


def compare_dataframes_with_tolerance(
    df1: pandas.DataFrame, df2: pandas.DataFrame, rtol: float = 1e-8, atol: float = 1e-8
) -> None:
    """Compare two dataframes with numerical tolerance.

    Args:
        df1: First dataframe to compare
        df2: Second dataframe (reference) to compare against
        rtol: Relative tolerance for numerical comparison (default: 1e-5)
        atol: Absolute tolerance for numerical comparison (default: 1e-8)

    Raises:
        AssertionError: If dataframes differ beyond tolerance
    """
    # Check that columns match
    assert set(df1.columns) == set(df2.columns), (
        f"Column mismatch:\n"
        f"  df1 columns: {sorted(df1.columns)}\n"
        f"  df2 columns: {sorted(df2.columns)}\n"
        f"  Missing in df1: {set(df2.columns) - set(df1.columns)}\n"
        f"  Missing in df2: {set(df1.columns) - set(df2.columns)}"
    )

    # Check that shape matches
    assert df1.shape == df2.shape, f"Shape mismatch: {df1.shape} vs {df2.shape}"

    # Reorder columns to match
    df1 = df1[df2.columns]

    # Compare each column
    for col in df2.columns:
        if np.issubdtype(df1[col].dtype, np.number) and np.issubdtype(df2[col].dtype, np.number):
            # Numerical comparison with tolerance
            np.testing.assert_allclose(
                df1[col].values,
                df2[col].values,
                rtol=rtol,
                atol=atol,
                err_msg=f"Column '{col}' values differ beyond tolerance",
            )
        else:
            # Exact comparison for non-numerical columns
            assert df1[col].equals(df2[col]), f"Column '{col}' values differ"


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture()
def data_path(mock_id):
    return os.path.join("data", mock_id)


def test_actuator_disk_results(mock_id, mock_response, data_path):
    case = fl.Case(id=mock_id)

    with fl.SI_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry(
                mesh_unit=u1.m,
            ),
            freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
            fluid_properties=fl.air,
            boundaries={},
        )

    results = case.results
    results.actuator_disks.load_from_local(
        os.path.join(data_path, "results", "actuatorDisk_output_v2.csv")
    )

    print(results.actuator_disks.as_dataframe())
    assert results.actuator_disks.values["Disk0_Power"][0] == 30.0625485898572

    results.actuator_disks.to_base("SI", params=params)

    assert isinstance(results.actuator_disks.as_dataframe(), pandas.DataFrame)
    assert isinstance(results.actuator_disks.as_dict(), dict)
    assert isinstance(results.actuator_disks.as_numpy(), np.ndarray)

    assert float(results.actuator_disks.values["Disk0_Power"][0].v) == 1451191686.9478528
    assert str(results.actuator_disks.values["Disk0_Power"][0].units) == "kg*m**2/s**3"

    assert float(results.actuator_disks.values["Disk0_Force"][0].v) == 106613080.32014923
    assert str(results.actuator_disks.values["Disk0_Force"][0].units) == "kg*m/s**2"

    assert float(results.actuator_disks.values["Disk0_Moment"][0].v) == 1494767678.3286672
    assert str(results.actuator_disks.values["Disk0_Moment"][0].units) == "kg*m**2/s**2"

    # should be no change is calling again:
    results.actuator_disks.to_base("SI", params=params)

    assert float(results.actuator_disks.values["Disk0_Power"][0].v) == 1451191686.9478528
    assert str(results.actuator_disks.values["Disk0_Power"][0].units) == "kg*m**2/s**3"

    results.actuator_disks.to_base("Imperial", params=params)

    assert float(results.actuator_disks.values["Disk0_Power"][0].v) == 34437301746.89787
    assert str(results.actuator_disks.values["Disk0_Power"][0].units) == "ft**2*lb/s**3"


def test_bet_disk_results(mock_id, mock_response, data_path):
    case = fl.Case(id=mock_id)

    with fl.SI_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry(
                mesh_unit=u1.m,
            ),
            freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
            fluid_properties=fl.air,
            boundaries={},
        )

    results = case.results
    results.bet_forces.load_from_local(os.path.join(data_path, "results", "bet_forces_v2.csv"))

    print(results.bet_forces.as_dataframe())
    assert results.bet_forces.values["Disk0_Force_x"][0] == -1397.09615312895

    results.bet_forces.to_base("SI", params=params)

    assert isinstance(results.bet_forces.as_dataframe(), pandas.DataFrame)
    assert isinstance(results.bet_forces.as_dict(), dict)
    assert isinstance(results.bet_forces.as_numpy(), np.ndarray)

    assert float(results.bet_forces.values["Disk0_Force_x"][0].v) == -198185092.5822863
    assert str(results.bet_forces.values["Disk0_Force_x"][0].units) == "kg*m/s**2"

    assert float(results.bet_forces.values["Disk0_Moment_x"][0].v) == 23068914203.12496
    assert str(results.bet_forces.values["Disk0_Moment_x"][0].units) == "kg*m**2/s**2"


def test_bet_disk_results_with_simulation_interface(mock_id, mock_response, data_path):
    case = fl.Case(id=mock_id)

    with u2.SI_unit_system:
        params = SimulationParams(operating_condition=AerospaceCondition(velocity_magnitude=286))
        with model_attribute_unlock(params.private_attribute_asset_cache, "project_length_unit"):
            params.private_attribute_asset_cache.project_length_unit = 1 * u2.m

    results = case.results
    results.bet_forces.load_from_local(os.path.join(data_path, "results", "bet_forces_v2.csv"))

    print(results.bet_forces.as_dataframe())
    assert results.bet_forces.values["Disk0_Force_x"][0] == -1397.09615312895

    results.bet_forces.to_base("SI", params=params)

    assert isinstance(results.bet_forces.as_dataframe(), pandas.DataFrame)
    assert isinstance(results.bet_forces.as_dict(), dict)
    assert isinstance(results.bet_forces.as_numpy(), np.ndarray)

    assert float(results.bet_forces.values["Disk0_Force_x"][0].v) == -198185092.5822863
    assert str(results.bet_forces.values["Disk0_Force_x"][0].units) == "kg*m/s**2"

    assert float(results.bet_forces.values["Disk0_Moment_x"][0].v) == 23068914203.12496
    assert str(results.bet_forces.values["Disk0_Moment_x"][0].units) == "kg*m**2/s**2"

    results.bet_forces_radial_distribution.load_from_local(
        os.path.join(data_path, "results", "bet_forces_radial_distribution_v2.csv")
    )

    print(results.bet_forces_radial_distribution.as_dataframe())
    assert isinstance(results.bet_forces_radial_distribution.as_dataframe(), pandas.DataFrame)
    assert isinstance(results.bet_forces_radial_distribution.as_dict(), dict)
    assert isinstance(results.bet_forces_radial_distribution.as_numpy(), np.ndarray)

    assert (
        results.bet_forces_radial_distribution.values["Disk0_Blade0_All_ThrustCoeff"][0]
        == 0.015451537799664
    )

    assert (
        results.bet_forces_radial_distribution.values["Disk0_Blade0_All_TorqueCoeff"][0]
        == 0.0002627693012437
    )


def test_downloading(mock_id, mock_response, s3_download_override):
    case = fl.Case(id=mock_id)
    results = case.results

    with tempfile.TemporaryDirectory() as dir:
        temp_file_name = os.path.join(dir, "temp.csv")
        results.bet_forces.download(temp_file_name, overwrite=True)
        results.bet_forces.load_from_local(temp_file_name)
        assert results.bet_forces.values["Disk0_Force_x"][0] == -1397.09615312895

    case = deepcopy(fl.Case(id=mock_id))
    results = case.results
    assert results.bet_forces.values["Disk0_Force_x"][0] == -1397.09615312895

    case = deepcopy(fl.Case(id=mock_id))
    results = case.results

    with tempfile.TemporaryDirectory() as temp_dir:
        results.bet_forces.download(os.path.join(temp_dir, "bet"))
        results.bet_forces.load_from_local(os.path.join(temp_dir, "bet.csv"))
        assert results.bet_forces.values["Disk0_Force_x"][0] == -1397.09615312895


@pytest.mark.usefixtures("s3_download_override")
def test_downloader(mock_id, mock_response):
    print(mock_id)
    case = fl.Case(id=mock_id)
    results = case.results

    with tempfile.TemporaryDirectory() as temp_dir:
        results.download(all=True, destination=temp_dir)
        files = os.listdir(temp_dir)
        assert len(files) == 15
        results.total_forces.load_from_local(os.path.join(temp_dir, "total_forces_v2.csv"))
        assert results.total_forces.values["CL"][0] == 0.400770406499246

    case = deepcopy(fl.Case(id=mock_id))
    results = case.results

    with tempfile.TemporaryDirectory() as temp_dir:
        results.download(all=True, total_forces=False, destination=temp_dir)
        files = os.listdir(temp_dir)
        assert len(files) == 14

    case = deepcopy(fl.Case(id=mock_id))
    results = case.results

    with tempfile.TemporaryDirectory() as temp_dir:
        results.download(total_forces=True, destination=temp_dir)
        files = os.listdir(temp_dir)
        assert len(files) == 1
        results.total_forces.load_from_local(os.path.join(temp_dir, "total_forces_v2.csv"))
        assert results.total_forces.values["CL"][0] == 0.400770406499246


def test_include_filter_with_suffixes():
    headers = [
        "boundary_a_A",
        "boundary_a_B",
        "boundary_a_a_A",
        "boundary_a_a_B",
    ]
    result = _filter_headers_by_prefix(headers, include=["boundary_a"], suffixes=["A", "B"])
    result = _filter_headers_by_prefix(headers, include=["boundary_a"], suffixes=["A", "B"])
    assert sorted(result) == sorted(["boundary_a_A", "boundary_a_B"])

    result = _filter_headers_by_prefix(headers, exclude=["boundary_a_a"], suffixes=["A", "B"])
    assert sorted(result) == sorted(["boundary_a_A", "boundary_a_B"])


def test_include_and_exclude_filter_with_suffixes():
    headers = [
        "prefix1_A",
        "prefix2_A",
        "prefix3_A",
        "prefix1_B",
        "prefix2_B",
        "prefix3_B",
    ]
    result = _filter_headers_by_prefix(
        headers, include=["prefix1", "prefix2"], exclude=["prefix2"], suffixes=["A", "B"]
    )
    assert sorted(result) == sorted(["prefix1_A", "prefix1_B"])


def test_regex_suffix_provided_headers_with_underscore():
    headers = [
        "abc_def",
        "xyz_123",
        "nounderscore",
        "abc_xyz",
    ]
    result = _filter_headers_by_prefix(headers, include=["abc"], suffixes=[".*"])
    assert sorted(result) == sorted(["abc_def", "abc_xyz"])


def test_no_suffixes_provided_headers_with_underscore():
    headers = [
        "abc_def",
        "xyz_123",
        "nounderscore",
        "abc_xyz",
    ]
    result = _filter_headers_by_prefix(headers, include=["abc", "abc_def"])
    assert sorted(result) == sorted(["abc_def"])


def test_empty_headers():
    headers: List[str] = []
    result = _filter_headers_by_prefix(headers, include=["anything"], suffixes=["A", "B"])
    assert result == []


def test_filter():

    class TempPerEntityResultCSVModel(PerEntityResultCSVModel):
        """ForceDistributionResultCSVModel"""

        remote_file_name: str = "tempfile"
        _variables: List[str] = ["A", "B"]
        _x_columns: List[str] = ["X"]

    data = TempPerEntityResultCSVModel()
    data._raw_values = {
        "X": [0, 1],
        "boundary_a_A": [0, 1],
        "boundary_a_B": [2, 3],
        "boundary_a_a_A": [4, 5],
        "boundary_a_a_B": [6, 7],
        "boundary_aa_A": [8, 9],
        "boundary_aa_B": [10, 11],
    }

    assert data.as_dataframe()["totalA"].to_list() == [12, 15]
    assert data.as_dataframe()["totalB"].to_list() == [18, 21]
    assert sorted(data.as_dataframe().keys()) == sorted(
        [
            "X",
            "boundary_a_A",
            "boundary_a_B",
            "boundary_a_a_A",
            "boundary_a_a_B",
            "boundary_aa_A",
            "boundary_aa_B",
            "totalA",
            "totalB",
        ]
    )

    data.filter(include=["boundary_a"])

    assert data.as_dataframe()["totalA"].to_list() == [0, 1]
    assert data.as_dataframe()["totalB"].to_list() == [2, 3]
    assert sorted(data.as_dataframe().keys()) == sorted(
        [
            "X",
            "boundary_a_A",
            "boundary_a_B",
            "totalA",
            "totalB",
        ]
    )

    data.filter(exclude=["boundary_a"])

    assert data.as_dataframe()["totalA"].to_list() == [12, 14]
    assert data.as_dataframe()["totalB"].to_list() == [16, 18]
    assert sorted(data.as_dataframe().keys()) == sorted(
        [
            "X",
            "boundary_a_a_A",
            "boundary_a_a_B",
            "boundary_aa_A",
            "boundary_aa_B",
            "totalA",
            "totalB",
        ]
    )

    data.filter(exclude=["boundary_a*"])

    assert data.as_dataframe()["totalA"].to_list() == [0, 0]
    assert data.as_dataframe()["totalB"].to_list() == [0, 0]
    assert sorted(data.as_dataframe().keys()) == sorted(
        [
            "X",
            "totalA",
            "totalB",
        ]
    )

    data.filter(include=["boundary_a*"])

    assert data.as_dataframe()["totalA"].to_list() == [12, 15]
    assert data.as_dataframe()["totalB"].to_list() == [18, 21]
    assert sorted(data.as_dict().keys()) == sorted(
        [
            "X",
            "boundary_a_A",
            "boundary_a_B",
            "boundary_a_a_A",
            "boundary_a_a_B",
            "boundary_aa_A",
            "boundary_aa_B",
            "totalA",
            "totalB",
        ]
    )


@pytest.mark.usefixtures("s3_download_override")
def test_average(mock_id, mock_response):
    case = fl.Case(id="case-70489f25-d6b7-4a0b-81e1-2fa2e82fc57b")
    surface_forces = case.results.surface_forces
    data_df = surface_forces.as_dataframe()
    data_avg_dict = surface_forces.averages

    for key in [_PSEUDO_STEP, _PHYSICAL_STEP, _TIME, _TIME_UNITS]:
        assert key not in data_avg_dict.keys()

    for col in data_avg_dict.keys():
        assert compare_values(data_avg_dict[col], data_df[col].tail(int(len(data_df) * 0.1)).mean())


@pytest.mark.usefixtures("s3_download_override")
def test_x_sectional_results(mock_id, mock_response):
    case = fl.Case(id=mock_id)
    cd_curve = case.results.x_slicing_force_distribution
    cd_curve.wait()

    boundaries = ["blk-1/fuselage", "blk-1/leftWing", "blk-1/rightWing"]
    variables = ["Cumulative_CD_Curve", "CD_per_strip"]
    x_columns = ["X"]
    total = [f"total{postfix}" for postfix in variables]

    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )

    total_cd_on_all_walls = 0.0148069815193822
    assert cd_curve.as_dataframe().iloc[-1]["totalCumulative_CD_Curve"] == total_cd_on_all_walls
    assert set(cd_curve.values.keys()) == set(all_headers)
    num_total_rows = cd_curve.as_dataframe().shape[0]
    assert cd_curve.as_dataframe().shape[0] == 300

    # filter
    cd_curve.filter(include="*Wing*")
    cd_on_both_wings = 0.0104545376519996
    assert cd_curve.as_dataframe().iloc[-1]["totalCumulative_CD_Curve"] == cd_on_both_wings

    boundaries = ["blk-1/leftWing", "blk-1/rightWing"]
    all_headers_both_wings = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(cd_curve.values.keys()) == set(all_headers_both_wings)
    assert cd_curve.as_dataframe().shape[0] == 168

    cd_curve.filter(exclude="*fuselage*")
    assert cd_curve.as_dataframe().iloc[-1]["totalCumulative_CD_Curve"] == cd_on_both_wings
    assert set(cd_curve.values.keys()) == set(all_headers_both_wings)
    assert cd_curve.as_dataframe().shape[0] == 168

    cd_on_fuselage = 0.0043524438673826
    cd_curve.filter(include="*fuselage*")
    assert cd_curve.as_dataframe().iloc[-1]["totalCumulative_CD_Curve"] == cd_on_fuselage
    assert cd_curve.as_dataframe().shape[0] == 300

    cd_curve.filter(include=["blk-1/leftWing", "blk-1/rightWing"])
    assert cd_curve.as_dataframe().iloc[-1]["totalCumulative_CD_Curve"] == cd_on_both_wings

    boundaries = ["blk-1/leftWing", "blk-1/rightWing"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(cd_curve.values.keys()) == set(all_headers_both_wings)
    assert cd_curve.as_dataframe().shape[0] == 168

    cd_curve.filter(exclude=["blk-1/leftWing", "blk-1/rightWing"])
    assert cd_curve.as_dataframe().iloc[-1]["totalCumulative_CD_Curve"] == cd_on_fuselage
    assert cd_curve.as_dataframe().shape[0] == 300


@pytest.mark.usefixtures("s3_download_override")
def test_y_sectional_results(mock_id, mock_response):
    case = fl.Case(id=mock_id)
    y_slicing = case.results.y_slicing_force_distribution
    y_slicing.wait()

    boundaries = ["blk-1/fuselage", "blk-1/leftWing", "blk-1/rightWing"]
    variables = ["CFx_per_span", "CFz_per_span", "CMy_per_span"]
    x_columns = ["Y"]
    total = [f"total{postfix}" for postfix in variables]

    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )

    assert y_slicing.as_dataframe().iloc[-1]["totalCFx_per_span"] == 0.0004722955787145
    assert set(y_slicing.values.keys()) == set(all_headers)
    assert y_slicing.as_dataframe().shape[0] == 300

    y_slicing.filter(include="*Wing*")
    assert y_slicing.as_dataframe().iloc[-1]["totalCFx_per_span"] == 0.0004722955787145

    boundaries = ["blk-1/leftWing", "blk-1/rightWing"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(y_slicing.values.keys()) == set(all_headers)
    assert y_slicing.as_dataframe().shape[0] == 280

    # make sure the data excluded in the previous filter operation can still be retrieved
    y_slicing.filter(include="*fuselage*")
    assert y_slicing.as_dataframe().iloc[-1]["totalCFz_per_span"] == -0.0015624292568078

    boundaries = ["blk-1/fuselage"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(y_slicing.values.keys()) == set(all_headers)

    y_slicing.filter(exclude="*fuselage*")
    assert y_slicing.as_dataframe().iloc[-1]["totalCFx_per_span"] == 0.0004722955787145

    boundaries = ["blk-1/leftWing", "blk-1/rightWing"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(y_slicing.values.keys()) == set(all_headers)
    assert y_slicing.as_dataframe().shape[0] == 280

    y_slicing.filter(include="*fuselage*")
    assert y_slicing.as_dataframe().iloc[-1]["totalCFx_per_span"] == 0.0010109367119019

    boundaries = ["blk-1/fuselage"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(y_slicing.values.keys()) == set(all_headers)
    assert y_slicing.as_dataframe().shape[0] == 28

    y_slicing.filter(include=["blk-1/leftWing", "blk-1/rightWing"])
    assert y_slicing.as_dataframe().iloc[-1]["totalCFx_per_span"] == 0.0004722955787145

    boundaries = ["blk-1/leftWing", "blk-1/rightWing"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(y_slicing.values.keys()) == set(all_headers)
    assert y_slicing.as_dataframe().shape[0] == 280

    y_slicing.filter(include=["blk-1/leftWing"])
    assert y_slicing.as_dataframe().iloc[-1]["totalCFx_per_span"] == 0.000145645121735

    boundaries = ["blk-1/leftWing"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(y_slicing.values.keys()) == set(all_headers)
    assert y_slicing.as_dataframe().shape[0] == 140


@pytest.mark.usefixtures("s3_download_override")
def test_surface_forces_result(mock_id, mock_response):

    def validate_grouped_results(grouped_data, entity_group, flat_boundary_list, valid_names):
        """Validate that grouped surface-force results match expected aggregated values.

        Parameters
        - grouped_data: Result object exposing `as_dataframe()` and `_variables`. Its
          dataframe is expected to contain columns named "{group_name}_{variable}".
        - entity_group: Mapping from group name to the list of boundary identifiers
          that belong to the group.
        - flat_boundary_list: Ordered iterable of all boundary identifiers used to
          compute synthetic values; each identifier's index determines its weight.
        - valid_names: Iterable of group names that should be validated; other names
          are skipped (e.g., groups not part of this check).

        Method
        - For each valid group, time step, and variable, compute the expected value as
          the sum over all boundaries in that group's list that also appear in
          `flat_boundary_list` of:
              100 * (1 + boundary_index) + (1 + variable_index) + 0.0001 * (step + 1)
        - Compare the expected value with the corresponding entry in the grouped
          dataframe column "{group_name}_{variable}".

        Raises
        - AssertionError: If any computed expected value differs from the value stored
          in the grouped dataframe.
        """
        grouped_dict = grouped_data.as_dataframe().to_dict()
        variables = grouped_data._variables
        for name in entity_group.keys():
            if name not in valid_names:
                continue
            sub_boundaries = entity_group[name]
            for step in range(grouped_data.as_dataframe().shape[0]):
                for i_var in range(len(variables)):
                    expected_value = 0
                    computed_value = grouped_dict[f"{name}_{variables[i_var]}"][step]

                    for boundary_name in sub_boundaries:
                        if boundary_name not in flat_boundary_list:
                            continue
                        i_boundary = flat_boundary_list.index(boundary_name)
                        expected_value += 100 * (1 + i_boundary) + (1 + i_var) + 0.0001 * (step + 1)
                    assert expected_value == computed_value

    case = fl.Case(id="case-63fd6b73-cbd5-445c-aec6-e62eca4467e6")
    params = case.params
    surface_forces = case.results.surface_forces
    surface_forces_by_boundary = surface_forces.by_boundary_condition(params=params)

    ref_entity_group_by_boundary = {
        "Wall body 1 and 2": ["body00001", "body00002"],
        "Wall body3": ["body00003"],
        "Freestream": ["farfield"],
    }
    assert surface_forces_by_boundary._entity_groups == ref_entity_group_by_boundary
    validate_grouped_results(
        surface_forces_by_boundary,
        ref_entity_group_by_boundary,
        flat_boundary_list=("body00001", "body00002", "body00003"),
        valid_names=("Wall body 1 and 2", "Wall body3"),
    )
    ########## Body group ##########
    surface_forces_by_body_group = surface_forces.by_body_group(params=params)
    ref_entity_group_by_body_group = {
        "geo-11321727-9bb1-4fd5-b88d-19f360fb2149_box.csm": ["body00001", "body00002", "body00003"]
    }
    assert surface_forces_by_body_group._entity_groups == ref_entity_group_by_body_group
    validate_grouped_results(
        surface_forces_by_boundary,
        ref_entity_group_by_boundary,
        flat_boundary_list=("body00001", "body00002", "body00003"),
        valid_names=("geo-11321727-9bb1-4fd5-b88d-19f360fb2149_box.csm"),
    )

    params_with_dict_entities = deepcopy(params)
    for model in params_with_dict_entities.models:
        if isinstance(model, BoundaryBase):
            model.entities.stored_entities = [
                entity.model_dump() for entity in model.entities.stored_entities
            ]
    grouped_with_unexpanded_entities = surface_forces.by_boundary_condition(
        params=params_with_dict_entities
    )
    assert grouped_with_unexpanded_entities._entity_groups == ref_entity_group_by_boundary

    params_with_selectors = deepcopy(params)
    for model in params_with_selectors.models:
        if isinstance(model, BoundaryBase) and model.name == "Wall body 1 and 2":
            model.entities.stored_entities = []
            model.entities.selectors = [
                EntitySelector(
                    target_class="Surface",
                    name="select_wall_bodies",
                    logic="AND",
                    children=[
                        {
                            "attribute": "name",
                            "operator": "any_of",
                            "value": [
                                "body00001",
                                "body00002",
                                "farfield",  # farfield to ensure auto filtering
                            ],
                        }
                    ],
                )
            ]
    grouped_with_selectors = surface_forces.by_boundary_condition(params=params_with_selectors)
    assert grouped_with_selectors._entity_groups == ref_entity_group_by_boundary

    selector_model = next(
        model
        for model in params_with_selectors.models
        if isinstance(model, BoundaryBase) and model.name == "Wall body 1 and 2"
    )
    expanded_entities = expand_entity_list_in_context(
        selector_model.entities, params_with_selectors
    )
    assert {entity.name for entity in expanded_entities} == {"body00001", "body00002"}
    expanded_names = expand_entity_list_in_context(
        selector_model.entities, params_with_selectors, return_names=True
    )
    assert expanded_names == [
        "body00001",
        "body00002",
    ]  # farfield got filtered out by `Wall`'s EntityList

    serialized_params = params_with_selectors.model_dump(mode="json", exclude_none=True)
    _, registry = get_entity_info_and_registry_from_dict(serialized_params)
    expanded_entities_via_registry = expand_entity_list_selectors(
        registry,
        selector_model.entities,
        selector_cache={},
        merge_mode="merge",
    )
    dict_names = [entity.name for entity in expanded_entities_via_registry]
    assert dict_names == expanded_names


@pytest.mark.usefixtures("s3_download_override")
def test_force_distribution_result(mock_id, mock_response, data_path):
    case = fl.Case(id="case-b3927b83-8af6-49cf-aa98-00d4f1838941")
    params = case.params
    x_force_dist = case.results.x_slicing_force_distribution
    x_force_dist_by_boundary = x_force_dist.by_boundary_condition(params=params)

    # Load reference CSVs
    ref_x_force_dist_path = os.path.join(
        "data", "case-b3927b83-8af6-49cf-aa98-00d4f1838941", "results", "x_force_dist_reference.csv"
    )
    ref_x_force_dist_by_boundary_path = os.path.join(
        "data",
        "case-b3927b83-8af6-49cf-aa98-00d4f1838941",
        "results",
        "x_force_dist_by_boundary_reference.csv",
    )

    ref_x_force_dist_df = pandas.read_csv(ref_x_force_dist_path)
    ref_x_force_dist_by_boundary_df = pandas.read_csv(ref_x_force_dist_by_boundary_path)

    ref_entity_group_by_boundary = {
        "Wall-1": [
            "body00001_face00001",
            "body00001_face00002",
            "body00001_face00003",
            "body00001_face00004",
            "body00001_face00005",
            "body00001_face00006",
        ],
        "Freestream": ["farfield"],
        "Wall-23": [
            "body00002_face00001",
            "body00002_face00002",
            "body00002_face00003",
            "body00002_face00004",
            "body00002_face00005",
            "body00002_face00006",
            "body00003_face00001",
            "body00003_face00002",
            "body00003_face00003",
            "body00003_face00004",
            "body00003_face00005",
            "body00003_face00006",
        ],
    }

    assert x_force_dist_by_boundary._entity_groups == ref_entity_group_by_boundary

    # Compare current results with reference
    compare_dataframes_with_tolerance(x_force_dist.as_dataframe(), ref_x_force_dist_df)

    compare_dataframes_with_tolerance(
        x_force_dist_by_boundary.as_dataframe(),
        ref_x_force_dist_by_boundary_df,
    )

    # Test y distribution
    y_force_dist = case.results.y_slicing_force_distribution
    y_force_dist_by_body = y_force_dist.by_body_group(params=params)

    # Load reference CSVs for y distribution
    ref_y_force_dist_path = os.path.join(
        "data", "case-b3927b83-8af6-49cf-aa98-00d4f1838941", "results", "y_force_dist_reference.csv"
    )
    ref_y_force_dist_by_body_path = os.path.join(
        "data",
        "case-b3927b83-8af6-49cf-aa98-00d4f1838941",
        "results",
        "y_force_dist_by_body_reference.csv",
    )

    ref_y_force_dist_df = pandas.read_csv(ref_y_force_dist_path)
    ref_y_force_dist_by_body_df = pandas.read_csv(ref_y_force_dist_by_body_path)

    ref_entity_group_by_body = {
        "body00001": [
            "body00001_face00001",
            "body00001_face00002",
            "body00001_face00003",
            "body00001_face00004",
            "body00001_face00005",
            "body00001_face00006",
        ],
        "body00002": [
            "body00002_face00001",
            "body00002_face00002",
            "body00002_face00003",
            "body00002_face00004",
            "body00002_face00005",
            "body00002_face00006",
        ],
        "body00003": [
            "body00003_face00001",
            "body00003_face00002",
            "body00003_face00003",
            "body00003_face00004",
            "body00003_face00005",
            "body00003_face00006",
        ],
    }

    assert y_force_dist_by_body._entity_groups == ref_entity_group_by_body

    # Compare current results with reference
    compare_dataframes_with_tolerance(y_force_dist.as_dataframe(), ref_y_force_dist_df)

    compare_dataframes_with_tolerance(
        y_force_dist_by_body.as_dataframe(),
        ref_y_force_dist_by_body_df,
    )


@pytest.mark.usefixtures("s3_download_override")
def test_udd_monitors_get_params_method(mock_id, mock_response, data_path):
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")
    params = case.params

    udd = case.results.user_defined_dynamics["massInflowController_Exhaust"]
    assert udd._get_params_method() == params

    monitor = case.results.monitors["massFluxIntake"]
    assert monitor._get_params_method() == params


@pytest.mark.usefixtures("s3_download_override")
def test_custom_forces_result_model(mock_id, mock_response):
    """Test CustomForceResultModel - get custom force names and access by name"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # Test that custom_force_names property returns correct names
    custom_force_names = case.results.custom_forces.custom_force_names
    assert "wing_all_planes_forces" in custom_force_names
    assert "wing_all_planes_forces_moving_statistic" in custom_force_names
    assert len(custom_force_names) == 2

    # Test get_custom_force_by_name method
    wing_force = case.results.custom_forces.get_custom_force_by_name("wing_all_planes_forces")
    assert wing_force is not None
    assert wing_force.remote_file_name == "force_output_wing_all_planes_forces_v2.csv"

    moving_stats = case.results.custom_forces.get_custom_force_by_name(
        "wing_all_planes_forces_moving_statistic"
    )
    assert moving_stats is not None
    assert (
        moving_stats.remote_file_name
        == "force_output_wing_all_planes_forces_moving_statistic_v2.csv"
    )


@pytest.mark.usefixtures("s3_download_override")
def test_custom_forces_bracket_access(mock_id, mock_response):
    """Test CustomForceResultModel - bracket access"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # Test bracket access
    wing_force = case.results.custom_forces["wing_all_planes_forces"]
    assert wing_force is not None
    assert wing_force.remote_file_name == "force_output_wing_all_planes_forces_v2.csv"


@pytest.mark.usefixtures("s3_download_override")
def test_custom_forces_invalid_name(mock_id, mock_response):
    """Test CustomForceResultModel - invalid name raises error"""
    from flow360.exceptions import Flow360ValueError

    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # Test that invalid name raises Flow360ValueError
    with pytest.raises(Flow360ValueError) as exc_info:
        case.results.custom_forces.get_custom_force_by_name("nonExistentForce")

    assert "nonExistentForce" in str(exc_info.value)
    assert "wing_all_planes_forces" in str(exc_info.value)


@pytest.mark.usefixtures("s3_download_override")
def test_custom_forces_get_params_method(mock_id, mock_response):
    """Test CustomForceResultModel - _get_params_method is passed correctly"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")
    params = case.params

    wing_force = case.results.custom_forces["wing_all_planes_forces"]
    assert wing_force._get_params_method() == params


@pytest.mark.usefixtures("s3_download_override")
def test_custom_forces_download_method(mock_id, mock_response):
    """Test CustomForceResultModel - _download_method is passed correctly"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    wing_force = case.results.custom_forces["wing_all_planes_forces"]
    # pylint: disable=protected-access
    assert wing_force._download_method is not None
    assert callable(wing_force._download_method)


@pytest.mark.usefixtures("s3_download_override")
def test_custom_forces_data_loading(mock_id, mock_response):
    """Test CustomForceResultModel - data can be loaded from CSV"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    wing_force = case.results.custom_forces["wing_all_planes_forces"]
    # Load the data
    wing_force.reload_data()

    # Check that values are loaded
    assert wing_force.values is not None
    assert "totalCFx" in wing_force.values
    assert "totalCFy" in wing_force.values
    assert "totalCFz" in wing_force.values
    assert "totalCMy" in wing_force.values

    # Check specific values from the CSV
    cfx_values = wing_force.values["totalCFx"]
    assert len(cfx_values) > 0
    # Verify first value matches CSV data
    assert abs(cfx_values[0] - 0.130178023471194) < 1e-10


@pytest.mark.usefixtures("s3_download_override")
def test_custom_forces_moving_statistic_data_loading(mock_id, mock_response):
    """Test CustomForceResultModel - moving statistic data can be loaded from CSV"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    moving_stats = case.results.custom_forces["wing_all_planes_forces_moving_statistic"]
    # Load the data
    moving_stats.reload_data()

    # Check that values are loaded
    assert moving_stats.values is not None
    assert "totalCFx_mean" in moving_stats.values
    assert "totalCFy_mean" in moving_stats.values
    assert "totalCFz_mean" in moving_stats.values
    assert "totalCMy_mean" in moving_stats.values
    assert "window_size" in moving_stats.values

    # Check specific values from the CSV
    window_sizes = moving_stats.values["window_size"]
    assert len(window_sizes) > 0
    assert window_sizes[0] == 20


@pytest.mark.usefixtures("s3_download_override")
def test_custom_forces_is_downloadable(mock_id, mock_response):
    """Test CustomForceResultModel - _is_downloadable is set correctly"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # The _is_downloadable function should be set from has_function_map
    # pylint: disable=protected-access
    assert case.results.custom_forces._is_downloadable is not None
    assert callable(case.results.custom_forces._is_downloadable)
    # For this test case, has_custom_forces should return based on params
    # Since simulation.json doesn't have ForceOutput, it should return False
    assert case.results.custom_forces._is_downloadable() == case.has_custom_forces()


# Tests for _populate_* private methods


@pytest.mark.usefixtures("s3_download_override")
def test_monitors_populate_monitors(mock_id, mock_response):
    """Test MonitorsResultModel._populate populates names and models correctly"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # Before calling _populate, internal lists should be empty
    # pylint: disable=protected-access
    assert len(case.results.monitors._names) == 0
    assert len(case.results.monitors._result_collection) == 0

    # Call the private populate method
    case.results.monitors._populate()

    # After calling, internal lists should be populated
    assert len(case.results.monitors._names) == 2
    assert "massFluxExhaust" in case.results.monitors._names
    assert "massFluxIntake" in case.results.monitors._names

    # Check that models are created with correct remote file names
    assert len(case.results.monitors._result_collection) == 2
    assert (
        case.results.monitors._result_collection["massFluxExhaust"].remote_file_name
        == "monitor_massFluxExhaust_v2.csv"
    )
    assert (
        case.results.monitors._result_collection["massFluxIntake"].remote_file_name
        == "monitor_massFluxIntake_v2.csv"
    )


@pytest.mark.usefixtures("s3_download_override")
def test_monitors_names_property_idempotent(mock_id, mock_response):
    """Test MonitorsResultModel.monitor_names property is idempotent (accessing multiple times has same result)"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # Access property first time
    first_call_names = list(case.results.monitors.monitor_names)
    # pylint: disable=protected-access
    first_call_count = len(case.results.monitors._result_collection)

    # Access property again
    second_call_names = list(case.results.monitors.monitor_names)
    second_call_count = len(case.results.monitors._result_collection)

    assert first_call_names == second_call_names
    assert first_call_count == second_call_count


@pytest.mark.usefixtures("s3_download_override")
def test_udd_populate_udds(mock_id, mock_response):
    """Test UserDefinedDynamicsResultModel._populate populates names and models correctly"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # Before calling _populate, internal lists should be empty
    # pylint: disable=protected-access
    assert len(case.results.user_defined_dynamics._names) == 0
    assert len(case.results.user_defined_dynamics._result_collection) == 0

    # Call the private populate method
    case.results.user_defined_dynamics._populate()

    # After calling, internal lists should be populated
    assert len(case.results.user_defined_dynamics._names) == 2
    assert "massInflowController_Exhaust" in case.results.user_defined_dynamics._names
    assert "massInflowController_Intake" in case.results.user_defined_dynamics._names

    # Check that models are created with correct remote file names
    assert len(case.results.user_defined_dynamics._result_collection) == 2
    assert (
        case.results.user_defined_dynamics._result_collection[
            "massInflowController_Exhaust"
        ].remote_file_name
        == "udd_massInflowController_Exhaust_v2.csv"
    )
    assert (
        case.results.user_defined_dynamics._result_collection[
            "massInflowController_Intake"
        ].remote_file_name
        == "udd_massInflowController_Intake_v2.csv"
    )


@pytest.mark.usefixtures("s3_download_override")
def test_udd_names_property_idempotent(mock_id, mock_response):
    """Test UserDefinedDynamicsResultModel.udd_names property is idempotent"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # Access property first time
    first_call_names = list(case.results.user_defined_dynamics.udd_names)
    # pylint: disable=protected-access
    first_call_count = len(case.results.user_defined_dynamics._result_collection)

    # Access property again
    second_call_names = list(case.results.user_defined_dynamics.udd_names)
    second_call_count = len(case.results.user_defined_dynamics._result_collection)

    assert first_call_names == second_call_names
    assert first_call_count == second_call_count


@pytest.mark.usefixtures("s3_download_override")
def test_custom_forces_populate_custom_forces(mock_id, mock_response):
    """Test CustomForceResultModel._populate populates names and models correctly"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # Before calling _populate, internal lists should be empty
    # pylint: disable=protected-access
    assert len(case.results.custom_forces._names) == 0
    assert len(case.results.custom_forces._result_collection) == 0

    # Call the private populate method
    case.results.custom_forces._populate()

    # After calling, internal lists should be populated
    assert len(case.results.custom_forces._names) == 2
    assert "wing_all_planes_forces" in case.results.custom_forces._names
    assert "wing_all_planes_forces_moving_statistic" in case.results.custom_forces._names

    # Check that models are created with correct remote file names
    assert len(case.results.custom_forces._result_collection) == 2
    assert (
        case.results.custom_forces._result_collection["wing_all_planes_forces"].remote_file_name
        == "force_output_wing_all_planes_forces_v2.csv"
    )
    assert (
        case.results.custom_forces._result_collection[
            "wing_all_planes_forces_moving_statistic"
        ].remote_file_name
        == "force_output_wing_all_planes_forces_moving_statistic_v2.csv"
    )


@pytest.mark.usefixtures("s3_download_override")
def test_custom_forces_names_property_idempotent(mock_id, mock_response):
    """Test CustomForceResultModel.custom_force_names property is idempotent"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # Access property first time
    first_call_names = list(case.results.custom_forces.custom_force_names)
    # pylint: disable=protected-access
    first_call_count = len(case.results.custom_forces._result_collection)

    # Access property again
    second_call_names = list(case.results.custom_forces.custom_force_names)
    second_call_count = len(case.results.custom_forces._result_collection)

    assert first_call_names == second_call_names
    assert first_call_count == second_call_count


@pytest.mark.usefixtures("s3_download_override")
def test_udd_download_after_populate(mock_id, mock_response):
    """Test UserDefinedDynamicsResultModel.download works after _populate"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # pylint: disable=protected-access
    # First populate
    case.results.user_defined_dynamics._populate()

    # Verify download method is set on individual UDDs
    for udd_name, udd in case.results.user_defined_dynamics._result_collection.items():
        assert udd._download_method is not None
        assert callable(udd._download_method)
        assert udd._get_params_method is not None
        assert callable(udd._get_params_method)


@pytest.mark.usefixtures("s3_download_override")
def test_custom_forces_download_after_populate(mock_id, mock_response):
    """Test CustomForceResultModel.download works after _populate"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # pylint: disable=protected-access
    # First populate
    case.results.custom_forces._populate()

    # Verify download method is set on individual custom forces
    for force_name, force in case.results.custom_forces._result_collection.items():
        assert force._download_method is not None
        assert callable(force._download_method)
        assert force._get_params_method is not None
        assert callable(force._get_params_method)


@pytest.mark.usefixtures("s3_download_override")
def test_monitors_download_after_populate(mock_id, mock_response):
    """Test MonitorsResultModel download works after _populate"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # pylint: disable=protected-access
    # First populate
    case.results.monitors._populate()

    # Verify download method is set on individual monitors
    for monitor_name, monitor in case.results.monitors._result_collection.items():
        assert monitor._download_method is not None
        assert callable(monitor._download_method)
        assert monitor._get_params_method is not None
        assert callable(monitor._get_params_method)


@pytest.mark.usefixtures("s3_download_override")
def test_force_distributions_bracket_access(mock_id, mock_response):
    """Test ForceDistributionsResultModel - bracket access"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    names = case.results.force_distributions.names
    assert "force_distro_cumul" in names
    assert "ta_distro" in names
    assert len(names) == 2

    cumul = case.results.force_distributions["force_distro_cumul"]
    assert cumul is not None
    assert cumul.remote_file_name == "force_distro_cumul_forceDistribution.csv"

    incr = case.results.force_distributions["ta_distro"]
    assert incr is not None
    assert incr.remote_file_name == "ta_distro_forceDistribution.csv"


@pytest.mark.usefixtures("s3_download_override")
def test_force_distributions_invalid_name(mock_id, mock_response):
    """Test ForceDistributionsResultModel - invalid name raises error"""
    from flow360.exceptions import Flow360ValueError

    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    with pytest.raises(Flow360ValueError) as exc_info:
        case.results.force_distributions["nonexistent"]
    assert "nonexistent" in str(exc_info.value)
    assert "force_distro_cumul" in str(exc_info.value)


@pytest.mark.usefixtures("s3_download_override")
def test_force_distributions_populate(mock_id, mock_response):
    """Test ForceDistributionsResultModel._populate populates names and models correctly"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    # pylint: disable=protected-access
    assert len(case.results.force_distributions._names) == 0
    assert len(case.results.force_distributions._result_collection) == 0

    case.results.force_distributions._populate()

    assert len(case.results.force_distributions._names) == 2
    assert "force_distro_cumul" in case.results.force_distributions._names
    assert "ta_distro" in case.results.force_distributions._names

    assert len(case.results.force_distributions._result_collection) == 2
    assert (
        case.results.force_distributions._result_collection["force_distro_cumul"].remote_file_name
        == "force_distro_cumul_forceDistribution.csv"
    )
    assert (
        case.results.force_distributions._result_collection["ta_distro"].remote_file_name
        == "ta_distro_forceDistribution.csv"
    )


@pytest.mark.usefixtures("s3_download_override")
def test_force_distributions_cumulative_data_loading(mock_id, mock_response):
    """Test ForceDistributionsResultModel - cumulative data can be loaded from CSV"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    cumul = case.results.force_distributions["force_distro_cumul"]
    cumul.reload_data()
    data = cumul.as_dict()

    # Verify cumulative columns are present (CFx_cumulative etc.)
    cumulative_suffixes = [
        "CFx_cumulative",
        "CFy_cumulative",
        "CFz_cumulative",
        "CMx_cumulative",
        "CMy_cumulative",
        "CMz_cumulative",
    ]
    headers = list(data.keys())
    for suffix in cumulative_suffixes:
        assert any(
            h.endswith(suffix) for h in headers
        ), f"No column ending with '{suffix}' found in headers: {headers}"

    # Verify no incremental columns are present
    incremental_suffixes = ["CFx_per_span", "CFy_per_span", "CFz_per_span"]
    for suffix in incremental_suffixes:
        assert not any(
            h.endswith(suffix) for h in headers
        ), f"Unexpected incremental column ending with '{suffix}' found in cumulative data"


@pytest.mark.usefixtures("s3_download_override")
def test_force_distributions_incremental_data_loading(mock_id, mock_response):
    """Test ForceDistributionsResultModel - incremental (per_span) data can be loaded from CSV"""
    case = fl.Case(id="case-666666666-66666666-666-6666666666666")

    incr = case.results.force_distributions["ta_distro"]
    incr.reload_data()
    data = incr.as_dict()

    # Verify incremental columns are present (CFx_per_span etc.)
    incremental_suffixes = [
        "CFx_per_span",
        "CFy_per_span",
        "CFz_per_span",
        "CMx_per_span",
        "CMy_per_span",
        "CMz_per_span",
    ]
    headers = list(data.keys())
    for suffix in incremental_suffixes:
        assert any(
            h.endswith(suffix) for h in headers
        ), f"No column ending with '{suffix}' found in headers: {headers}"

    # Verify no cumulative columns are present
    cumulative_suffixes = ["CFx_cumulative", "CFy_cumulative", "CFz_cumulative"]
    for suffix in cumulative_suffixes:
        assert not any(
            h.endswith(suffix) for h in headers
        ), f"Unexpected cumulative column ending with '{suffix}' found in incremental data"


def test_custom_force_distribution_preprocess_detects_cumulative():
    """
    _preprocess must recognise cumulative headers when each column ends with
    exactly ONE of the cumulative suffixes (not all of them).

    Regression: the original implementation used a Cartesian-product
    comprehension (`for h in headers for suffix in suffixes`) inside `all()`,
    which required every header to endswith *every* suffix — always False.
    """
    from flow360.component.results.case_results import (
        ForceDistributionCSVModel,
    )

    model = ForceDistributionCSVModel(remote_file_name="dummy.csv")

    dummy = [0.0, 1.0]
    model._values = {
        "normal_direction": dummy,
        "farfield/fuselage_CFx_cumulative": dummy,
        "farfield/fuselage_CFy_cumulative": dummy,
        "farfield/fuselage_CFz_cumulative": dummy,
        "farfield/fuselage_CMx_cumulative": dummy,
        "farfield/fuselage_CMy_cumulative": dummy,
        "farfield/fuselage_CMz_cumulative": dummy,
    }

    model._preprocess()

    assert model._variables == [
        "CFx_cumulative",
        "CFy_cumulative",
        "CFz_cumulative",
        "CMx_cumulative",
        "CMy_cumulative",
        "CMz_cumulative",
    ]
    assert model._filter_when_zero == model._variables


def test_custom_force_distribution_preprocess_detects_incremental():
    """_preprocess must recognise incremental (per_span) headers."""
    from flow360.component.results.case_results import (
        ForceDistributionCSVModel,
    )

    model = ForceDistributionCSVModel(remote_file_name="dummy.csv")

    dummy = [0.0, 1.0]
    model._values = {
        "normal_direction": dummy,
        "wing_CFx_per_span": dummy,
        "wing_CFy_per_span": dummy,
        "wing_CFz_per_span": dummy,
        "wing_CMx_per_span": dummy,
        "wing_CMy_per_span": dummy,
        "wing_CMz_per_span": dummy,
    }

    model._preprocess()

    assert model._variables == [
        "CFx_per_span",
        "CFy_per_span",
        "CFz_per_span",
        "CMx_per_span",
        "CMy_per_span",
        "CMz_per_span",
    ]
    assert model._filter_when_zero == model._variables


def test_custom_force_distribution_preprocess_raises_on_mixed():
    """_preprocess must raise when headers mix cumulative and incremental columns."""
    from flow360.component.results.case_results import (
        ForceDistributionCSVModel,
    )
    from flow360.exceptions import Flow360NotImplementedError

    model = ForceDistributionCSVModel(remote_file_name="dummy.csv")

    dummy = [0.0, 1.0]
    model._values = {
        "normal_direction": dummy,
        "wing_CFx_cumulative": dummy,
        "wing_CFy_per_span": dummy,
    }

    with pytest.raises(Flow360NotImplementedError, match="Unknown type of data"):
        model._preprocess()


def test_force_distribution_pattern_excludes_slicing():
    """FORCE_DISTRIBUTION_PATTERN must not match X/Y slicing filenames."""
    import re

    from flow360.component.results.case_results import CaseDownloadable

    pattern = CaseDownloadable.FORCE_DISTRIBUTION_PATTERN.value

    assert re.match(pattern, "X_slicing_forceDistribution.csv") is None
    assert re.match(pattern, "Y_slicing_forceDistribution.csv") is None

    m = re.match(pattern, "wing_forceDistribution.csv")
    assert m is not None
    assert m.group(1) == "wing"

    m = re.match(pattern, "my_surface_forceDistribution.csv")
    assert m is not None
    assert m.group(1) == "my_surface"

    m = re.match(pattern, "X_slicing_custom_forceDistribution.csv")
    assert m is not None
    assert m.group(1) == "X_slicing_custom"

    m = re.match(pattern, "Y_slicing_custom_forceDistribution.csv")
    assert m is not None
    assert m.group(1) == "Y_slicing_custom"
