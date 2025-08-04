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
from flow360.component.simulation.framework.updater_utils import (
    compare_dicts,
    compare_values,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.utils import model_attribute_unlock

log.set_logging_level("DEBUG")


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
        assert len(files) == 14
        results.total_forces.load_from_local(os.path.join(temp_dir, "total_forces_v2.csv"))
        assert results.total_forces.values["CL"][0] == 0.400770406499246

    case = deepcopy(fl.Case(id=mock_id))
    results = case.results

    with tempfile.TemporaryDirectory() as temp_dir:
        results.download(all=True, total_forces=False, destination=temp_dir)
        files = os.listdir(temp_dir)
        assert len(files) == 13

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


def test_no_suffixes_provided_no_include_filter():
    headers = [
        "abc_def",
        "xyz_123",
        "nounderscore",
        "abc_xyz",
    ]
    result = _filter_headers_by_prefix(headers)
    assert sorted(result) == sorted(
        [
            "abc_def",
            "xyz_123",
            "nounderscore",
            "abc_xyz",
        ]
    )


test_no_suffixes_provided_no_include_filter()


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
    x_columns = ["Y", "stride"]
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
    case = fl.Case(id="case-63fd6b73-cbd5-445c-aec6-e62eca4467e6")
    params = case.params
    entity_info = params.private_attribute_asset_cache.project_entity_info
    surface_forces = case.results.surface_forces
    surface_forces_by_boundary = surface_forces.by_boundary_condition(params=params)

    def compare_surface_force_groups(surface_forces, surface_forces_group):
        surface_forces_group_df = surface_forces_group.as_dataframe()
        for groupName, faces in surface_forces_group._entity_groups.items():
            surface_forces.filter(include=faces)
            total_force_faces_df = surface_forces.as_dataframe()
            for force_name in ["CL", "CD", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz"]:
                assert compare_values(
                    total_force_faces_df.iloc[-1][f"total{force_name}"],
                    surface_forces_group_df.iloc[-1][f"{groupName}_{force_name}"],
                )

    ref_entity_group_by_boundary = {
        "Wall": ["boundary2", "boundary3"],
        "Freestream": ["farfield"],
        "Slip wall": ["boundary1"],
    }
    assert compare_dicts(surface_forces_by_boundary._entity_groups, ref_entity_group_by_boundary)
    compare_surface_force_groups(surface_forces, surface_forces_by_boundary)

    surface_forces_by_body_group = surface_forces.by_body_group(params=params)
    ref_entity_group_by_body_group = {
        "two_boxes_conflict.csm": ["boundary1", "boundary2", "boundary3"]
    }
    assert compare_dicts(
        surface_forces_by_body_group._entity_groups, ref_entity_group_by_body_group
    )
    compare_surface_force_groups(surface_forces, surface_forces_by_body_group)

    entity_info._group_entity_by_tag("body", "bodyId")
    with pytest.raises(
        ValueError,
        match=r"Face group 'boundary2' contains faces belonging to multiple body groups: \['body00001', 'body00002'\]. The mapping between body and face groups cannot be created.",
    ):
        surface_forces.by_body_group(params=params)

    # For force distribution
