import os
import tempfile
from copy import deepcopy
from itertools import product

import numpy as np
import pandas
import pytest

import flow360.component.v1.units as u1
import flow360.v1 as fl
from flow360 import log
from flow360.component.simulation import units as u2
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.utils import model_attribute_unlock

log.set_logging_level("DEBUG")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_actuator_disk_results(mock_id, mock_response):
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
    results.actuator_disks.load_from_local("data/results/actuatorDisk_output_v2.csv")

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


def test_bet_disk_results(mock_id, mock_response):
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
    results.bet_forces.load_from_local("data/results/bet_forces_v2.csv")

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


def test_bet_disk_results_with_simulation_interface(mock_id, mock_response):
    case = fl.Case(id=mock_id)

    with u2.SI_unit_system:
        params = SimulationParams(operating_condition=AerospaceCondition(velocity_magnitude=286))
        with model_attribute_unlock(params.private_attribute_asset_cache, "project_length_unit"):
            params.private_attribute_asset_cache.project_length_unit = 1 * u2.m

    results = case.results
    results.bet_forces.load_from_local("data/results/bet_forces_v2.csv")

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


@pytest.mark.usefixtures("s3_download_override")
def test_x_sectional_results(mock_id, mock_response):
    case = fl.Case(id=mock_id)
    cd_curve = case.results.x_slicing_force_distribution
    # wait for postprocessing to finish
    cd_curve.wait()

    boundaries = ["fluid/fuselage", "fluid/leftWing", "fluid/rightWing"]
    variables = ["Cumulative_CD_Curve", "CD_per_length"]
    x_columns = ["X"]
    total = [f"total{postfix}" for postfix in variables]

    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )

    assert cd_curve.as_dataframe().iloc[-1]["totalCumulative_CD_Curve"] == 0.42326354287032886
    assert set(cd_curve.values.keys()) == set(all_headers)

    cd_curve.filter(include="*Wing*")
    assert cd_curve.as_dataframe().iloc[-1]["totalCumulative_CD_Curve"] == 0.3217360243988844

    boundaries = ["fluid/leftWing", "fluid/rightWing"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(cd_curve.values.keys()) == set(all_headers)

    cd_curve.filter(exclude="*fuselage*")
    assert cd_curve.as_dataframe().iloc[-1]["totalCumulative_CD_Curve"] == 0.3217360243988844

    boundaries = ["fluid/leftWing", "fluid/rightWing"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(cd_curve.values.keys()) == set(all_headers)

    cd_curve.filter(include=["fluid/leftWing", "fluid/rightWing"])
    assert cd_curve.as_dataframe().iloc[-1]["totalCumulative_CD_Curve"] == 0.3217360243988844

    boundaries = ["fluid/leftWing", "fluid/rightWing"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(cd_curve.values.keys()) == set(all_headers)


@pytest.mark.usefixtures("s3_download_override")
def test_y_sectional_results(mock_id, mock_response):
    case = fl.Case(id=mock_id)
    y_slicing = case.results.y_slicing_force_distribution
    # wait for postprocessing to finish
    y_slicing.wait()

    boundaries = ["fluid/fuselage", "fluid/leftWing", "fluid/rightWing"]
    variables = ["CFx_per_span", "CFz_per_span", "CMy_per_span"]
    x_columns = ["Y"]
    total = [f"total{postfix}" for postfix in variables]

    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )

    assert y_slicing.as_dataframe().iloc[-1]["totalCFx_per_span"] == 0.0
    assert set(y_slicing.values.keys()) == set(all_headers)

    y_slicing.filter(include="*Wing*")
    assert y_slicing.as_dataframe().iloc[-1]["totalCFx_per_span"] == 0.0

    boundaries = ["fluid/leftWing", "fluid/rightWing"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(y_slicing.values.keys()) == set(all_headers)

    y_slicing.filter(exclude="*fuselage*")
    assert y_slicing.as_dataframe().iloc[-1]["totalCFx_per_span"] == 0.0

    boundaries = ["fluid/leftWing", "fluid/rightWing"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(y_slicing.values.keys()) == set(all_headers)

    y_slicing.filter(include=["fluid/leftWing", "fluid/rightWing"])
    assert y_slicing.as_dataframe().iloc[-1]["totalCFx_per_span"] == 0.0

    boundaries = ["fluid/leftWing", "fluid/rightWing"]
    all_headers = (
        [f"{prefix}_{postfix}" for prefix, postfix in product(boundaries, variables)]
        + x_columns
        + total
    )
    assert set(y_slicing.values.keys()) == set(all_headers)
