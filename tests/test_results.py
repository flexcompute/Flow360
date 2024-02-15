import pytest
import tempfile
import os
import flow360 as fl
import flow360.units as u
from flow360 import log
from flow360.component.results.case_results import ActuatorDiskResultCSVModel
import pandas
import numpy as np
from copy import deepcopy

from .mock_server import mock_response
from .utils import mock_id, s3_download_override


log.set_logging_level("DEBUG")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)



# def test_actuator_disk_results(mock_response):
#     case = fl.Case(id=mock_id)

#     with fl.SI_unit_system:
#         params = fl.Flow360Params(
#             geometry=fl.Geometry(
#                 mesh_unit=u.m,
#             ),
#             freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
#             fluid_properties=fl.air,
#             boundaries={},
#         )

#     results = case.results
#     results.actuator_disks.load_from_local('data/results/actuatorDisk_output_v2.csv')

#     print(results.actuator_disks.as_dataframe())
#     assert results.actuator_disks.values['Disk0_Power'][0] == 30.0625485898572

#     results.actuator_disks.to_base("SI", params=params)
    
#     assert isinstance(results.actuator_disks.as_dataframe(), pandas.DataFrame)
#     assert isinstance(results.actuator_disks.as_dict(), dict)
#     assert isinstance(results.actuator_disks.as_numpy(), np.ndarray)


#     assert float(results.actuator_disks.values['Disk0_Power'][0].v) == 1451191686.9478528
#     assert str(results.actuator_disks.values['Disk0_Power'][0].units) == 'kg*m**2/s**3'

#     assert float(results.actuator_disks.values['Disk0_Force'][0].v) == 106613080.32014923
#     assert str(results.actuator_disks.values['Disk0_Force'][0].units) == 'kg*m/s**2'

#     assert float(results.actuator_disks.values['Disk0_Moment'][0].v) == 1494767678.3286672
#     assert str(results.actuator_disks.values['Disk0_Moment'][0].units) == 'kg*m**2/s**2'


#     # should be no change is calling again:
#     results.actuator_disks.to_base("SI", params=params)

#     assert float(results.actuator_disks.values['Disk0_Power'][0].v) == 1451191686.9478528
#     assert str(results.actuator_disks.values['Disk0_Power'][0].units) == 'kg*m**2/s**3'

#     results.actuator_disks.to_base("Imperial", params=params)

#     assert float(results.actuator_disks.values['Disk0_Power'][0].v) == 34437301746.89787
#     assert str(results.actuator_disks.values['Disk0_Power'][0].units) == 'ft**2*lb/s**3'


# def test_bet_disk_results(mock_response):
#     case = fl.Case(id=mock_id)

#     with fl.SI_unit_system:
#         params = fl.Flow360Params(
#             geometry=fl.Geometry(
#                 mesh_unit=u.m,
#             ),
#             freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
#             fluid_properties=fl.air,
#             boundaries={},
#         )

#     results = case.results
#     results.bet_forces.load_from_local('data/results/bet_forces_v2.csv')

#     print(results.bet_forces.as_dataframe())
#     assert results.bet_forces.values['Disk0_Force_x'][0] == -1397.09615312895

#     results.bet_forces.to_base("SI", params=params)
    
#     assert isinstance(results.bet_forces.as_dataframe(), pandas.DataFrame)
#     assert isinstance(results.bet_forces.as_dict(), dict)
#     assert isinstance(results.bet_forces.as_numpy(), np.ndarray)

#     assert float(results.bet_forces.values['Disk0_Force_x'][0].v) == -198185092.5822863
#     assert str(results.bet_forces.values['Disk0_Force_x'][0].units) == 'kg*m/s**2'

#     assert float(results.bet_forces.values['Disk0_Moment_x'][0].v) ==  23068914203.12496
#     assert str(results.bet_forces.values['Disk0_Moment_x'][0].units) == 'kg*m**2/s**2'



@pytest.mark.usefixtures("s3_download_override")
def test_downloading(mock_response):
    case = fl.Case(id=mock_id)
    results = case.results

    with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
        results.bet_forces.download(temp_file.name, overwrite=True)
        results.bet_forces.load_from_local(temp_file.name)
        assert results.bet_forces.values['Disk0_Force_x'][0] == -1397.09615312895


    case = deepcopy(fl.Case(id=mock_id))
    results = case.results
    assert results.bet_forces.values['Disk0_Force_x'][0] == -1397.09615312895

    case = deepcopy(fl.Case(id=mock_id))
    results = case.results

    with tempfile.TemporaryDirectory() as temp_dir:
        results.bet_forces.download(os.path.join(temp_dir, 'bet'))
        results.bet_forces.load_from_local(os.path.join(temp_dir, 'bet.csv'))
        assert results.bet_forces.values['Disk0_Force_x'][0] == -1397.09615312895