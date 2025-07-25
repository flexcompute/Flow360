import pytest

from flow360.exceptions import Flow360RuntimeError
from flow360.log import Logger, log
from flow360.v1 import Case

Logger.log_to_file = False


def test_case(mock_id, mock_response):
    case = Case(id=mock_id)
    log.info(f"{case.info}")
    log.info(f"{case.params.json()}")
    log.info(f"case finished: {case.status.is_final()}")
    log.info(f"case parent (parent={case.info.parent_id}): {case.has_parent()}")

    assert case.status.is_final()
    assert case.is_steady()
    assert not case.has_actuator_disks()
    assert not case.has_bet_disks()
    assert not case.has_parent()
    with pytest.raises(Flow360RuntimeError):
        print(case.parent)


Logger.log_to_file = True
