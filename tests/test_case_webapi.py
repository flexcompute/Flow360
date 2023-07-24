import pytest

from flow360 import Case
from flow360.exceptions import RuntimeError
from flow360.log import Logger, log

from .mock_server import mock_response
from .utils import mock_id

Logger.log_to_file = False


def test_case(mock_response):
    case = Case(id=mock_id)
    log.info(f"{case.info}")
    log.info(f"{case.params.json()}")
    log.info(f"case finished: {case.is_finished()}")
    log.info(f"case parent (parent={case.info.parent_id}): {case.has_parent()}")

    assert case.is_finished()
    assert case.is_steady()
    assert not case.has_actuator_disks()
    assert not case.has_bet_disks()
    assert not case.has_parent()
    with pytest.raises(RuntimeError):
        case.parent


Logger.log_to_file = True
