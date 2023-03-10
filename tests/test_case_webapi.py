from flow360 import Case
from flow360.log import log


from .mock_server import mock_response


def test_case(mock_response):
    case = Case(case_id="00000000-0000-0000-0000-000000000000")
    log.info(f"{case.info}")
    log.info(f"{case.params.json()}")
    log.info(f"case finished: {case.is_finished()}")

    assert case.is_finished()
    assert case.is_steady()
    assert not case.has_actuator_disks()
    assert not case.has_bet_disks()
