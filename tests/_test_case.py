from flow360 import Env
from flow360.component.case import Case
from flow360.component.flow360_params.flow360_params import Flow360Params, TimeStepping


def test_from_cloud():
    Env.dev.active()
    case = Case.from_cloud("ebc2737d-718e-46e0-b46f-2c37c69fefea")
    assert case


def test_from_volume_mesh():
    Env.dev.active()
    case = Case.submit_from_volume_mesh(
        "test_client", "05aadf62-6624-48b2-ace1-ea002d4c66c1", Flow360Params()
    )
    assert case


def test_submit_multiple_phases():
    Env.dev.active()
    cases = Case.submit_multiple_phases(
        "test_client", "05aadf62-6624-48b2-ace1-ea002d4c66c1", Flow360Params()
    )
    assert len(cases) == 1


def test_submit_multiple_3_phases():
    Env.dev.active()
    params = Flow360Params(time_stepping=TimeStepping(max_physical_steps=5))
    cases = Case.submit_multiple_phases(
        "test_client", "05aadf62-6624-48b2-ace1-ea002d4c66c1", params, phase_steps=2
    )
    assert len(cases) == 3
    assert cases[0].parent_id == "None"
    assert cases[1].parent_id == cases[0].case_id
    assert cases[2].parent_id == cases[1].case_id
