import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.outputs.outputs import AeroAcousticOutput


def test_aeroacoustic_output_observer_converter():
    model = AeroAcousticOutput(
        name="test", observers=[[0.2, 0.02, 0.03] * u.m, [0.0001, 0.02, 0.03] * u.m]
    )
    assert all(model.observers[0].position == [0.2, 0.02, 0.03] * u.m)
    assert model.observers[0].group_name == "0"
    assert all(model.observers[1].position == [0.0001, 0.02, 0.03] * u.m)
    assert model.observers[1].group_name == "0"


def test_aeroacoustic_observer_unit_validator():
    with pytest.raises(
        ValueError,
        match="All observer locations should have the same unit. But now it has both `cm` and `mm`.",
    ):
        AeroAcousticOutput(
            name="test", observers=[[0.2, 0.02, 0.03] * u.cm, [0.0001, 0.02, 0.03] * u.mm]
        )
