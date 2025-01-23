import re

import pydantic as pd
import pytest

import flow360 as fl
from flow360 import SI_unit_system, u
from flow360.component.simulation.operating_condition.operating_condition import (
    ThermalState,
)
from flow360.component.simulation.primitives import Surface


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_temperature_offset():
    ThermalState.from_standard_atmosphere(
        altitude=10 * u.m, temperature_offset=11.11 * u.delta_degC
    )

    with pytest.raises(
        pd.ValidationError,
        match=re.escape(
            r"arg '11.11 °C' does not match unit representing difference in (temperature)."
        ),
    ):
        ThermalState.from_standard_atmosphere(altitude=10 * u.m, temperature_offset=11.11 * u.degC)

    with fl.imperial_unit_system:
        ts: ThermalState = ThermalState.from_standard_atmosphere(
            altitude=10 * u.m, temperature_offset=11.11
        )
        assert ts.temperature_offset == 11.11 * u.delta_degF

    with fl.SI_unit_system:
        ts: ThermalState = ThermalState.from_standard_atmosphere(
            altitude=10 * u.m, temperature_offset=11.11
        )
        assert ts.temperature_offset == 11.11 * u.K

    with fl.CGS_unit_system:
        ts: ThermalState = ThermalState.from_standard_atmosphere(
            altitude=10 * u.m, temperature_offset=11.11
        )
        assert ts.temperature_offset == 11.11 * u.K
