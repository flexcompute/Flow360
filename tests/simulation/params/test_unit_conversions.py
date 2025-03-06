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


def test_unit_conversions():

    with SI_unit_system:
        far_field_zone = fl.AutomatedFarfield()
        params = fl.SimulationParams(
            meshing=fl.MeshingParams(
                defaults=fl.MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                    surface_max_edge_length=1,
                ),
                volume_zones=[far_field_zone],
            ),
            reference_geometry=fl.ReferenceGeometry(),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=100,
                alpha=5 * u.deg,
            ),
            time_stepping=fl.Steady(max_steps=1000),
            models=[
                fl.Wall(
                    surfaces=[Surface(name="surface")],
                    name="Wall",
                ),
                fl.Freestream(
                    surfaces=[far_field_zone.farfield],
                    name="Freestream",
                ),
            ],
        )

    mach = 0.2938635365101296
    velocity = 100
    converted = params.convert_unit(value=velocity * u.m / u.s, target_system="flow360")
    assert float(converted.value) == mach
    assert str(converted.units) == "flow360_velocity_unit"

    converted = params.convert_unit(value=converted, target_system="SI")
    assert float(converted.value) == velocity
    assert str(converted.units) == "m/s"

    converted = params.convert_unit(value=mach * u.flow360_velocity_unit, target_system="SI")
    assert float(converted.value) == velocity
    assert str(converted.units) == "m/s"

    converted = params.convert_unit(value=velocity * u.m / u.s, target_system="SI")
    assert float(converted.value) == velocity
    assert str(converted.units) == "m/s"

    converted = params.convert_unit(value=mach * u.flow360_velocity_unit, target_system="flow360")
    assert float(converted.value) == mach
    assert str(converted.units) == "flow360_velocity_unit"

    converted = params.convert_unit(value=velocity * u.m / u.s, target_system="Imperial")
    assert float(converted.value) == 328.0839895013123
    assert str(converted.units) == "ft/s"

    converted = params.convert_unit(value=328.0839895013123 * u.ft / u.s, target_system="SI")
    assert float(converted.value) == 100
    assert str(converted.units) == "m/s"

    pressure_flow360 = 1 / 1.4
    pressure = 101325.009090375
    converted = params.convert_unit(
        value=pressure * u.Pa, target_system="flow360", length_unit=1 * u.m
    )
    assert float(converted.value) == pressure_flow360
    assert str(converted.units) == "flow360_pressure_unit"

    converted = params.convert_unit(value=converted, target_system="SI")
    assert float(converted.value) == pressure
    assert str(converted.units) == "kg/(m*s**2)"

    converted = params.convert_unit(
        value=pressure_flow360 * u.flow360_pressure_unit, target_system="SI"
    )
    assert float(converted.value) == pressure
    assert str(converted.units) == "kg/(m*s**2)"


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
