import json
from typing import Optional, Union

import pydantic.v1 as pd
import pytest

import flow360 as fl
from flow360 import units as u
from flow360.component.flow360_params.params_base import Flow360BaseModel
from flow360.component.flow360_params.unit_system import (
    AngularVelocityType,
    AreaType,
    DensityType,
    ForceType,
    LengthType,
    MassType,
    PressureType,
    TemperatureType,
    TimeType,
    VelocityType,
    ViscosityType,
)
from tests.utils import to_file_from_file_test


class DataWithUnits(pd.BaseModel):
    L: LengthType = pd.Field()
    m: MassType = pd.Field()
    t: TimeType = pd.Field()
    T: TemperatureType = pd.Field()
    v: VelocityType = pd.Field()
    A: AreaType = pd.Field()
    F: ForceType = pd.Field()
    p: PressureType = pd.Field()
    r: DensityType = pd.Field()
    mu: ViscosityType = pd.Field()
    omega: AngularVelocityType = pd.Field()


class DataWithUnitsConstrained(pd.BaseModel):
    L: Optional[LengthType.NonNegative] = pd.Field()
    m: MassType.Positive = pd.Field()
    t: TimeType.Negative = pd.Field()
    T: TemperatureType.NonNegative = pd.Field()
    v: VelocityType.NonNegative = pd.Field()
    A: AreaType.Positive = pd.Field()
    F: ForceType.NonPositive = pd.Field()
    p: Union[PressureType.Constrained(ge=5, lt=9), PressureType.Constrained(ge=10, lt=12)] = (
        pd.Field()
    )
    r: DensityType = pd.Field()
    mu: ViscosityType.Constrained(ge=2) = pd.Field()
    omega: AngularVelocityType.NonNegative = pd.Field()


class VectorDataWithUnits(pd.BaseModel):
    pt: Optional[LengthType.Point] = pd.Field()
    vec: Union[VelocityType.Direction, ForceType.Point] = pd.Field()
    ax: LengthType.Axis = pd.Field()
    omega: AngularVelocityType.Moment = pd.Field()


class Flow360DataWithUnits(Flow360BaseModel):
    L: LengthType = pd.Field()
    pt: Optional[LengthType.Point] = pd.Field()
    lc: LengthType.NonNegative = pd.Field()


def test_unit_access():
    assert fl.units.flow360_area_unit
    assert fl.units.CGS_unit_system
    assert u.CGS_unit_system
    assert fl.units.kg
    assert fl.units.inch
    assert u.inch


def test_unit_systems_compare():
    assert fl.SI_unit_system != fl.flow360_unit_system
    assert fl.SI_unit_system != fl.CGS_unit_system

    assert fl.SI_unit_system == fl.SI_unit_system
    assert fl.flow360_unit_system == fl.flow360_unit_system

    assert fl.flow360_unit_system == fl.UnitSystem(base_system="Flow360")
    assert fl.SI_unit_system == fl.UnitSystem(base_system="SI")


@pytest.mark.usefixtures("array_equality_override")
def test_flow360_unit_arithmetic():
    assert 1 * u.flow360_area_unit
    assert u.flow360_area_unit * 1

    assert u.flow360_area_unit == u.flow360_area_unit
    assert u.flow360_area_unit != u.flow360_density_unit
    assert 1 * u.flow360_area_unit == u.flow360_area_unit * 1
    assert 1 * u.flow360_area_unit != 1 * u.flow360_density_unit
    assert 1 * u.flow360_area_unit != 1

    assert (
        6 * u.flow360_area_unit
        == 1.0 * u.flow360_area_unit + 2.0 * u.flow360_area_unit + 3.0 * u.flow360_area_unit
    )
    assert -3 * u.flow360_area_unit == 1.0 * u.flow360_area_unit - 4.0 * u.flow360_area_unit
    assert -3 * u.flow360_area_unit == 1.0 * u.flow360_area_unit - 4.0 * u.flow360_area_unit
    assert -3 * u.flow360_area_unit == -1.0 * u.flow360_area_unit - 2.0 * u.flow360_area_unit

    with pytest.raises(TypeError):
        1 * u.flow360_area_unit + 2

    with pytest.raises(TypeError):
        1 * u.flow360_area_unit - 2

    with pytest.raises(TypeError):
        2 + 1 * u.flow360_area_unit

    with pytest.raises(TypeError):
        2 - 1 * u.flow360_area_unit

    with pytest.raises(ValueError):
        2 - u.flow360_area_unit

    with pytest.raises(ValueError):
        2 + u.flow360_area_unit

    with pytest.raises(TypeError):
        1 * u.flow360_area_unit + 2 * u.flow360_density_unit

    with pytest.raises(TypeError):
        1 * u.flow360_area_unit - 2 * u.flow360_density_unit

    with pytest.raises(TypeError):
        1 * u.flow360_area_unit - 2 * u.m**2

    with pytest.raises(TypeError):
        1 * u.flow360_area_unit * u.flow360_area_unit

    with pytest.raises(TypeError):
        1 * u.flow360_viscosity_unit + 1 * u.Pa * u.s

    with pytest.raises(TypeError):
        1 * u.flow360_angular_velocity_unit - 1 * u.rad / u.s

    assert (1, 1, 1) * u.flow360_area_unit
    assert u.flow360_area_unit * (1, 1, 1)
    assert (1, 1, 1) * u.flow360_mass_unit + (1, 1, 1) * u.flow360_mass_unit
    assert (1, 1, 1) * u.flow360_mass_unit - (1, 1, 1) * u.flow360_mass_unit

    with pytest.raises(TypeError):
        assert (1, 1, 1) * u.flow360_mass_unit * (1, 1, 1) * u.flow360_mass_unit

    with pytest.raises(TypeError):
        assert (1, 1, 1) * u.flow360_mass_unit * u.flow360_mass_unit

    with pytest.raises(TypeError):
        assert (1, 1, 1) * u.flow360_mass_unit + (1, 1, 1) * u.flow360_length_unit

    data = VectorDataWithUnits(
        pt=(1, 1, 1) * u.flow360_length_unit,
        vec=(1, 1, 1) * u.flow360_velocity_unit,
        ax=(1, 1, 1) * u.flow360_length_unit,
        omega=(1, 1, 1) * u.flow360_angular_velocity_unit,
    )

    with fl.flow360_unit_system:
        data_flow360 = VectorDataWithUnits(
            pt=(1, 1, 1),
            vec=(1, 1, 1),
            ax=(1, 1, 1),
            omega=(1, 1, 1),
        )
    assert data == data_flow360

    with pytest.raises(TypeError):
        data.pt + (1, 1, 1) * u.m

    with pytest.raises(TypeError):
        data.vec + (1, 1, 1) * u.m / u.s


def test_unit_system():
    # No inference outside of context
    with pytest.raises(ValueError):
        data = DataWithUnits(L=1, m=2, t=3, T=300, v=2 / 3, A=2 * 3, F=4, p=5, r=2)

    # But we can still specify units explicitly
    data = DataWithUnits(
        L=1 * u.m,
        m=2 * u.kg,
        t=3 * u.s,
        T=300 * u.K,
        v=2 / 3 * u.m / u.s,
        A=2 * 3 * u.m * u.m,
        F=4 * u.kg * u.m / u.s**2,
        p=5 * u.Pa,
        r=2 * u.kg / u.m**3,
        mu=3 * u.Pa * u.s,
        omega=5 * u.rad / u.s,
    )

    assert data.L == 1 * u.m
    assert data.m == 2 * u.kg
    assert data.t == 3 * u.s
    assert data.T == 300 * u.K
    assert data.v == 2 / 3 * u.m / u.s
    assert data.A == 6 * u.m * u.m
    assert data.F == 4 * u.kg * u.m / u.s**2
    assert data.p == 5 * u.Pa
    assert data.r == 2 * u.kg / u.m**3
    assert data.mu == 3 * u.Pa * u.s
    assert data.omega == 5 * u.rad / u.s

    # When using a unit system the units can be inferred

    # SI
    with fl.SI_unit_system:
        data = DataWithUnits(L=1, m=2, t=3, T=300, v=2 / 3, A=2 * 3, F=4, p=5, r=2, mu=3, omega=5)

        assert data.L == 1 * u.m
        assert data.m == 2 * u.kg
        assert data.t == 3 * u.s
        assert data.T == 300 * u.K
        assert data.v == 2 / 3 * u.m / u.s
        assert data.A == 6 * u.m**2
        assert data.F == 4 * u.N
        assert data.p == 5 * u.Pa
        assert data.r == 2 * u.kg / u.m**3
        assert data.mu == 3 * u.Pa * u.s
        assert data.omega == 5 * u.rad / u.s

    # CGS
    with fl.CGS_unit_system:
        data = DataWithUnits(L=1, m=2, t=3, T=300, v=2 / 3, A=2 * 3, F=4, p=5, r=2, mu=3, omega=5)

        assert data.L == 1 * u.cm
        assert data.m == 2 * u.g
        assert data.t == 3 * u.s
        assert data.T == 300 * u.K
        assert data.v == 2 / 3 * u.cm / u.s
        assert data.A == 6 * u.cm**2
        assert data.F == 4 * u.dyne
        assert data.p == 5 * u.dyne / u.cm**2
        assert data.r == 2 * u.g / u.cm**3
        assert data.mu == 3 * u.dyn * u.s / u.cm**2
        assert data.omega == 5 * u.rad / u.s

    # Imperial
    with fl.imperial_unit_system:
        data = DataWithUnits(L=1, m=2, t=3, T=300, v=2 / 3, A=2 * 3, F=4, p=5, r=2, mu=3, omega=5)

        assert data.L == 1 * u.ft
        assert data.m == 2 * u.lb
        assert data.t == 3 * u.s
        assert data.T == 300 * u.R
        assert data.v == 2 / 3 * u.ft / u.s
        assert data.A == 6 * u.ft**2
        assert data.F == 4 * u.lbf
        assert data.p == 5 * u.lbf / u.ft**2
        assert data.r == 2 * u.lb / u.ft**3
        assert data.mu == 3 * u.lbf * u.s / u.ft**2
        assert data.omega == 5 * u.rad / u.s

    # Flow360
    with fl.flow360_unit_system:
        data = DataWithUnits(L=1, m=2, t=3, T=300, v=2 / 3, A=2 * 3, F=4, p=5, r=2, mu=3, omega=5)

        assert data.L == 1 * u.flow360_length_unit
        assert data.m == 2 * u.flow360_mass_unit
        assert data.t == 3 * u.flow360_time_unit
        assert data.T == 300 * u.flow360_temperature_unit
        assert data.v == 2 / 3 * u.flow360_velocity_unit
        assert data.A == 6 * u.flow360_area_unit
        assert data.F == 4 * u.flow360_force_unit
        assert data.p == 5 * u.flow360_pressure_unit
        assert data.r == 2 * u.flow360_density_unit
        assert data.mu == 3 * u.flow360_viscosity_unit
        assert data.omega == 5 * u.flow360_angular_velocity_unit

    # Constraints
    with fl.SI_unit_system:
        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(
                L=-1, m=2, t=-3, T=300, v=2 / 3, A=2 * 3, F=-4, p=5, r=2, mu=3, omega=5
            )

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(
                L=1, m=0, t=-3, T=300, v=2 / 3, A=2 * 3, F=-4, p=5, r=2, mu=3, omega=5
            )

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(
                L=1, m=2, t=0, T=300, v=2 / 3, A=2 * 3, F=-4, p=5, r=2, mu=3, omega=5
            )

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(
                L=1, m=2, t=-3, T=-300, v=2 / 3, A=2 * 3, F=-4, p=5, r=2, mu=3, omega=5
            )

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(
                L=-1, m=2, t=-3, T=300, v=-2 / 3, A=2 * 3, F=-4, p=5, r=2, mu=3, omega=5
            )

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(
                L=-1, m=2, t=-3, T=300, v=2 / 3, A=0, F=-4, p=5, r=2, mu=3, omega=5
            )

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(
                L=1, m=2, t=-3, T=300, v=2 / 3, A=2 * 3, F=4, p=5, r=2, mu=3, omega=5
            )

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(
                L=1, m=2, t=-3, T=300, v=2 / 3, A=2 * 3, F=-4, p=9, r=2, mu=3, omega=5
            )

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(
                L=1, m=2, t=-3, T=300, v=2 / 3, A=2 * 3, F=-4, p=5, r=2, mu=3, omega=-5
            )

        data = DataWithUnitsConstrained(
            L=1, m=2, t=-3, T=300, v=2 / 3, A=2 * 3, F=-4, p=7, r=2, mu=3, omega=5
        )

        data = DataWithUnitsConstrained(
            L=1, m=2, t=-3, T=300, v=2 / 3, A=2 * 3, F=-4, p=11, r=2, mu=3, omega=5
        )

        data = DataWithUnitsConstrained(
            L=None, m=2, t=-3, T=300, v=2 / 3, A=2 * 3, F=-4, p=7, r=2, mu=3, omega=5
        )

    # Vector data
    data = VectorDataWithUnits(
        pt=(1, 1, 1) * u.m,
        vec=(1, 1, 1) * u.m / u.s,
        ax=(1, 1, 1) * u.m,
        omega=(1, 1, 1) * u.rad / u.s,
    )

    assert all(coord == 1 * u.m for coord in data.pt)
    assert all(coord == 1 * u.m / u.s for coord in data.vec)
    assert all(coord == 1 * u.m for coord in data.ax)
    assert all(coord == 1 * u.rad / u.s for coord in data.omega)

    with pytest.raises(
        ValueError,
        match=r"arg '\[1 1 1 1\] m' needs to be a collection of 3 values",
    ):
        data = VectorDataWithUnits(
            pt=(1, 1, 1, 1) * u.m,
            vec=(1, 0, 0) * u.m / u.s,
            ax=(1, 1, 1) * u.m,
            omega=(1, 1, 1) * u.rad / u.s,
        )
    with pytest.raises(ValueError):
        data = VectorDataWithUnits(
            pt=(1, 1, 1) * u.m,
            vec=(0, 0, 0) * u.m / u.s,
            ax=(1, 1, 1) * u.m,
            omega=(1, 1, 1) * u.rad / u.s,
        )

    data = VectorDataWithUnits(
        pt=(1, 1, 1) * u.m,
        vec=(1, 0, 0) * u.m / u.s,
        ax=(1, 1, 1) * u.m,
        omega=(1, 1, 1) * u.rad / u.s,
    )

    with pytest.raises(ValueError):
        data = VectorDataWithUnits(
            pt=(1, 1, 1) * u.m,
            vec=(1, 1, 1) * u.m / u.s,
            ax=(0, 0, 0) * u.m,
            omega=(1, 1, 1) * u.rad / u.s,
        )

    data = VectorDataWithUnits(
        pt=(1, 1, 1) * u.m,
        vec=(1, 1, 1) * u.m / u.s,
        ax=(1, 0, 0) * u.m,
        omega=(1, 1, 1) * u.rad / u.s,
    )

    with pytest.raises(ValueError):
        data = VectorDataWithUnits(
            pt=(1, 1, 1) * u.m,
            vec=(1, 1, 1) * u.m / u.s,
            ax=(1, 1, 1) * u.m,
            omega=(0, 1, 1) * u.rad / u.s,
        )

    data = VectorDataWithUnits(
        pt=None,
        vec=(1, 1, 1) * u.m / u.s,
        ax=(1, 0, 0) * u.m,
        omega=(1, 1, 1) * u.rad / u.s,
    )

    data = VectorDataWithUnits(
        pt=None,
        vec=(1, 1, 1) * u.N,
        ax=(1, 0, 0) * u.m,
        omega=(1, 1, 1) * u.rad / u.s,
    )

    with fl.SI_unit_system:
        # Note that for union types the first element of union that passes validation is inferred!
        data = VectorDataWithUnits(pt=(1, 1, 1), vec=(1, 1, 1), ax=(1, 1, 1), omega=(1, 1, 1))

        assert all(coord == 1 * u.m for coord in data.pt)
        assert all(coord == 1 * u.m / u.s for coord in data.vec)
        assert all(coord == 1 * u.m for coord in data.ax)
        assert all(coord == 1 * u.rad / u.s for coord in data.omega)

        data = VectorDataWithUnits(pt=None, vec=(1, 1, 1), ax=(1, 1, 1), omega=(1, 1, 1))

        assert data.pt is None
        assert all(coord == 1 * u.m / u.s for coord in data.vec)
        assert all(coord == 1 * u.m for coord in data.ax)
        assert all(coord == 1 * u.rad / u.s for coord in data.omega)


@pytest.mark.usefixtures("array_equality_override")
def test_units_serializer():
    with fl.SI_unit_system:
        data = Flow360DataWithUnits(L=2 * u.mm, pt=(2, 3, 4), lc=2)

    data_as_json = data.json(indent=4)

    with fl.CGS_unit_system:
        data_reimport = Flow360DataWithUnits(**json.loads(data_as_json))

    with fl.SI_unit_system:
        data = Flow360DataWithUnits(L=2 * u.mm, pt=(2, 3, 4), lc=2)

    data_as_json = data.json(indent=4)

    data_schema = data.schema()

    assert data_schema["properties"]["L"]["properties"]["units"]["type"] == "string"
    assert data_schema["properties"]["L"]["properties"]["value"]["type"] == "number"
    assert data_schema["properties"]["lc"]["properties"]["value"]["minimum"] == 0
    assert data_schema["properties"]["pt"]["properties"]["value"]["type"] == "array"
    assert data_schema["properties"]["pt"]["properties"]["value"]["minItems"] == 3
    assert data_schema["properties"]["pt"]["properties"]["value"]["maxItems"] == 3

    with fl.CGS_unit_system:
        data_reimport = Flow360DataWithUnits(**json.loads(data_as_json))

    assert data_reimport.L == data.L
    assert data_reimport.lc == data.lc
    assert data_reimport.pt.value.tolist() == data.pt.value.tolist()

    assert data_reimport.L == data.L
    assert data_reimport.lc == data.lc
    assert data_reimport.pt.value.tolist() == data.pt.value.tolist()

    to_file_from_file_test(data)


def test_unit_system_init():
    unit_system_dict = {
        "mass": {"value": 1.0, "units": "kg"},
        "length": {"value": 1.0, "units": "m"},
        "time": {"value": 1.0, "units": "s"},
        "temperature": {"value": 1.0, "units": "K"},
        "velocity": {"value": 1.0, "units": "m/s"},
        "area": {"value": 1.0, "units": "m**2"},
        "force": {"value": 1.0, "units": "N"},
        "pressure": {"value": 1.0, "units": "Pa"},
        "density": {"value": 1.0, "units": "kg/m**3"},
        "viscosity": {"value": 1.0, "units": "Pa*s"},
        "power": {"value": 1.0, "units": "W"},
        "moment": {"value": 1.0, "units": "N*m"},
        "angular_velocity": {"value": 1.0, "units": "rad/s"},
        "heat_flux": {"value": 1.0, "units": "kg/s**3"},
        "heat_source": {"value": 1.0, "units": "kg/s**3/m"},
        "specific_heat_capacity": {"value": 1.0, "units": "kg/s**2/m/K"},
        "thermal_conductivity": {"value": 1.0, "units": "kg/s**3*m/K"},
        "inverse_length": {"value": 1.0, "units": "m**(-1)"},
        "inverse_area": {"value": 1.0, "units": "m**(-2)"},
    }
    us = fl.UnitSystem(**unit_system_dict)
    print(us)
    print(fl.SI_unit_system)
    assert us == fl.SI_unit_system
