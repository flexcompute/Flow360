import json
from copy import deepcopy
from typing import Optional, Union

import pydantic as pd
import pytest
from numpy import nan

from flow360.component.simulation import units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.updater_utils import compare_dicts
from flow360.component.simulation.unit_system import (
    AbsoluteTemperatureType,
    AngleType,
    AngularVelocityType,
    AreaType,
    DensityType,
    ForceType,
    FrequencyType,
    KinematicViscosityType,
    LengthType,
    MassFlowRateType,
    MassType,
    PressureType,
    SpecificEnergyType,
    TimeType,
    VelocityType,
    ViscosityType,
)


class DataWithUnits(pd.BaseModel):
    L: LengthType = pd.Field()
    a: AngleType = pd.Field()
    m: MassType = pd.Field()
    t: TimeType = pd.Field()
    T: AbsoluteTemperatureType = pd.Field()
    v: VelocityType = pd.Field()
    A: AreaType = pd.Field()
    F: ForceType = pd.Field()
    p: PressureType = pd.Field()
    r: DensityType = pd.Field()
    mu: ViscosityType = pd.Field()
    nu: KinematicViscosityType = pd.Field()
    m_dot: MassFlowRateType = pd.Field()
    v_sq: SpecificEnergyType = pd.Field()
    fqc: FrequencyType = pd.Field()
    omega: AngularVelocityType = pd.Field()


class DataWithOptionalUnion(pd.BaseModel):
    L: LengthType = pd.Field()
    m: Optional[MassType] = pd.Field(None)
    t: Union[TimeType, AbsoluteTemperatureType] = pd.Field()
    v: Optional[Union[TimeType, AbsoluteTemperatureType]] = pd.Field(None)


class DataWithUnitsConstrained(pd.BaseModel):
    L: Optional[LengthType.NonNegative] = pd.Field(None)
    a: AngleType.NonNegative = pd.Field()
    m: MassType.Positive = pd.Field()
    t: TimeType.Negative = pd.Field()
    T: AbsoluteTemperatureType.NonNegative = pd.Field()
    v: VelocityType.NonNegative = pd.Field()
    A: AreaType.Positive = pd.Field()
    F: ForceType.NonPositive = pd.Field()
    p: Union[PressureType.Constrained(ge=5, lt=9), PressureType.Constrained(ge=10, lt=12)] = (
        pd.Field()
    )
    r: DensityType = pd.Field()
    mu: ViscosityType.Constrained(ge=2) = pd.Field()
    nu: KinematicViscosityType.Constrained(ge=2) = pd.Field()
    m_dot: MassFlowRateType.Constrained(ge=3) = pd.Field()
    v_sq: SpecificEnergyType.Constrained(le=2) = pd.Field()
    fqc: FrequencyType.Constrained(gt=22) = pd.Field()
    omega: AngularVelocityType.NonNegative = pd.Field()


class MatrixDataWithUnits(pd.BaseModel):
    locations: LengthType.CoordinateGroup = pd.Field()
    locationsT: LengthType.CoordinateGroupTranspose = pd.Field()


class VectorDataWithUnits(pd.BaseModel):
    pt: Optional[LengthType.Point] = pd.Field(None)
    vec: Union[VelocityType.Direction, ForceType.Point] = pd.Field()
    ax: LengthType.Axis = pd.Field()
    omega: AngularVelocityType.Moment = pd.Field()
    lp: LengthType.PositiveVector = pd.Field()


class ArrayDataWithUnits(Flow360BaseModel):
    l_arr: AngleType.Array = pd.Field()
    l_arr_nonneg: LengthType.NonNegativeArray = pd.Field()


class Flow360DataWithUnits(Flow360BaseModel):
    l: LengthType = pd.Field()
    lp: LengthType.Point = pd.Field()
    lc: LengthType.NonNegative = pd.Field()


class ScalarOrVector(Flow360BaseModel):
    l: Union[LengthType, LengthType.Point] = pd.Field()


def test_unit_access():
    assert u.CGS_unit_system
    assert u.inch


def test_unit_systems_compare():
    assert u.SI_unit_system != u.CGS_unit_system

    assert u.SI_unit_system == u.SI_unit_system

    assert u.SI_unit_system == u.UnitSystem(base_system="SI")


# test_flow360_unit_arithmetic: REMOVED — tested _Flow360BaseUnit arithmetic (deleted in Phase 4)
# Tracked in plans/removed_tests.markdown for future migration to schema side.


def _assert_exact_same_unyt(input, ref):
    assert input.value == ref.value and str(input.units.expr) == str(ref.units.expr)


def test_unit_system():
    # No inference outside of context
    with pytest.raises(pd.ValidationError):
        data = DataWithUnits(L=1, a=2, m=2, t=3, T=300, v=2 / 3, A=2 * 3, F=4, p=5, r=2)

    # But we can still specify units explicitly
    data = DataWithUnits(
        L=1 * u.m,
        a=1 * u.degree,
        m=2 * u.kg,
        t=3 * u.s,
        T=300 * u.K,
        v=2 / 3 * u.m / u.s,
        A=2 * 3 * u.m * u.m,
        F=4 * u.kg * u.m / u.s**2,
        p=5 * u.Pa,
        r=2 * u.kg / u.m**3,
        mu=3 * u.Pa * u.s,
        nu=4 * u.m**2 / u.s,
        omega=5 * u.rad / u.s,
        m_dot=12 * u.kg / u.s,
        v_sq=4 * u.m**2 / u.s**2,
        fqc=1234 / u.s,
    )

    _assert_exact_same_unyt(data.L, 1 * u.m)
    _assert_exact_same_unyt(data.m, 2 * u.kg)
    _assert_exact_same_unyt(data.t, 3 * u.s)
    _assert_exact_same_unyt(data.T, 300 * u.K)
    _assert_exact_same_unyt(data.v, 2 / 3 * u.m / u.s)
    _assert_exact_same_unyt(data.A, 6 * u.m * u.m)
    _assert_exact_same_unyt(data.F, 4 * u.kg * u.m / u.s**2)
    _assert_exact_same_unyt(data.p, 5 * u.Pa)
    _assert_exact_same_unyt(data.r, 2 * u.kg / u.m**3)
    _assert_exact_same_unyt(data.mu, 3 * u.Pa * u.s)
    _assert_exact_same_unyt(data.omega, 5 * u.rad / u.s)
    _assert_exact_same_unyt(data.m_dot, 12 * u.kg / u.s)
    _assert_exact_same_unyt(data.v_sq, 4 * u.m**2 / u.s**2)
    _assert_exact_same_unyt(data.fqc, 1234 / u.s)

    # When using a unit system the units can be inferred

    input = {
        "L": 1,
        "m": 2,
        "t": 3,
        "T": 300,
        "v": 2 / 3,
        "A": 2 * 3,
        "F": 4,
        "p": 5,
        "r": 2,
        "mu": 3,
        "nu": 4,
        "m_dot": 11,
        "v_sq": 123,
        "fqc": 1111,
    }
    # SI
    with u.SI_unit_system:
        data = DataWithUnits(**input, a=1 * u.degree, omega=1 * u.radian / u.s)

        _assert_exact_same_unyt(data.L, 1 * u.m)
        _assert_exact_same_unyt(data.m, 2 * u.kg)
        _assert_exact_same_unyt(data.t, 3 * u.s)
        _assert_exact_same_unyt(data.T, 300 * u.K)
        _assert_exact_same_unyt(data.v, 2 / 3 * u.m / u.s)
        _assert_exact_same_unyt(data.A, 6 * u.m**2)
        _assert_exact_same_unyt(data.F, 4 * u.N)
        _assert_exact_same_unyt(data.p, 5 * u.Pa)
        _assert_exact_same_unyt(data.r, 2 * u.kg / u.m**3)
        _assert_exact_same_unyt(data.mu, 3 * u.kg / (u.m * u.s))
        _assert_exact_same_unyt(data.nu, 4 * u.m**2 / u.s)
        _assert_exact_same_unyt(data.m_dot, 11 * u.kg / u.s)
        _assert_exact_same_unyt(data.v_sq, 123 * u.J / u.kg)
        _assert_exact_same_unyt(data.fqc, 1111 * u.Hz)

    # CGS
    with u.CGS_unit_system:
        data = DataWithUnits(**input, a=1 * u.degree, omega=1 * u.radian / u.s)

        _assert_exact_same_unyt(data.L, 1 * u.cm)
        _assert_exact_same_unyt(data.m, 2 * u.g)
        _assert_exact_same_unyt(data.t, 3 * u.s)
        _assert_exact_same_unyt(data.T, 300 * u.K)
        _assert_exact_same_unyt(data.v, 2 / 3 * u.cm / u.s)
        _assert_exact_same_unyt(data.A, 6 * u.cm**2)
        _assert_exact_same_unyt(data.F, 4 * u.dyne)
        _assert_exact_same_unyt(data.p, 5 * u.dyne / u.cm**2)
        _assert_exact_same_unyt(data.r, 2 * u.g / u.cm**3)
        _assert_exact_same_unyt(data.mu, 3 * u.g / u.s / u.cm)
        _assert_exact_same_unyt(data.nu, 4 * u.cm**2 / u.s)
        _assert_exact_same_unyt(data.m_dot, 11 * u.g / u.s)
        _assert_exact_same_unyt(data.v_sq, 123 * u.erg / u.g)
        _assert_exact_same_unyt(data.fqc, 1111 / u.s)

    # Imperial
    with u.imperial_unit_system:
        data = DataWithUnits(**input, a=1 * u.degree, omega=1 * u.radian / u.s)
        _assert_exact_same_unyt(data.L, 1 * u.ft)
        _assert_exact_same_unyt(data.m, 2 * u.lb)
        _assert_exact_same_unyt(data.t, 3 * u.s)
        _assert_exact_same_unyt(data.T, 300 * u.degF)
        _assert_exact_same_unyt(data.v, 2 / 3 * u.ft / u.s)
        _assert_exact_same_unyt(data.A, 6 * u.ft**2)
        _assert_exact_same_unyt(data.F, 4 * u.lbf)
        _assert_exact_same_unyt(data.p, 5 * u.lbf / u.ft**2)
        _assert_exact_same_unyt(data.r, 2 * u.lb / u.ft**3)
        _assert_exact_same_unyt(data.mu, 3 * u.lb / (u.ft * u.s))
        _assert_exact_same_unyt(data.nu, 4 * u.ft**2 / u.s)
        _assert_exact_same_unyt(data.m_dot, 11 * u.lb / u.s)
        _assert_exact_same_unyt(data.v_sq, 123 * u.ft**2 / u.s**2)
        _assert_exact_same_unyt(data.fqc, 1111 / u.s)

    # Flow360 section removed — tested _Flow360BaseUnit inference (deleted in Phase 4)
    # Tracked in plans/removed_tests.markdown

    correct_input = {
        "L": 1,
        "a": 1 * u.degree,
        "m": 2,
        "t": -3,
        "T": 300,
        "v": 2 / 3,
        "A": 2 * 3,
        "F": -4,
        "p": 5,
        "r": 2,
        "mu": 3,
        "nu": 4,
        "omega": 1 * u.radian / u.s,
        "m_dot": 10,
        "v_sq": 0.2,
        "fqc": 123,
    }
    # Constraints
    with u.SI_unit_system:
        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "L": -1})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "m": 0})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "t": 0})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "T": -300})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "v": -2 / 3})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "A": 0})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "F": 4})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "p": 9})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "p": 13})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "mu": 1.9})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "nu": 1.9})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "m_dot": 1})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "v_sq": 12})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "fqc": 12})

        with pytest.raises(ValueError):
            data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "omega": -12})

        data = DataWithUnitsConstrained(**correct_input)

        data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "p": 11})

        data = DataWithUnitsConstrained(**{**deepcopy(correct_input), "L": None})

    # Vector data
    data = VectorDataWithUnits(
        pt=(1, 1, 1) * u.m,
        vec=(1, 1, 1) * u.m / u.s,
        ax=(1, 1, 1) * u.m,
        omega=(1, 1, 1) * u.rad / u.s,
        lp=(1, 1, 1) * u.m,
    )

    assert all(coord == 1 * u.m for coord in data.pt)
    assert all(coord == 1 * u.m / u.s for coord in data.vec)
    assert all(coord == 1 * u.m for coord in data.ax)
    assert all(coord == 1 * u.rad / u.s for coord in data.omega)
    assert all(coord == 1 * u.m for coord in data.lp)

    with pytest.raises(
        ValueError,
        match=r"arg '\[1 1 1 1\] m' needs to be a collection of 3 values",
    ):
        data = VectorDataWithUnits(
            pt=(1, 1, 1, 1) * u.m,
            vec=(1, 0, 0) * u.m / u.s,
            ax=(1, 1, 1) * u.m,
            omega=(1, 1, 1) * u.rad / u.s,
            lp=(1, 1, 1) * u.m,
        )
    with pytest.raises(
        ValueError,
        match=r"arg '\[0 0 0\] m/s' cannot have zero norm",
    ):
        data = VectorDataWithUnits(
            pt=(1, 1, 1) * u.m,
            vec=(0, 0, 0) * u.m / u.s,
            ax=(1, 1, 1) * u.m,
            omega=(1, 1, 1) * u.rad / u.s,
            lp=(1, 1, 1) * u.m,
        )

    data = VectorDataWithUnits(
        pt=(1, 1, 1) * u.m,
        vec=(1, 0, 0) * u.m / u.s,
        ax=(1, 1, 1) * u.m,
        omega=(1, 1, 1) * u.rad / u.s,
        lp=(1, 1, 1) * u.m,
    )

    with pytest.raises(ValueError, match=r"arg '\[0 0 0\] m' cannot have zero norm"):
        data = VectorDataWithUnits(
            pt=(1, 1, 1) * u.m,
            vec=(1, 1, 1) * u.m / u.s,
            ax=(0, 0, 0) * u.m,
            omega=(1, 1, 1) * u.rad / u.s,
            lp=(1, 1, 1) * u.m,
        )

    data = VectorDataWithUnits(
        pt=(1, 1, 1) * u.m,
        vec=(1, 1, 1) * u.m / u.s,
        ax=(1, 0, 0) * u.m,
        omega=(1, 1, 1) * u.rad / u.s,
        lp=(1, 1, 1) * u.m,
    )

    with pytest.raises(ValueError, match=r"arg '\[0 1 1\] rad/s' cannot have zero component"):
        data = VectorDataWithUnits(
            pt=(1, 1, 1) * u.m,
            vec=(1, 1, 1) * u.m / u.s,
            ax=(1, 1, 1) * u.m,
            omega=(0, 1, 1) * u.rad / u.s,
            lp=(1, 1, 1) * u.m,
        )

    data = VectorDataWithUnits(
        pt=(1, 1, 1) * u.m,
        vec=(1, 1, 1) * u.m / u.s,
        ax=(1, 0, 0) * u.m,
        omega=(1, 1, 1) * u.rad / u.s,
        lp=(1, 1, 1) * u.m,
    )

    with pytest.raises(ValueError, match=r"arg '\[1 1 0\] m' cannot have zero component"):
        data = VectorDataWithUnits(
            pt=(1, 1, 1) * u.m,
            vec=(1, 1, 1) * u.m / u.s,
            ax=(1, 0, 0) * u.m,
            omega=(1, 1, 1) * u.rad / u.s,
            lp=(1, 1, 0) * u.m,
        )

    data = VectorDataWithUnits(
        pt=None,
        vec=(1, 1, 1) * u.m / u.s,
        ax=(1, 0, 0) * u.m,
        omega=(1, 1, 1) * u.rad / u.s,
        lp=(1, 1, 1) * u.m,
    )

    data = VectorDataWithUnits(
        pt=None,
        vec=(1, 1, 1) * u.N,
        ax=(1, 0, 0) * u.m,
        omega=(1, 1, 1) * u.rad / u.s,
        lp=(1, 1, 1) * u.m,
    )

    data = VectorDataWithUnits(
        pt=None,
        vec={"value": [1, 1, 1], "units": "N"},
        ax={"value": [0, 0, 1], "units": "m"},
        omega={"value": [1, 1, 1], "units": "rad/s"},
        lp={"value": [1, 1, 1], "units": "m"},
    )

    # Nested dict with wrong inner shape — rejected as not 3 values
    with pytest.raises(
        pd.ValidationError,
        match=r"needs to be a collection of 3 values",
    ):
        data = VectorDataWithUnits(
            pt=None,
            vec={"value": {"value": [1, 2], "units": "wrong"}, "units": "N"},
            ax={"value": [0, 0, 1], "units": "m"},
            omega={"value": [1, 1, 1], "units": "rad/s"},
            lp={"value": [1, 1, 1], "units": "m"},
        )

    # Invalid unit name — rejected as dimension mismatch
    with pytest.raises(
        pd.ValidationError,
        match=r"does not match \(length\) dimension",
    ):
        Flow360DataWithUnits(l={"value": 1.0, "units": "bogus_unit"}, lp=[1, 2, 3] * u.m, lc=u.m)

    with pytest.raises(
        pd.ValidationError,
        match=r"NaN/Inf/None found in input array. Please ensure your input is complete.",
    ):
        data = VectorDataWithUnits(
            pt=None,
            vec={"value": [1, 1, None], "units": "N"},
            ax={"value": [0, 0, 1], "units": "m"},
            omega={"value": [1, 1, 1], "units": "rad/s"},
            lp={"value": [1, 1, 1], "units": "m"},
        )

    with u.SI_unit_system:
        # Note that for union types the first element of union that passes validation is inferred!
        data = VectorDataWithUnits(
            pt=(1, 1, 1), vec=(1, 1, 1), ax=(1, 1, 1), omega=(1, 1, 1) * u.rpm, lp=(1, 1, 1)
        )

        assert all(coord == 1 * u.m for coord in data.pt)
        assert all(coord == 1 * u.m / u.s for coord in data.vec)
        assert all(coord == 1 * u.m for coord in data.ax)
        assert all(coord == 1 * u.rpm for coord in data.omega)
        assert all(coord == 1 * u.m for coord in data.lp)

        data = VectorDataWithUnits(
            pt=None, vec=(1, 1, 1), ax=(1, 1, 1), omega=(1, 1, 1) * u.rpm, lp=(1, 1, 1)
        )

        assert data.pt is None
        assert all(coord == 1 * u.m / u.s for coord in data.vec)
        assert all(coord == 1 * u.m for coord in data.ax)
        assert all(coord == 1 * u.rpm for coord in data.omega)
        assert all(coord == 1 * u.m for coord in data.lp)

    # Array data
    data = ArrayDataWithUnits(
        l_arr=[-1, -1, -1, -1] * u.rad,
        l_arr_nonneg=[0, 0, 0, 0] * u.m,
    )

    assert all(coord == -1 * u.rad for coord in data.l_arr)
    assert all(coord == 0 * u.m for coord in data.l_arr_nonneg)

    with pytest.raises(ValueError, match=r"arg '\[ 0  0  0 -1\] m' cannot have negative value"):
        data = ArrayDataWithUnits(
            l_arr=[-1, -1, -1, -1] * u.rad,
            l_arr_nonneg=[0, 0, 0, -1] * u.m,
        )

    with pytest.raises(
        pd.ValidationError,
        match=r"NaN/Inf/None found in input array. Please ensure your input is complete.",
    ):
        data = ArrayDataWithUnits(
            l_arr={"value": [-1, -1, -1, None], "units": "rad"},
            l_arr_nonneg={"value": [1, 1, 1, 1], "units": "m"},
        )

    with u.SI_unit_system:
        data = ArrayDataWithUnits(
            l_arr=[-1, -1, -1, -1] * u.rad,
            l_arr_nonneg=[1, 1, 1, 1],
        )

        assert all(coord == -1 * u.rad for coord in data.l_arr)
        assert all(coord == 1 * u.m for coord in data.l_arr_nonneg)

    # Matrix data
    data = MatrixDataWithUnits(
        locations=[[-1, -1, -1], [-1, -1, -1]] * u.inch, locationsT=[[1, 1], [1, 1], [1, 1]] * u.m
    )

    assert all(all(value == -1 * u.inch for value in coord) for coord in data.locations)
    assert all(all(value == 1 * u.m for value in coord) for coord in data.locationsT)

    with pytest.raises(
        ValueError, match=r"arg '\[-1 -1 -1\] m' needs to be a 2-dimensional collection of values."
    ):
        data = MatrixDataWithUnits(
            locations=[-1, -1, -1] * u.m,
            locationsT=[[1, 1], [1, 1], [1, 1]] * u.m,
        )

    with pytest.raises(
        ValueError,
        match=r"setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions.",
    ):
        data = MatrixDataWithUnits(
            locations=[[-1, -1, -1], [-1, -1, -1, -2]] * u.m,
            locationsT=[[1, 1], [1, 1], [1, 1]] * u.m,
        )

    with pytest.raises(
        ValueError,
        match=r"arg '\[\[-1 -1\]\n \[-1 -1\]\] m' needs to be a 2-dimensional collection of values with the 2nd dimension as 3.",
    ):
        data = MatrixDataWithUnits(
            locations=[[-1, -1], [-1, -1]] * u.m, locationsT=[[1, 1], [1, 1], [1, 1]] * u.m
        )

    with pytest.raises(
        ValueError,
        match=r"arg '\[\[1 1\]\n \[1 1\]\n \[1 1\]\n \[1 1\]\] m' needs to be a 2-dimensional collection of values with the 1st dimension as 3.",
    ):
        data = MatrixDataWithUnits(
            locations=[[-1, -1, -1], [-1, -1, -1]] * u.m,
            locationsT=[[1, 1], [1, 1], [1, 1], [1, 1]] * u.m,
        )

    with u.SI_unit_system:
        data = MatrixDataWithUnits(
            locations=[[-1, -1, -1], [-1, -1, -1]], locationsT=[[1, 1], [1, 1], [1, 1]]
        )

        assert all(all(value == -1 * u.m for value in coord) for coord in data.locations)
        assert all(all(value == 1 * u.m for value in coord) for coord in data.locationsT)


def test_optionals_and_unions():

    data = DataWithOptionalUnion(
        L=1 * u.m,
        m=2 * u.kg,
        t=3 * u.s,
        v=300 * u.K,
    )

    data = DataWithOptionalUnion(
        L=1 * u.m,
        t=3 * u.s,
    )

    data = DataWithOptionalUnion(
        L=1 * u.m,
        t=3 * u.K,
    )

    data = DataWithOptionalUnion(
        L=1 * u.m,
        t=3 * u.s,
        v=300 * u.s,
    )


def test_scalar_or_vector():

    expected_s = {"value": 1, "units": "m"}
    expected_v = {"value": (1, 1, 1), "units": "m"}

    with u.SI_unit_system:
        m = ScalarOrVector(l=1)
        assert compare_dicts(m.model_dump()["l"], expected_s)

    with u.SI_unit_system:
        m = ScalarOrVector(l=(1, 1, 1))
        assert compare_dicts(m.model_dump()["l"], expected_v)

    m = ScalarOrVector(l=1 * u.m)
    assert compare_dicts(m.model_dump()["l"], expected_s)

    m = ScalarOrVector(l=(1, 1, 1) * u.m)
    assert compare_dicts(m.model_dump()["l"], expected_v)

    m = ScalarOrVector(l={"value": 1, "units": "m"})
    assert compare_dicts(m.model_dump()["l"], expected_s)

    m = ScalarOrVector(l={"value": (1, 1, 1), "units": "m"})
    assert compare_dicts(m.model_dump()["l"], expected_v)


@pytest.mark.usefixtures("array_equality_override")
def test_units_serializer():
    with u.SI_unit_system:
        data = Flow360DataWithUnits(l=2 * u.mm, lp=[1, 2, 3], lc=u.mm)

    data_as_json = data.model_dump_json(indent=2)

    with u.CGS_unit_system:
        data_reimport = Flow360DataWithUnits(**json.loads(data_as_json))

    assert data_reimport == data

    #####
    with u.SI_unit_system:
        data = ArrayDataWithUnits(
            l_arr=[0] * u.rad,  # Single element array should come back as an array.
            l_arr_nonneg=[1, 1, 1, 1],
        )

    data_as_json = data.model_dump_json()


def test_units_schema():
    schema = Flow360DataWithUnits.model_json_schema()

    assert schema


def test_unit_system_init():
    # NEW UnitSystem only accepts base_system or 5 base dimensions
    us = u.UnitSystem(base_system="SI")
    assert us == u.SI_unit_system


def test_custom_unit_string_deserialization():
    assert u.unyt.unyt_quantity(1, "degC") == 1 * u.degC
    assert u.unyt.unyt_quantity(2, "degF") == 2 * u.degF


def test_below_absolute_zero_temperature():
    with pytest.raises(
        pd.ValidationError,
        match=r"The specified temperature -333.0 K is below absolute zero. Please input a physical temperature value.",
    ):

        class tester(pd.BaseModel):
            temp: AbsoluteTemperatureType = pd.Field()

        tester(temp=-333 * u.K)
