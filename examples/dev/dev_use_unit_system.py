from pprint import pprint
from threading import Thread
from time import sleep

import numpy as np
import pydantic.v1 as pd
import unyt

import flow360.component.v1xxx as fl
from flow360.component.v1xxx import Geometry
from flow360.component.v1 import units as u
from flow360.component.v1.unit_system import (
    AngularVelocityType,
    AreaType,
    BaseSystemType,
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


class DataWithUnits(pd.BaseModel):
    L: LengthType = pd.Field()
    m: MassType.Constrained(ge=2, allow_inf_nan=True) = pd.Field()
    t: TimeType.Positive = pd.Field()
    T: TemperatureType = pd.Field()
    v: VelocityType = pd.Field()
    A: AreaType = pd.Field()
    F: ForceType = pd.Field()
    p: PressureType = pd.Field()
    r: DensityType = pd.Field()
    mu: ViscosityType = pd.Field()
    omega: AngularVelocityType = pd.Field()


class VectorDataWithUnits(pd.BaseModel):
    pt: LengthType.Point = pd.Field()
    vec: VelocityType.Direction = pd.Field()
    ax: LengthType.Axis = pd.Field()
    m: MassType.Moment = pd.Field()


vector_data1 = VectorDataWithUnits(
    pt=(1, 1, 1) * u.m, vec=(1, 1, 1) * u.m / u.s, ax=(1, 1, 1) * u.m, m=(1, 1, 1) * u.kg
)

vector_data2 = VectorDataWithUnits(
    pt=(2, 1, 1) * u.m, vec=(1, 2, 1) * u.m / u.s, ax=(1, 1, 2) * u.m, m=(2, 1, 1) * u.kg
)

result = vector_data1.pt + vector_data2.pt

data = DataWithUnits(
    l=2 * u.m + 1 * u.cm,
    m=np.inf * u.kg,
    t=3 * u.s,
    T=-300 * u.K,
    v=2 / 3 * u.m / u.s,
    A=2 * 3 * u.m * u.m,
    F=4 * u.kg * u.m / u.s**2,
    p=5 * u.Pa,
    r=2 * u.kg / u.m**3,
    mu=2 * u.Pa * u.s,
    omega=5 * u.rad / u.s,
)

schema = data.schema()

pprint(schema)

for n, v in data:
    print(f"{n}={v}")

# Types in unit system context can be specified explicitly or inferred
with fl.SI_unit_system:
    data_with_context = DataWithUnits(
        l=1 * u.inch,
        m=2,
        t=3,
        T=300,
        v=2 / 3 * u.m / u.s,
        A=2 * 3,
        F=4 * u.kg * u.m / u.s**2,
        p=5,
        r=2 * u.kg / u.m**3,
        mu=2,
        omega=5 * u.rad / u.s,
    )

    vector_data1 = VectorDataWithUnits(pt=(1, 1, 1), vec=(1, 1, 1), ax=(1, 1, 1), m=(1, 1, 1))

    # Example of serializing and deserializing dimensioned fields
    d = data_with_context.dict()

    val = str(d["m"].value)
    unt = str(d["m"].units)

    q = unyt.unyt_quantity(float(val), unt)

    assert d["m"] == q

    reconstructed_data = DataWithUnits(**d)

    assert reconstructed_data == data_with_context

    for n, v in data_with_context:
        print(f"{n}={v}")

with fl.flow360_unit_system:
    data_with_context = DataWithUnits(
        l=1, m=2, t=3, T=300, v=2 / 3, A=2 * 3, F=4, p=5, r=2, mu=2, omega=5
    )

    for n, v in data_with_context:
        print(f"{n}={v}")

    data = VectorDataWithUnits(pt=(1, 1, 1), vec=(1, 1, 1), ax=(1, 1, 1), m=(1, 1, 1))

    schema = data.schema()

    pprint(schema)

    value = data.vec - (1, 1, 1) * u.flow360_velocity_unit

    data = DataWithUnits(l=1, m=2, t=3, T=300, v=2 / 3, A=2 * 3, F=4, p=5, r=2, mu=3, omega=5)

    value = data.m - 4 * u.flow360_mass_unit

with u.UnitSystem(base_system=BaseSystemType.SI, length=u.flow360_length_unit):
    data_with_context = DataWithUnits(
        l=1, m=2, t=u.flow360_time_unit * 3, T=300, v=2 / 3, A=2 * 3, F=4, p=5, r=2, mu=2, omega=5
    )

    for n, v in data_with_context:
        print(f"{n}={v}")

threaded_data = DataWithUnits(
    l=1 * u.m,
    m=2 * u.kg,
    t=3 * u.s,
    T=300 * u.K,
    v=2 / 3 * u.m / u.s,
    A=2 * 3 * u.m**2,
    F=4 * u.N,
    p=5 * u.Pa,
    r=2 * u.kg / u.m**3,
    mu=2 * u.Pa * u.s,
    omega=5 * u.rad / u.s,
)


def thread1():
    with fl.SI_unit_system:
        sleep(1)

        t1_data = DataWithUnits(
            l=1, m=2, t=3, T=300, v=2 / 3, A=2 * 3, F=4, p=5, r=2, mu=2, omega=5
        )

        # Should be in meters
        print(f"Length in thread 1: {t1_data.l}")

        threaded_data.m += 2 * u.kg


def thread2():
    with fl.CGS_unit_system:
        t2_data = DataWithUnits(
            l=1, m=2, t=3, T=300, v=2 / 3, A=2 * 3, F=4, p=5, r=2, mu=2, omega=5
        )

        # Should be in centimeters
        print(f"Length in thread 2: {t2_data.l}")

        threaded_data.m *= 2


t1 = Thread(target=thread1)
t2 = Thread(target=thread2)

# start the threads
t1.start()
t2.start()

# wait for the threads to complete
t1.join()
t2.join()

# Without locks thread2 executes before thread 1
print(f"After running both threads: {threaded_data.m}")

schema = fl.SI_unit_system.schema()

print(schema)

data = {
    "refArea": {"units": "flow360_area_unit"},
    "momentCenter": {"value": [1, 2, 3], "units": "flow360_length_unit"},
    "momentLength": {"value": [1.47602, 0.801672958512342, 1.47602], "units": "inch"},
    "meshUnit": {"value": 1.0, "units": "mm"},
}

model = Geometry(**data)
