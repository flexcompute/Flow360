import flow360 as fl
from flow360.examples import OM6wing
import pydantic as pd
import unyt
import unyt as u

from flow360.component.flow360_params.unit_system import LengthType, MassType, TimeType, TemperatureType, VelocityType, AreaType, ForceType, PressureType, DensityType



# TODO:
# 1. infer non-primitive units (eg pressure) inside UnitSystem context
# 2. add easy validations for PositveFloat, NonNegative etc
# 3. Improve unit system definition
# 4. dump / retrive pydantic model (Flow360BaseModel) to file and from file with units
# 5. replace all dimensional numerical types in Flow360Params
# 6. introduce flow360 units definitions eg: ref_area = 1 flow360_length_unit**2 where definitions of flow360 units can be found in docs


class DataWithUnits(pd.BaseModel):
    l: LengthType = pd.Field()
    m: MassType = pd.Field()
    t: TimeType = pd.Field()
    T: TemperatureType = pd.Field()
    v: VelocityType = pd.Field()
    A: AreaType = pd.Field()
    F: ForceType = pd.Field()
    p: PressureType = pd.Field()
    r: DensityType = pd.Field()



data = DataWithUnits(
    l = 1 * u.m,
    m = 2 * u.kg,
    t = 3 * u.s,
    T = 300 * u.K,
    v = 2/3 * u.m / u.s,
    A = 2 * 3 * u.m * u.m,
    F = 4 * u.kg * u.m / u.s**2,
    p = 5 * u.Pa,
    r = 2 * u.kg / u.m**3
)

for n, v in data:
    print(f"{n}={v}")

print(fl.SI_unit_system.length)
print(fl.SI_unit_system['length'])


with fl.SI_unit_system:
    print(fl.SI_unit_system.length)
    print(fl.SI_unit_system['length'])
    data_with_context = DataWithUnits(
        l = 1,
        m = 2,
        t = 3,
        T = 300,
        v = 2/3 * u.m / u.s, # how to infer these types inside validator
        A = 2 * 3 * u.m * u.m,
        F = 4 * u.kg * u.m / u.s**2,
        p = 5 * u.Pa,
        r = 2 * u.kg / u.m**3
    )

    for n, v in data_with_context:
        print(f"{n}={v}")




velocity = unyt.Unit()
print(velocity)
print(velocity.dimensions)
velocity = 2 * unyt.m / ( 3 * unyt.s)

print(velocity, type(velocity))
# >>> 0.6666666666666666 m/s

print(velocity.units.dimensions)


my_length = 1 * unyt.m + 2 * unyt.ft
print(my_length)
# >>> 1.6096 m

my_dimesionless_value = my_length / (1.0 * unyt.cm)

print(my_dimesionless_value)
# >>> 100.0 dimensionless

print(my_dimesionless_value.units.is_dimensionless)
# >>> True

# (1.0*unyt.mile).to('lb')  
# >>> unyt.exceptions.UnitConversionError: Cannot convert between 'mile' (dim '(length)') and 'lb' (dim '(mass)').


velocity = 2 * unyt.m / ( 3 * unyt.s) + 300 * unyt.K
# >> unyt.exceptions.UnitOperationError: The <ufunc 'add'> operator for unyt_arrays with units "m" (dimensions "(length)") and "s" (dimensions "(time)") is not well defined.




# fluid_properties=Air(), in future Water(), FluidProperties(pressure + temperature, density + temperature), FluidPropertiesPressureTemp(), FluidPropertiesDensityTemp()



