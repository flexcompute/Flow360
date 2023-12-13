import json

import flow360 as fl
from flow360 import units as u
from flow360.examples import OM6wing
from flow360.log import log, set_logging_level

set_logging_level("DEBUG")


with fl.SI_unit_system:
    params = fl.Flow360Params(
        geometry=fl.Geometry(
            ref_area=1,
            moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
            moment_center=(1, 2, 3) * u.flow360_length_unit,
            mesh_unit=u.mm,
        ),
        fluid_properties=fl.air,
        freestream=fl.FreestreamFromVelocity(velocity=286),
        time_stepping=fl.TimeStepping(
            max_pseudo_steps=500, CFL=fl.AdaptiveCFL(), time_step_size=1.2 * u.s
        ),
    )

    try:
        params.unit_system = fl.CGS_unit_system
        assert False
    except ValueError as err:
        # should raise ValueError error from assignment
        print(err)


with fl.CGS_unit_system:
    params = fl.Flow360Params(
        geometry=fl.Geometry(
            ref_area=1,
            moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
            moment_center=(1, 2, 3) * u.flow360_length_unit,
            mesh_unit=u.mm,
        ),
        fluid_properties=fl.air,
        freestream=fl.FreestreamFromVelocity(velocity=286),
        time_stepping=fl.TimeStepping(
            max_pseudo_steps=500, CFL=fl.AdaptiveCFL(), time_step_size=1.2 * u.s
        ),
    )


params_as_json = params.json()

with fl.UnitSystem(base_system=u.BaseSystemType.CGS, length=2.0 * u.cm):
    try:
        params_reimport = fl.Flow360Params(**json.loads(params_as_json))
        assert False
    except RuntimeError as err:
        # should raise RuntimeError error from inconsistent unit systems
        print(err)


with fl.CGS_unit_system:
    # should NOT raise RuntimeError error from inconsistent unit systems because systems are consistent
    params_reimport = fl.Flow360Params(**json.loads(params_as_json))


with fl.SI_unit_system:
    try:
        params_copy = params_reimport.copy()
        assert False
    except RuntimeError as err:
        # should raise RuntimeError error from inconsistent unit systems
        print(err)


with fl.CGS_unit_system:
    # should NOT raise RuntimeError error from inconsistent unit systems because systems are consistent
    params_copy = params_reimport.copy()


try:
    params = fl.Flow360Params(
        geometry=fl.Geometry(
            ref_area=u.m**2,
            moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
            moment_center=(1, 2, 3) * u.flow360_length_unit,
            mesh_unit=u.mm,
        ),
        freestream=fl.FreestreamFromVelocity(velocity=286 * u.m / u.s),
        time_stepping=fl.TimeStepping(
            max_pseudo_steps=500, CFL=fl.AdaptiveCFL(), time_step_size=1.2 * u.s
        ),
    )

    assert False
except RuntimeError as err:
    # should raise RuntimeError error from no context
    print(err)


try:
    with fl.CGS_unit_system:
        fl.Flow360Params(OM6wing.case_json)

    assert False
except RuntimeError as err:
    # should raise RuntimeError error from using context on file import
    print(err)


# should NOT raise RuntimeError error from NOT using context on file import
fl.Flow360Params(OM6wing.case_json)


with fl.SI_unit_system:
    try:
        params_copy.to_solver()
        assert False
    except RuntimeError as err:
        # should raise RuntimeError error from inconsistent unit systems
        print(err)


with fl.CGS_unit_system:
    # should NOT raise RuntimeError error from inconsistent unit systems because systems are consistent
    params_copy.to_solver()


# should NOT raise RuntimeError error from inconsistent unit systems because systems NO system
params_copy.to_solver()


with fl.SI_unit_system:
    try:
        params_copy.to_flow360_json()
        assert False
    except RuntimeError as err:
        # should raise RuntimeError error from inconsistent unit systems
        print(err)


with fl.CGS_unit_system:
    # should NOT raise RuntimeError error from inconsistent unit systems because systems are consistent
    params_copy.to_flow360_json()


# should NOT raise RuntimeError error from inconsistent unit systems because systems NO system
params_copy.to_flow360_json()
