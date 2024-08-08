import json
from pprint import pprint

import flow360 as fl
from flow360 import log
from flow360 import units as u
from flow360.component.flow360_params.services import validate_model

log.set_logging_level("DEBUG")

with fl.SI_unit_system:
    params = fl.Flow360Params(
        geometry=fl.Geometry(
            ref_area=1.0 * u.flow360_area_unit,
            moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
            moment_center=(1, 2, 3) * u.flow360_length_unit,
            mesh_unit=u.mm,
        ),
        freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
        time_stepping=fl.UnsteadyTimeStepping(
            max_pseudo_steps=500, CFL=fl.AdaptiveCFL(), time_step_size=1.2 * u.s
        ),
        boundaries={
            "1": fl.NoSlipWall(name="wing", velocity=(1, 2, 3) * u.km / u.hr),
            "2": fl.SlipWall(name="symmetry"),
            "3": fl.FreestreamBoundary(name="freestream"),
        },
        fluid_properties=fl.air,
        volume_zones={
            "zone1": fl.FluidDynamicsVolumeZone(
                reference_frame=fl.ReferenceFrame(
                    center=(0, 0, 0), axis=(1, 0, 0), omega=10 * u.rpm
                )
            ),
            "zone2": fl.FluidDynamicsVolumeZone(
                reference_frame=fl.ReferenceFrame(
                    center=(0, 0, 0), axis=(1, 0, 0), omega=10 * 2 * fl.pi / 60
                )
            ),
            "zone3": fl.FluidDynamicsVolumeZone(
                reference_frame=fl.ReferenceFrame(
                    center=(0, 0, 0), axis=(1, 0, 0), omega=10 * 360 / 60 * u.deg / u.s
                )
            ),
        },
    )

log.set_logging_level("INFO")

params_copy = params.copy()

params_as_dict = params_copy.dict()

del params_as_dict["fluid_properties"]

errors, warnings = validate_model(params_as_dict, "SI")
pprint(errors)

params_as_json = params.json(indent=4)
print(params_as_json)

with fl.UnitSystem(base_system=u.BaseSystemType.SI):
    params_reimport = fl.Flow360Params(**json.loads(params_as_json))
    assert params_reimport.geometry.ref_area == params.geometry.ref_area

params_solver = params.to_solver()
params_as_json = params_solver.json(indent=4)
print(params_as_json)

params_as_json = params_solver.flow360_json()
print(params_as_json)

# Class name removal from loc
params_as_dict["freestream"]["foo"] = "bar"
errors, warnings = validate_model(params_as_dict, "SI")
pprint(errors)
