from typing import Type

import flow360 as fl
from flow360.component.flow360_params.params_base import Flow360BaseModel


def write_to_file(name, content):
    with open(name, "w") as outfile:
        outfile.write(content)


def write_schemas(type_obj: Type[Flow360BaseModel]):
    schema = type_obj.generate_schema()
    write_to_file(f"./data/{type_obj.__name__}.json", schema)
    ui_schema = type_obj.generate_ui_schema()
    if ui_schema is not None:
        write_to_file(f"./data/{type_obj.__name__}.ui.json", ui_schema)


write_schemas(fl.NavierStokesSolver)
write_schemas(fl.Geometry)
# write_schemas(fl.Freestream)
write_schemas(fl.SlidingInterface)
write_schemas(fl.TurbulenceModelSolverSA)
write_schemas(fl.TurbulenceModelSolverSST)
write_schemas(fl.TransitionModelSolver)
write_schemas(fl.HeatEquationSolver)
write_schemas(fl.NoneSolver)
write_schemas(fl.PorousMedium)
write_schemas(fl.TimeStepping)
write_schemas(fl.ActuatorDisk)
write_schemas(fl.BETDisk)
write_schemas(fl.SurfaceOutput)
write_schemas(fl.SliceOutput)
write_schemas(fl.VolumeOutput)
write_schemas(fl.AeroacousticOutput)
write_schemas(fl.MonitorOutput)
write_schemas(fl.IsoSurfaceOutput)

write_schemas(fl.Surfaces)
write_schemas(fl.VolumeZones)
write_schemas(fl.Boundaries)
write_schemas(fl.Slices)
write_schemas(fl.IsoSurfaces)
