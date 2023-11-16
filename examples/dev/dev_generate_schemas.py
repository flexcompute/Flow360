import flow360 as fl


def write_to_file(name, content):
    with open(f"{name}.json", "w") as outfile:
        outfile.write(content)


write_to_file(fl.Geometry.__name__, fl.Geometry.generate_schema())
write_to_file(fl.Freestream.__name__, fl.Freestream.generate_schema())
write_to_file(fl.SlidingInterface.__name__, fl.SlidingInterface.generate_schema())
write_to_file(fl.NavierStokesSolver.__name__, fl.NavierStokesSolver.generate_schema())
write_to_file(fl.TurbulenceModelSolverSA.__name__, fl.TurbulenceModelSolverSA.generate_schema())
write_to_file(fl.TurbulenceModelSolverSST.__name__, fl.TurbulenceModelSolverSST.generate_schema())
write_to_file(fl.TransitionModelSolver.__name__, fl.TransitionModelSolver.generate_schema())
write_to_file(fl.HeatEquationSolver.__name__, fl.HeatEquationSolver.generate_schema())
write_to_file(fl.NoneSolver.__name__, fl.NoneSolver.generate_schema())
write_to_file(fl.PorousMedium.__name__, fl.PorousMedium.generate_schema())
write_to_file(fl.TimeStepping.__name__, fl.TimeStepping.generate_schema())
write_to_file(fl.ActuatorDisk.__name__, fl.ActuatorDisk.generate_schema())
write_to_file(fl.BETDisk.__name__, fl.BETDisk.generate_schema())
write_to_file(fl.SurfaceOutput.__name__, fl.SurfaceOutput.generate_schema())
write_to_file(fl.SliceOutput.__name__, fl.SliceOutput.generate_schema())
write_to_file(fl.VolumeOutput.__name__, fl.VolumeOutput.generate_schema())
write_to_file(fl.AeroacousticOutput.__name__, fl.AeroacousticOutput.generate_schema())
write_to_file(fl.MonitorOutput.__name__, fl.MonitorOutput.generate_schema())
write_to_file(fl.IsoSurfaceOutput.__name__, fl.IsoSurfaceOutput.generate_schema())

write_to_file(fl.Surfaces.__name__, fl.Surfaces.generate_schema())
write_to_file(fl.VolumeZones.__name__, fl.VolumeZones.generate_schema())
write_to_file(fl.Boundaries.__name__, fl.Boundaries.generate_schema())
write_to_file(fl.Slices.__name__, fl.Slices.generate_schema())
write_to_file(fl.IsoSurfaces.__name__, fl.IsoSurfaces.generate_schema())

