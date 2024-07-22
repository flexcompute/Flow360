import os

import flow360 as fl

fl.Env.dev.active()

from flow360.component.geometry import Geometry
from flow360.component.meshing.params import Farfield, Volume, VolumeMeshingParams
from flow360.examples import Airplane

# geometry
geometry_files = ["./data/BallValve/BallValve.SLDASM","./data/BallValve/Housing.SLDPRT","./data/BallValve/Pipe.SLDPRT","./data/BallValve/Valve.SLDPRT"]
geometry_draft = Geometry.from_file(geometry_files, name="geometry-ballValve-complete", solver_version="geoHoops-24.7.0")
geometry = geometry_draft.submit()
print(geometry)

