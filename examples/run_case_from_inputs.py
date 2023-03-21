from flow360 import VolumeMesh, Case
from flow360 import Flow360MeshParams, MeshBoundary, Flow360Params

from testcases import OM6test

OM6test.get_files()

# submit mesh with manual configuration
meshParams = Flow360MeshParams(boundaries=MeshBoundary(noSlipWalls=[1]))
volumeMesh = VolumeMesh.from_file(OM6test.mesh_filename, meshParams, name="OM6wing-mesh")
print(volumeMesh)


# submit case manual configuration

from flow360.component.flow360_solver_params import (
    Geometry,
    TimeStepping,
    Freestream,
    NoSlipWall,
    SlipWall,
    FreestreamBoundary,
)

params = Flow360Params()

params.geometry = Geometry(
    refArea=1.15315084119231, momentLength=[1.47602, 0.801672958512342, 1.47602]
)
params.freestream = Freestream(
    muRef=4.2925193198151646e-8, Mach=0.84, Temperature=288.15, alpha=3.06
)
params.time_stepping = TimeStepping(maxPseudoSteps=500)
params.boundaries = {
    "1": NoSlipWall(name="wing"),
    "2": SlipWall(name="symmetry"),
    "3": FreestreamBoundary(name="freestream"),
}

print(params.geometry)
print(params.freestream)
print(params.navier_stokes_solver)
print(params.turbulence_model_solver)
print(params.time_stepping)
print(params.boundaries)

case = Case.new("OM6wing", params, volumeMesh.id)
case.submit()
print(case)
