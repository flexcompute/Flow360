"""
I have my surface mesh, want to run my first simulation
"""

from ..inputs import SurfaceMesh
from ..meshing_param.params import MeshingParams, ZoneRefinement
from ..models.volume_models import FluidDynamics
from ..operating_condition import ExternalFlowOperatingConditions
from ..primitives import Box
from ..references import ReferenceGeometry
from ..simulation import Simulation

volume_zone = Box(name="WholeDomain", x_range=(1, 2), y_range=(1, 2), z_range=(1, 2))

sim = Simulation(
    SurfaceMesh.from_file("mesh.cgns"),
    meshing=MeshingParams(
        zone_refinements=[ZoneRefinement(shape=volume_zone, spacing=0.1)],
    ),
    reference_geometry=ReferenceGeometry(area=0.1),
    zones=[FluidDynamics(entities=[volume_zone])],
)


# execute
results = sim.run()
# or
results = web.run(sim)