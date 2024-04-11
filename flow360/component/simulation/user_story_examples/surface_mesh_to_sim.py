"""
I have my surface mesh, want to run my first simulation
"""

from ..inputs import SurfaceMesh
from ..mesh import MeshingParameters, ZoneRefinement
from ..operating_condition import ExternalFlowOperatingConditions
from ..references import ReferenceGeometry
from ..simulation import Simulation
from ..volumes import FluidDynamics
from ..zones import BoxZone

volume_zone = BoxZone(name="WholeDomain", x_range=(1, 2), y_range=(1, 2), z_range=(1, 2))

sim = Simulation(
    SurfaceMesh.from_file("mesh.cgns"),
    meshing=MeshingParameters(
        zone_refinements=[ZoneRefinement(shape=volume_zone, spacing=0.1)],
    ),
    reference_geometry=ReferenceGeometry(area=0.1),
    zones=[FluidDynamics(entities=[volume_zone])],
)


# execute
results = sim.run()
# or
results = web.run(sim)
