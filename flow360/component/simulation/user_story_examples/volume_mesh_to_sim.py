"""
I have my volume mesh, want to run my first simulation
"""

from ..inputs import VolumeMesh
from ..meshing_param.params import MeshingParams, ZoneRefinement
from ..models.volume_models import FluidDynamics
from ..operating_condition import ExternalFlowOperatingConditions
from ..primitives import Box
from ..references import ReferenceGeometry
from ..simulation import Simulation

volume_zone = Box(name="WholeDomain", x_range=(1, 2), y_range=(1, 2), z_range=(1, 2))

sim = Simulation(
    VolumeMesh.from_file("mesh.cgns"),
    reference_geometry=ReferenceGeometry(area=0.1),
    zones=[FluidDynamics(entities=[volume_zone])],
)


# execute
results = sim.run()
# or
results = web.run(sim)
