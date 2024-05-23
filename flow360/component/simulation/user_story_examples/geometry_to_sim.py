"""
I have geometry file (eg. STEP), I want to run my first simulation
"""

from ..inputs import Geometry
from ..operating_condition import ExternalFlowOperatingConditions
from ..simulation import Simulation
from ..models.volumes.volumes import FluidDynamics

geometry = Geometry.from_file("geometry_1.step")

sim = Simulation(
    geometry,
    operating_conditions=ExternalFlowOperatingConditions(pressure=1, alpha=0),
    volumes=[FluidDynamics()],
)

# execute
results = sim.run()
# or
results = web.run(sim)
