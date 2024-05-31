"""
I have geometry file (eg. STEP), I want to run my first simulation
"""

from ..inputs import Geometry
from ..models.volume_models import FluidDynamics
from ..operating_condition import ExternalFlowOperatingConditions
from ..simulation import Simulation

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
