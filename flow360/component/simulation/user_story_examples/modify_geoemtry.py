"""
I already run my first simulation, I modified my geometry (added a flap), I want to re-run it
"""

from ..inputs import Geometry
from ..simulation import Simulation
from ..models.volumes.volumes import FluidDynamics

simulation_ID = "f113d93a-c61a-4438-84af-f760533bbce4"
new_geometry = Geometry.from_file("geometry_1.step")
sim = Simulation(simulation_ID)
sim.geometry = new_geometry

# execute
results = sim.run()
# or
results = web.run(sim)
