"""
I run my first simulation, I want to run simulations on 100 variations of my geometries
"""

from ..inputs import Geometry
from ..meshing_param.params import MeshingParams, SurfaceRefinement, ZoneRefinement
from ..operating_condition import ExternalFlowOperatingConditions
from ..simulation import Simulation

simulation_ID = "f113d93a-c61a-4438-84af-f760533bbce4"
simulation = Simulation(simulation_ID)

file_list = [f"geometry_{i}.step" for i in range(100)]

for file in file_list:
    simulation.geometry = Geometry.from_file(file)
    simulation.run()
