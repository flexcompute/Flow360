"""
My simulation diverged, I need to modify meshing parameters on one of the patches
"""

from ..meshing_param.params import FaceRefinement, MeshingParameters, ZoneRefinement
from ..operating_condition import ExternalFlowOperatingConditions
from ..simulation import Simulation

simulation_ID = "f113d93a-c61a-4438-84af-f760533bbce4"
simulation = Simulation(simulation_ID)

simulation.meshing.by_name("This_Zone").first_layer_thickness = 1e-4
simulation.run()
