"""
I want to fine-tune my turbulence model solver
"""

from ..inputs import Geometry
from ..meshing_param.params import MeshingParams, SurfaceRefinement, ZoneRefinement
from ..operating_condition import ExternalFlowOperatingConditions
from ..simulation import Simulation

simulation_ID = "f113d93a-c61a-4438-84af-f760533bbce4"
simulation = Simulation(simulation_ID)

for new_CDES in [0.1, 1, 10]:
    simulation.volumes.by_name("this_zone").turbulence_model_solver.turbulence_constants.C_DES = (
        new_CDES
    )
    simulation.run()
