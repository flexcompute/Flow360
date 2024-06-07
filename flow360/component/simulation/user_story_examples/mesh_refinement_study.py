"""
I want to perform mesh refinement study
"""

from ..inputs import Geometry
from ..meshing_param.params import MeshingParams, SurfaceRefinement, ZoneRefinement
from ..operating_condition import ExternalFlowOperatingConditions
from ..simulation import Simulation

simulation_ID = "f113d93a-c61a-4438-84af-f760533bbce4"
simulation = Simulation(simulation_ID)

for max_edge_length_new, first_layer_thickness_new in zip([0.01, 0.05, 0.25], [1e-4, 1e-5, 1e-6]):
    new_mesh_param = MeshingParams(
        face_refinement=[
            SurfaceRefinement(entities=["*"], max_edge_length=max_edge_length_new),
        ],
        zone_refinement=[
            ZoneRefinement(entities=["*"], first_layer_thickness=first_layer_thickness_new),
        ],
    )

    simulation.meshing = new_mesh_param
    simulation.run()
