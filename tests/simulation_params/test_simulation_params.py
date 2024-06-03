from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.meshing_param.params import MeshingParameters
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.primitives import Box
from flow360.component.simulation.unit_system import SI_unit_system
import flow360.component.simulation.units as u


with SI_unit_system:
    my_box = Box(
        name="my_box",
        center=(1.2, 2.3, 3.4) * u.m,
        size=(1.0, 2.0, 3.0) * u.m,
        axes=((1, 0, 1), (1, 2, 3)),
    )
    param = SimulationParams(
        meshing=MeshingParameters(
            farfield="auto",
            refinement_factor=1.0,
            gap_treatment_strength=0.5,
            surface_layer_growth_rate=1.5,
            refinements=[UniformRefinement(entities=[my_box], spacing=0.1)],
        )
    )
param.to_file("_test_param.json")
