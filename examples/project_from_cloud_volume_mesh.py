import flow360.component.simulation.units as u
from flow360.component.project import Project
from flow360.component.simulation.models.surface_models import Freestream, Wall
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.environment import dev

dev.active()

project = Project.from_cloud("prj-e8c6c7eb-c18b-4c15-bac8-edf5aaf9b155")

volume_mesh = project.volume_mesh

with SI_unit_system:
    params = SimulationParams(
        operating_condition=AerospaceCondition(velocity_magnitude=100 * u.m / u.s),
        models=[
            Wall(entities=[volume_mesh["fluid/wall"]]),
            Freestream(entities=[volume_mesh["fluid/farfield"]]),
        ],
    )

project.run_case(params=params)
