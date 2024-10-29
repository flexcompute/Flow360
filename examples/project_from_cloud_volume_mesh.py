from matplotlib.pyplot import show

import flow360.component.simulation.units as u
from flow360.component.project import Project
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SymmetryPlane,
    Wall,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.environment import dev

dev.active()

project = Project.from_cloud("prj-b8eb4cc7-4fb8-4baa-9bcd-f1cf6d73163d")

volume_mesh = project.volume_mesh

with SI_unit_system:
    params = SimulationParams(
        operating_condition=AerospaceCondition(velocity_magnitude=100 * u.m / u.s),
        models=[
            Wall(entities=[volume_mesh["1"]]),
            Freestream(entities=[volume_mesh["3"]]),
            SymmetryPlane(entities=[volume_mesh["2"]]),
        ],
    )

project.run_case(params=params)

residuals = project.case.results.nonlinear_residuals
residuals.as_dataframe().plot(x="pseudo_step", logy=True)
show()
