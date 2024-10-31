from matplotlib.pyplot import show

from flow360 import u
from flow360 import Project, Case
from flow360 import Freestream, SymmetryPlane, Wall
from flow360 import AerospaceCondition
from flow360 import SimulationParams
from flow360 import SI_unit_system
import flow360 as fl

fl.env.preprod.active()

project = Project.from_cloud("$PROJECT_ID$")
project._case_cache.add_asset(Case.from_cloud("$CASE_ID$"))

case = project.case


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
