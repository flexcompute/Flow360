import flow360 as fl
import flow360.component.simulation.units as u
from flow360.component.project import Project
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SymmetryPlane,
    Wall,
)
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.examples import OM6wing

fl.Env.dev.active()

OM6wing.get_files()
# Creating and uploading a volume mesh from file
project = Project.from_file(
    OM6wing.mesh_filename,
    name="wing-volume-mesh-python-upload",
    solver_version="workbench-24.9.3",
    tags=["python"],
)

volume_mesh = project.volume_mesh

with SI_unit_system:
    params = SimulationParams(
        operating_condition=AerospaceCondition(velocity_magnitude=100 * u.m / u.s),
        models=[
            Fluid(),
            Wall(entities=[volume_mesh["1"]]),
            Freestream(entities=[volume_mesh["3"]]),
            SymmetryPlane(entities=[volume_mesh["2"]]),
        ],
    )

project.run_case(params=params)
