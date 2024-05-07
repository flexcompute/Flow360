from flow360 import SI_unit_system
from flow360 import units as u
from flow360.component.simulation.operating_condition import (
    ExternalFlowOperatingConditions,
)
from flow360.component.simulation.physics_components import (
    LinearSolver,
    NavierStokesSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.references import ReferenceGeometry
from flow360.component.simulation.simulation import Simulation
from flow360.component.simulation.starting_points.volume_mesh import VolumeMesh
from flow360.component.simulation.surfaces import (
    FreestreamBoundary,
    NoSlipWall,
    Patch,
    SlipWall,
)
from flow360.component.simulation.volumes import FluidDynamics

om6wing_mesh = VolumeMesh.from_file(
    file_name="/local_data/ben/testcases/localTests/om6Wing/wing_tetra.1.lb8.ugrid",
    name="OM6wing-mesh",
)


wing_surface = Patch(mesh_patch_name="1", custom_name="wing_surface")
slip_wall = Patch(mesh_patch_name="2", custom_name="slip_wall")
far_field = Patch(mesh_patch_name="3", custom_name="far_field")

# with SI_unit_system:
simulation = Simulation(
    volume_mesh=om6wing_mesh,
    reference_geometry=ReferenceGeometry(
        area=1.15315084119231,
        moment_center=(0, 0, 0),
        moment_length=(0.801672958512342, 0.801672958512342, 0.801672958512342),
    ),
    operating_condition=ExternalFlowOperatingConditions(
        Mach=0.84, temperature=288.15, alpha=3.06 * u.deg, Reynolds=14.6e6
    ),
    volumes=[
        FluidDynamics(
            navier_stokes_solver=NavierStokesSolver(
                linear_solver=LinearSolver(absolute_tolerance=1e-10)
            ),
            turbulence_model_solver=SpalartAllmaras(),
        ),
    ],
    surfaces=[
        NoSlipWall(
            entities=[wing_surface],
        ),
        SlipWall(entities=[slip_wall]),
        FreestreamBoundary(entities=[far_field]),
    ],
)
simulation.run()
