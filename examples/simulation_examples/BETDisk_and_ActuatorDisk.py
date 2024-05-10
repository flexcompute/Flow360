from flow360 import SI_unit_system
from flow360 import units as u
from flow360.component.simulation.material import Air
from flow360.component.simulation.operating_condition import (
    ExternalFlowOperatingConditions,
)
from flow360.component.simulation.outputs import VolumeOutput
from flow360.component.simulation.physics_components import (
    LinearSolver,
    NavierStokesSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.references import ReferenceGeometry
from flow360.component.simulation.simulation import SimulationParams
from flow360.component.simulation.surfaces import (
    Surface,
)
from flow360.component.simulation.time_stepping import SteadyTimeStepping
from flow360.component.simulation.volumes import BETDisk, ActuatorDisk
from flow360.component.surface_mesh import SurfaceMesh
from flow360.component.simulation.zones import CylindricalZone
from flow360.component.case import Case

from flow360.component.simulation.meshing_param.params import MeshingParameters, Farfield
from flow360.component.simulation.meshing_param.volume_params import RotorDisk

fuselage = Surface(mesh_patch_name="fuselage")

my_actuator_disk = CylindricalZone(
    axis=(1, 1, 0), center=(0, 2, 1), height=1, inner_radius=0, outer_radius=2
)

my_BETDisk = CylindricalZone(
    axis=(1, 1, 0), center=(0, -2, -2), height=1, inner_radius=0, outer_radius=3
)


with SI_unit_system:
    simulationParams = SimulationParams(
        # Global settings, skiped in current example
        meshing=MeshingParameters(
            refinement_factor=1,
            farfield=Farfield(type="auto"),
            zone_refinement=[
                RotorDisk(
                    entities=[my_actuator_disk],
                    spacing_axial=my_actuator_disk.height / 1000,
                    spacing_radial=my_actuator_disk.outer_radius / 1000,
                    spacing_circumferential=my_actuator_disk.height / 2000,
                ),
                RotorDisk(
                    entities=[my_BETDisk],
                    spacing_axial=my_BETDisk.height / 1100,
                    spacing_radial=my_BETDisk.outer_radius / 1200,
                    spacing_circumferential=my_BETDisk.height / 2300,
                ),
            ],
        ),
        operating_condition=ExternalFlowOperatingConditions(  # Per volume override
            Mach=0.84, temperature=288.15, alpha=1.06 * u.deg, beta=1e-2 * u.rad
        ),
        reference_geometry=ReferenceGeometry(  # Per volume override
            area=2,
            moment_length=(1, 1, 1),
            mesh_unit=1 * u.m,
        ),
        volumes=[
            BETDisk(
                entities=[my_BETDisk],
                navier_stokes_solver=NavierStokesSolver(
                    linear_solver=LinearSolver(absolute_tolerance=1e-10)
                ),
                turbulence_model_solver=SpalartAllmaras(),
                material=Air(),
                rotation_direction_rule="leftHand",
                center_of_rotation=my_BETDisk.center,
                axis_of_rotation=my_BETDisk.axis,  # This should be automatic?
                number_of_blades=3,
                radius=my_BETDisk.outer_radius / 1.1,
                omega=5,
                chord_ref=14 * u.inch,
                thickness=my_BETDisk.height,
                n_loading_nodes=20,
                mach_numbers=0.4,
                reynolds_numbers=1000,
                twists=...,
                chords=...,
                alphas=...,
                sectional_radiuses=...,
                sectional_polars=...,
            ),
            ActuatorDisk(
                entities=[my_actuator_disk],
                center=my_actuator_disk.center,
                axis_thrust=my_actuator_disk.axis,
                thickness=my_actuator_disk.height / 1.1,
                force_per_area=ForcePerArea(
                    radius=[...],
                    thrust=[...],
                    circumferential=[...],
                ),
            ),
        ],
        time_stepping=SteadyTimeStepping(),  # MRF or use unsteady for sliding interface
        outputs=[
            VolumeOutput(entities=["*"], output_fields=["Cp"]),
        ],
    )

case = Case.submit(surface_mesh=SurfaceMesh.from_file("fuselage.cgns"), param=simulationParams)
