from flow360 import SI_unit_system
from flow360 import units as u
from flow360.component.case import Case
from flow360.component.simulation.meshing_param.params import Farfield, MeshingParams
from flow360.component.simulation.meshing_param.volume_params import (
    AxisymmetricRefinement,
)
from flow360.component.simulation.models.material import Air
from flow360.component.simulation.models.volume_models import (
    ActuatorDisk,
    BETDisk,
    FluidDynamics,
)
from flow360.component.simulation.operating_condition import (
    ExternalFlowOperatingConditions,
)
from flow360.component.simulation.outputs.outputs import VolumeOutput
from flow360.component.simulation.physics_components import (
    LinearSolver,
    NavierStokesSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.primitives import Cylinder
from flow360.component.simulation.references import ReferenceGeometry
from flow360.component.simulation.simulation import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady
from flow360.component.surface_mesh import SurfaceMesh

my_actuator_disk = Cylinder(
    axis=(1, 1, 0), center=(0, 2, 1), height=1, inner_radius=0, outer_radius=2
)

my_zone_for_BETDisk_1 = Cylinder(
    axis=(1, 1, 0), center=(0, -2, -2), height=1, inner_radius=0, outer_radius=3
)
my_zone_for_BETDisk_2 = my_zone_for_BETDisk_1.copy(
    center=(0, 0, 2)
)  # Shifted the center to mimic array of BETDisk


with SI_unit_system:
    simulationParams = SimulationParams(
        # Global settings, skiped in current example
        meshing=MeshingParams(
            refinement_factor=1,
            farfield=Farfield(type="auto"),
            refinements=[
                AxisymmetricRefinement(
                    entities=[my_actuator_disk],
                    spacing_axial=my_actuator_disk.height / 1000,
                    spacing_radial=my_actuator_disk.outer_radius / 1000,
                    spacing_circumferential=my_actuator_disk.height / 2000,
                ),
                AxisymmetricRefinement(
                    entities=[my_zone_for_BETDisk_1, my_zone_for_BETDisk_2],
                    spacing_axial=my_zone_for_BETDisk_1.height / 1100,
                    spacing_radial=my_zone_for_BETDisk_1.outer_radius / 1200,
                    spacing_circumferential=my_zone_for_BETDisk_1.height / 2300,
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
            FluidDynamics(
                entities=["*"],  # This means we apply settings to all the volume zones
                navier_stokes_solver=NavierStokesSolver(
                    linear_solver=LinearSolver(absolute_tolerance=1e-10)
                ),
                turbulence_model_solver=SpalartAllmaras(),
                material=Air(),
                operating_condition=ExternalFlowOperatingConditions(
                    Mach=0.3,
                    temperature=288.15,
                ),
                reference_geometry=ReferenceGeometry(
                    area=1,
                    moment_length=2,
                    mesh_unit=3 * u.m,
                ),
            ),
            BETDisk(
                entities=[my_zone_for_BETDisk_1, my_zone_for_BETDisk_2],
                rotation_direction_rule="leftHand",
                # `center_of_rotation` will be populated by entities center
                # `axis_of_rotation` will be populated by entities axis
                number_of_blades=3,
                radius=2.5,  # If left blank, it will be entities' outer radius
                omega=5,
                chord_ref=14 * u.inch,
                thickness=0.9,  # If left blank, it will be entities' height
                n_loading_nodes=20,
                mach_numbers=[0.4],
                reynolds_numbers=[1000],
                twists=...,  # Planned for loading BET setting from files.
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
        time_stepping=Steady(),  # MRF or use unsteady for sliding interface
        outputs=[
            VolumeOutput(entities=["*"], output_fields=["Cp"]),
        ],
    )

case = Case.submit(surface_mesh=SurfaceMesh.from_file("fuselage.cgns"), param=simulationParams)
