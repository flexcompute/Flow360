from flow360 import SI_unit_system
from flow360 import units as u
from flow360.component.case import Case
from flow360.component.not_implemented import Geometry
from flow360.component.simulation.meshing_param.params import Farfield, MeshingParams
from flow360.component.simulation.meshing_param.volume_params import (
    AxisymmetricRefinement,
)
from flow360.component.simulation.models.volumes_models import Rotation
from flow360.component.simulation.operating_condition import (
    ExternalFlowOperatingConditions,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput
from flow360.component.simulation.physics_components import (
    NavierStokesSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.primitives import Cylinder
from flow360.component.simulation.references import ReferenceGeometry
from flow360.component.simulation.simulation import SimulationParams
from flow360.component.simulation.surfaces import Surface, Wall
from flow360.component.simulation.time_stepping.time_stepping import Steady

wing_surface = Surface(mesh_patch_name="1")

rotation_zone_inner = Cylinder(
    axis=(1, 1, 0), center=(0, 0, 0), height=1, inner_radius=0, outer_radius=2
)

rotation_zone_outer = Cylinder(
    axis=(0, 1, 0), center=(0, 0, 0), height=4, inner_radius=0, outer_radius=5
)


with SI_unit_system:
    simulationParams = SimulationParams(
        # Global settings, skiped in current example
        meshing=MeshingParams(
            refinement_factor=1,
            farfield=Farfield(type="auto"),
            refinements=[
                AxisymmetricRefinement(
                    entities=[rotation_zone_inner],
                    # enclosed_objects=[wing_surface], # this will be automatically populated by analysing topology
                    spacing_axial=0.1,
                    spacing_radial=0.1,
                    spacing_circumferential=0.2,
                ),
                AxisymmetricRefinement(
                    entities=[rotation_zone_outer],
                    # enclosed_objects=[rotation_zone_inner], # this will be automatically populated by analysing topology
                    spacing_axial=0.1,
                    spacing_radial=0.1,
                    spacing_circumferential=0.2,
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
                navier_stokes_solver=NavierStokesSolver(),
                turbulence_model_solver=SpalartAllmaras(),
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
            Rotation(
                entities=[rotation_zone_inner],
                angular_velocity=5 * u.rpm,
            ),
            Rotation(
                entities=[rotation_zone_outer],
                angular_velocity=2 * u.rad / u.s,
            ),
        ],
        surfaces=[
            Wall(entities=[wing_surface], use_wall_function=False),
            ## TODO: FreestreamBoundary is supposed to be auto populated since we have farfield=Farfield(type="auto"),
            # FreestreamBoundary(entities=[far_field]),
        ],
        time_stepping=Steady(),  # MRF or use unsteady for sliding interface
        outputs=[
            SurfaceOutput(entities=[wing_surface], output_fields=["primitiveVars"]),
        ],
    )

om6wing_geometry = Geometry.from_file("./om6wing.csm", name="my_om6wing_geo")
case = Case.submit(geometry=om6wing_geometry, param=simulationParams)
