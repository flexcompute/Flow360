from flow360 import SI_unit_system
from flow360 import units as u
from flow360.component.case import Case
from flow360.component.simulation.material import Air
from flow360.component.simulation.meshing_param.face_params import FaceRefinement
from flow360.component.simulation.meshing_param.params import MeshingParameters
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.operating_condition import (
    ExternalFlowOperatingConditions,
)
from flow360.component.simulation.outputs import Slice, SliceOutput, SurfaceOutput
from flow360.component.simulation.physics_components import (
    LinearSolver,
    NavierStokesSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.primitives import Box
from flow360.component.simulation.references import ReferenceGeometry
from flow360.component.simulation.simulation import SimulationParams
from flow360.component.simulation.surfaces import (
    FreestreamBoundary,
    SlipWall,
    Surface,
    Wall,
)
from flow360.component.simulation.time_stepping import SteadyTimeStepping
from flow360.component.simulation.user_defined_dynamics import UserDefinedDynamics
from flow360.component.simulation.volumes import FluidDynamics, PorousMedium
from flow360.component.surface_mesh import SurfaceMesh

##:: Volume and Surface Definition ::##
wing_surface = Surface(mesh_patch_name="1")
slip_wall = Surface(mesh_patch_name="2")
far_field = Surface(mesh_patch_name="3")


porous_media_zone = Box(
    center=[0, 0, 0],
    lengths=[0.2, 0.3, 2],
)

##:: Output entities definition ::##
slice_a = Slice(
    name="slice_a",
    slice_normal=(1, 1, 0),
    slice_origin=(0.1, 0, 0),
)

slice_b = Slice(
    name="slice_b",
    output_fields=["Cp"],
    slice_normal=(1, 1, 1),
    slice_origin=(0.1, 2, 0),
)

with SI_unit_system:
    simulationParams = SimulationParams(
        # Global settings, skiped in current example
        meshing=MeshingParameters(
            refinement_factor=1,
            refinements=[
                FaceRefinement(entities=[far_field], growth_rate=0.2),
                UniformRefinement(entities=[porous_media_zone], spacing=0.002),
            ],
        ),
        volumes=[
            FluidDynamics(
                entities=["*"],  # This means we apply settings to all the volume zones
                navier_stokes_solver=NavierStokesSolver(
                    linear_solver=LinearSolver(absolute_tolerance=1e-10)
                ),
                turbulence_model_solver=SpalartAllmaras(),
                material=Air(),
                operating_condition=ExternalFlowOperatingConditions(  # Per volume override
                    Mach=0.84,
                    temperature=288.15,
                    alpha=3.06 * u.deg,
                    # Reynolds=14.6e6 should be obtained from user
                ),
                reference_geometry=ReferenceGeometry(  # Per volume override
                    area=1.15315084119231,
                    moment_length=(0.801672958512342, 0.801672958512342, 0.801672958512342),
                    mesh_unit=1 * u.m,
                ),
            ),
            PorousMedium(
                entities=[porous_media_zone],
                axes=[[0, 1, 0], [0, 0, 1]],
                darcy_coefficient=[1e6, 0, 0],
                forchheimer_coefficient=[1, 0, 0],
                volumetric_heat_source=0,
            ),
        ],
        surfaces=[
            Wall(entities=[wing_surface], use_wall_function=True),
            SlipWall(entities=[slip_wall]),
            FreestreamBoundary(entities=[far_field]),
        ],
        time_stepping=SteadyTimeStepping(),
        outputs=[
            SurfaceOutput(entities=[slip_wall, far_field], output_fields=["Cf", "Cp"]),
            SurfaceOutput(entities=[wing_surface], output_fields=["primitiveVars"]),
            SliceOutput(entities=[slice_a, slice_b], output_fields=["temperature"]),
        ],
        user_defined_dynamics=[
            UserDefinedDynamics(
                name="alphaController",
                input_vars=["CL"],
                constants={"CLTarget": 0.4, "Kp": 0.2, "Ki": 0.002},
                output_vars={"alphaAngle": "if (pseudoStep > 500) state[0]; else alphaAngle;"},
                state_vars_initial_value=["alphaAngle", "0.0"],
                update_law=[
                    "if (pseudoStep > 500) state[0] + Kp * (CLTarget - CL) + Ki * state[1]; else state[0];",
                    "if (pseudoStep > 500) state[1] + (CLTarget - CL); else state[1];",
                ],
                input_boundary_patches=[wing_surface],
            )
        ],
    )

case = Case.submit(
    # Specifies the starting point.
    surface_mesh=SurfaceMesh.from_file(
        file_name="./SurfaceMesh.ugrid",
        name="My-surface-mesh",
    ),
    param=simulationParams,
)
