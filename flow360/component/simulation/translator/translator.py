from flow360.component.simulation.simulation import SimulationParams
from flow360.component.simulation.volumes import FluidDynamics

from flow360.exceptions import (
    Flow360ConfigError,
    Flow360NotImplementedError,
    Flow360RuntimeError,
)

from flow360.component.flow360_params.flow360_params import Flow360Params
from flow360.component.flow360_params.time_stepping import SteadyTimeStepping, RampCFL

from flow360.component.flow360_params.flow360_params import Geometry as GeometryV1
from flow360.component.flow360_params.flow360_params import (
    NavierStokesSolver as NavierStokesSolverV1,
)
from flow360.component.flow360_params.solvers import (
    SpalartAllmaras as SpalartAllmarasV1,
)
from flow360.component.flow360_params.flow360_params import (
    FreestreamFromMachReynolds as FreestreamFromMachReynoldsV1,
)

from flow360.component.simulation.physics_components import SpalartAllmaras

import flow360.component.flow360_params.boundaries as bdV1

from flow360 import flow360_unit_system


def convert_SimulationParams_to_Flow360Params(input_param: SimulationParams) -> Flow360Params:
    if isinstance(input_param, SimulationParams) == False:
        raise Flow360ConfigError(
            f"Expected SimulationParams but got {input_param.__class__.__name__}"
        )

    main_volume = input_param.volumes[0]
    assert isinstance(main_volume, FluidDynamics)
    if main_volume.turbulence_model_solver is not None:
        assert isinstance(main_volume.turbulence_model_solver, SpalartAllmaras)

    boundaries = {}
    for surface in input_param.surfaces:
        for patch in surface.entities:
            # print(">>> CLASS = ", surface.__class__.__name__)
            boundaries[patch.mesh_patch_name] = getattr(bdV1, surface.__class__.__name__)(
                name=patch.custom_name
            )

    with flow360_unit_system:
        flow360_params = Flow360Params(
            geometry=GeometryV1(
                ref_area=input_param.reference_geometry.area,
                moment_center=input_param.reference_geometry.moment_center,
                moment_length=input_param.reference_geometry.moment_length,
                mesh_unit=1,
            ),
            navier_stokes_solver=NavierStokesSolverV1(
                **main_volume.navier_stokes_solver.model_dump()
            ),
            turbulence_model_solver=SpalartAllmarasV1(
                **main_volume.turbulence_model_solver.model_dump()
            ),
            freestream=FreestreamFromMachReynoldsV1(
                Mach=input_param.operating_condition.Mach,
                Reynolds=input_param.operating_condition.Reynolds,
                temperature=input_param.operating_condition.temperature,
                alpha=input_param.operating_condition.alpha,
                beta=input_param.operating_condition.beta,
            ),
            boundaries=boundaries,
            time_stepping=SteadyTimeStepping(CFL=RampCFL()),
        )

    return flow360_params
