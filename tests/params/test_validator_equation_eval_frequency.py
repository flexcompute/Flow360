import unittest

import pydantic as pd
import pytest

import flow360 as fl
from flow360.component.flow360_params.boundaries import (
    FreestreamBoundary,
    HeatFluxWall,
    IsothermalWall,
    MassInflow,
    MassOutflow,
    NoSlipWall,
    SlidingInterfaceBoundary,
    SlipWall,
    SolidAdiabaticWall,
    SolidIsothermalWall,
    SubsonicInflow,
    SubsonicOutflowMach,
    SubsonicOutflowPressure,
    SupersonicInflow,
    WallFunction,
)
from flow360.component.flow360_params.flow360_output import (
    IsoSurface,
    IsoSurfaceOutput,
    MonitorOutput,
    SliceOutput,
    SurfaceOutput,
    VolumeOutput,
)
from flow360.component.flow360_params.flow360_params import (
    Flow360Params,
    FreestreamFromMach,
    FreestreamFromVelocity,
    MeshBoundary,
    SteadyTimeStepping,
)
from flow360.component.flow360_params.initial_condition import (
    ExpressionInitialCondition,
)
from flow360.component.flow360_params.solvers import (
    HeatEquationSolver,
    IncompressibleNavierStokesSolver,
    SpalartAllmaras,
    TransitionModelSolver,
)
from flow360.component.flow360_params.time_stepping import UnsteadyTimeStepping

# release 23.3.2+ feature
# from flow360.component.flow360_params.turbulence_quantities import TurbulenceQuantities
from flow360.component.flow360_params.volume_zones import (
    FluidDynamicsVolumeZone,
    HeatTransferVolumeZone,
    InitialConditionHeatTransfer,
)
from flow360.exceptions import Flow360ValidationError
from tests.utils import compare_to_ref, to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_equation_eval_frequency_for_unsteady_simulations():
    with fl.SI_unit_system:
        param = Flow360Params(
            time_stepping=UnsteadyTimeStepping(max_pseudo_steps=30),
            turbulence_model_solver=SpalartAllmaras(equation_eval_frequency=2),
            transition_model_solver=TransitionModelSolver(equation_eval_frequency=4),
        )

    with fl.SI_unit_system:
        param = Flow360Params(
            time_stepping=SteadyTimeStepping(max_pseudo_steps=10),
            turbulence_model_solver=SpalartAllmaras(equation_eval_frequency=12),
            transition_model_solver=TransitionModelSolver(equation_eval_frequency=15),
        )

    with pytest.raises(
        ValueError,
        match="'equation evaluation frequency' in turbulence_model_solver is greater than max_pseudo_steps.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                time_stepping=UnsteadyTimeStepping(max_pseudo_steps=2),
                turbulence_model_solver=SpalartAllmaras(equation_eval_frequency=3),
            )
    with pytest.raises(
        ValueError,
        match="'equation evaluation frequency' in transition_model_solver is greater than max_pseudo_steps.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                time_stepping=UnsteadyTimeStepping(max_pseudo_steps=2),
                transition_model_solver=TransitionModelSolver(equation_eval_frequency=3),
            )
