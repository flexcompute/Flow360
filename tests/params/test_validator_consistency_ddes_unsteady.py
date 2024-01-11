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
from flow360.component.flow360_params.turbulence_quantities import TurbulenceQuantities
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


def test_consistency_ddes_unsteady():
    with fl.SI_unit_system:
        param = Flow360Params(
            time_stepping=UnsteadyTimeStepping(),
            turbulence_model_solver=SpalartAllmaras(DDES=True),
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

    with fl.SI_unit_system:
        param = Flow360Params(
            time_stepping=SteadyTimeStepping(),
            turbulence_model_solver=SpalartAllmaras(),
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )
    with fl.SI_unit_system:
        param = Flow360Params(
            turbulence_model_solver=SpalartAllmaras(),
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

    with pytest.raises(
        ValueError,
        match="Running DDES with steady simulation is invalid.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                time_stepping=SteadyTimeStepping(),
                turbulence_model_solver=SpalartAllmaras(DDES=True),
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )

    with pytest.raises(
        ValueError,
        match="Running DDES with steady simulation is invalid.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                turbulence_model_solver=SpalartAllmaras(DDES=True),
                boundaries={},
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
