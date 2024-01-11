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
    RotationallyPeriodic,
    SlidingInterfaceBoundary,
    SlipWall,
    SolidAdiabaticWall,
    SolidIsothermalWall,
    SubsonicInflow,
    SubsonicOutflowMach,
    SubsonicOutflowPressure,
    SupersonicInflow,
    TranslationallyPeriodic,
    WallFunction,
)
from flow360.component.flow360_params.flow360_output import (
    AeroacousticOutput,
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


def test_aero_acoustics():
    """
    todo: warning for inaccurate simulations
    """
    with fl.SI_unit_system:
        param = Flow360Params(
            boundaries={
                "blk-1/left": TranslationallyPeriodic(paired_patch_name="blk-1/right"),
                "blk-1/right": TranslationallyPeriodic(),
            }
        )
        param = Flow360Params(
            aeroacoustic_output=AeroacousticOutput(observers=[(1, 2, 3), (4, 5, 6)]),
            boundaries={
                "blk-1/right": NoSlipWall(),
            },
        )
