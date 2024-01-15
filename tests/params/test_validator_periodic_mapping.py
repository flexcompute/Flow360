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


def test_periodic_boundary_mapping():
    with fl.SI_unit_system:
        param = Flow360Params(
            boundaries={
                "blk-1/left": TranslationallyPeriodic(paired_patch_name="blk-1/right"),
                "blk-1/right": TranslationallyPeriodic(),
                "blk-1/top": RotationallyPeriodic(),
                "blk-1/bottom": RotationallyPeriodic(paired_patch_name="blk-1/top"),
            },
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

        with pytest.raises(
            ValueError,
            match="blk-1/left's paired_patch_name should not be equal to the name of itself.",
        ):
            param = Flow360Params(
                boundaries={
                    "blk-1/left": TranslationallyPeriodic(paired_patch_name="blk-1/left"),
                    "blk-1/right": TranslationallyPeriodic(),
                },
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
        with pytest.raises(
            ValueError,
            match="blk-1/left and its paired boundary blk-1/right do not have the same type of boundary condition.",
        ):
            param = Flow360Params(
                boundaries={
                    "blk-1/right": NoSlipWall(),
                    "blk-1/left": TranslationallyPeriodic(paired_patch_name="blk-1/right"),
                },
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )

        with pytest.raises(
            ValueError,
            match="blk-1/left's paired_patch_name does not exist in boundaries.",
        ):
            param = Flow360Params(
                boundaries={
                    "blk-1/left": TranslationallyPeriodic(paired_patch_name="blk-1/dummyRight"),
                },
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )

        param = Flow360Params(
            boundaries={
                "blk-1/left": TranslationallyPeriodic(
                    paired_patch_name="blk-1/right", translation_vector=(1, 2, 3)
                ),
                "blk-1/right": TranslationallyPeriodic(),
                "blk-1/top": RotationallyPeriodic(),
                "blk-1/bottom": RotationallyPeriodic(
                    paired_patch_name="blk-1/top", axis_of_rotation=(1, 2, 3), theta_radians=0.3
                ),
            },
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

        with pytest.raises(
            ValueError,
            match="Flow360 doesn't allow periodic pairing information of .* specified for both patches.",
        ):
            param = Flow360Params(
                boundaries={
                    "blk-1/left": TranslationallyPeriodic(
                        paired_patch_name="blk-1/right", translation_vector=(1, 2, 3)
                    ),
                    "blk-1/right": TranslationallyPeriodic(paired_patch_name="blk-1/left"),
                },
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
        with pytest.raises(
            ValueError,
            match="Flow360 doesn't allow periodic pairing information of .* specified for both patches.",
        ):
            param = Flow360Params(
                boundaries={
                    "blk-1/left": TranslationallyPeriodic(
                        paired_patch_name="blk-1/right", translation_vector=(1, 2, 3)
                    ),
                    "blk-1/right": TranslationallyPeriodic(translation_vector=(-1, -2, -3)),
                },
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )

        with pytest.raises(
            ValueError,
            match="Flow360 doesn't allow periodic pairing information of .* specified for both patches.",
        ):
            param = Flow360Params(
                boundaries={
                    "blk-1/top": RotationallyPeriodic(paired_patch_name="blk-1/bottom"),
                    "blk-1/bottom": RotationallyPeriodic(
                        paired_patch_name="blk-1/top", axis_of_rotation=(1, 2, 3), theta_radians=0.3
                    ),
                },
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
        with pytest.raises(
            ValueError,
            match="Flow360 doesn't allow periodic pairing information of .* specified for both patches.",
        ):
            param = Flow360Params(
                boundaries={
                    "blk-1/top": RotationallyPeriodic(axis_of_rotation=(1, 2, 3)),
                    "blk-1/bottom": RotationallyPeriodic(paired_patch_name="blk-1/top"),
                },
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
        with pytest.raises(
            ValueError,
            match="Periodic pair for patch blk-1.* is not specified.",
        ):
            param = Flow360Params(
                boundaries={
                    "blk-1/top": RotationallyPeriodic(),
                    "blk-1/bottom": RotationallyPeriodic(),
                },
                freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )
