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


def test_cht_solver_no_heat_transfer_zone():
    with fl.SI_unit_system:
        param = Flow360Params(
            volume_zones={
                "blk-1": FluidDynamicsVolumeZone(),
                "blk-2": FluidDynamicsVolumeZone(),
            }
        )

    with pytest.raises(ValueError, match="Heat equation solver activated with no zone definition."):
        with fl.SI_unit_system:
            param = Flow360Params(
                heat_equation_solver=HeatEquationSolver(),
                volume_zones={
                    "blk-1": FluidDynamicsVolumeZone(),
                    "blk-2": FluidDynamicsVolumeZone(),
                },
            )

    with pytest.raises(
        ValueError,
        match="SolidIsothermalWall boundary is defined with no definition of volume zone of heat transfer. ",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                boundaries={
                    "bnd1": SolidIsothermalWall(temperature=1.01),
                },
                volume_zones={
                    "blk-1": FluidDynamicsVolumeZone(),
                    "blk-2": FluidDynamicsVolumeZone(),
                },
            )

    with pytest.raises(
        ValueError,
        match="SolidAdiabaticWall boundary is defined with no definition of volume zone of heat transfer. ",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                boundaries={
                    "bnd1": SolidAdiabaticWall(),
                },
                volume_zones={
                    "blk-1": FluidDynamicsVolumeZone(),
                    "blk-2": FluidDynamicsVolumeZone(),
                },
            )

    with pytest.raises(
        ValueError,
        match="Heat equation output variables: residualHeatSolver is requested with no definition of volume zone of heat transfer.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                surface_output=SurfaceOutput(output_fields=["residualHeatSolver"]),
                volume_zones={
                    "blk-1": FluidDynamicsVolumeZone(),
                    "blk-2": FluidDynamicsVolumeZone(),
                },
            )

    with pytest.raises(
        ValueError,
        match="Heat equation output variables: residualHeatSolver is requested with no definition of volume zone of heat transfer.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                volume_output=VolumeOutput(output_fields=["residualHeatSolver"]),
                volume_zones={
                    "blk-1": FluidDynamicsVolumeZone(),
                    "blk-2": FluidDynamicsVolumeZone(),
                },
            )
    with pytest.raises(
        ValueError,
        match="Heat equation output variables: residualHeatSolver is requested with no definition of volume zone of heat transfer.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                slice_output=SliceOutput(output_fields=["residualHeatSolver"]),
                volume_zones={
                    "blk-1": FluidDynamicsVolumeZone(),
                    "blk-2": FluidDynamicsVolumeZone(),
                },
            )
    with pytest.raises(
        ValueError,
        match="Heat equation output variables: residualHeatSolver is requested with no definition of volume zone of heat transfer.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                iso_surface_output=IsoSurfaceOutput(
                    iso_surfaces={
                        "iso_surface_0": IsoSurface(output_fields=["residualHeatSolver"]),
                        "iso_surface_1": IsoSurface(output_fields=["residualHeatSolver"]),
                    }
                ),
                volume_zones={
                    "blk-1": FluidDynamicsVolumeZone(),
                    "blk-2": FluidDynamicsVolumeZone(),
                },
            )
    with pytest.raises(
        ValueError,
        match="Heat equation output variables: residualHeatSolver is requested with no definition of volume zone of heat transfer.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                monitor_output=MonitorOutput(output_fields=["residualHeatSolver"]),
                volume_zones={
                    "blk-1": FluidDynamicsVolumeZone(),
                    "blk-2": FluidDynamicsVolumeZone(),
                },
            )


def test_cht_solver_has_heat_transfer_zone():
    with fl.SI_unit_system:
        param = Flow360Params(
            volume_zones={
                "blk-1": HeatTransferVolumeZone(thermal_conductivity=0.1),
                "blk-2": FluidDynamicsVolumeZone(),
            }
        )

    with pytest.raises(
        ValueError, match="Conjugate heat transfer can not be used with incompressible flow solver."
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                navier_stokes_solver=IncompressibleNavierStokesSolver(),
                volume_zones={
                    "blk-1": HeatTransferVolumeZone(thermal_conductivity=0.1),
                    "blk-2": FluidDynamicsVolumeZone(),
                },
            )

    with fl.SI_unit_system:
        param = Flow360Params(
            time_stepping=UnsteadyTimeStepping(),
            volume_zones={
                "blk-1": HeatTransferVolumeZone(
                    thermal_conductivity=0.1,
                    heat_capacity=0.1,
                    initial_condition=InitialConditionHeatTransfer(T_solid=1.2),
                ),
                "blk-2": FluidDynamicsVolumeZone(),
            },
        )

    with pytest.raises(
        ValueError,
        match="Heat capacity needs to be specified for all heat transfer volume zones for unsteady simulations.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                time_stepping=UnsteadyTimeStepping(),
                volume_zones={
                    "blk-1": HeatTransferVolumeZone(thermal_conductivity=0.1),
                    "blk-2": FluidDynamicsVolumeZone(),
                },
            )
    with pytest.raises(
        ValueError,
        match="Initial condition needs to be specified for all heat transfer volume zones for unsteady simulations.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                time_stepping=UnsteadyTimeStepping(),
                volume_zones={
                    "blk-1": HeatTransferVolumeZone(thermal_conductivity=0.1, heat_capacity=0.1),
                    "blk-2": FluidDynamicsVolumeZone(),
                },
            )
    with fl.SI_unit_system:
        param = Flow360Params(
            initial_condition=ExpressionInitialCondition(rho=1, u=1, v=1, w=1, p=1),
            volume_zones={
                "blk-1": HeatTransferVolumeZone(
                    thermal_conductivity=0.1,
                    initial_condition=InitialConditionHeatTransfer(T_solid=1.1),
                ),
                "blk-2": FluidDynamicsVolumeZone(),
            },
        )

    with pytest.raises(
        ValueError,
        match="Initial condition needs to be specified for all heat transfer zones for initialization with expressions.",
    ):
        with fl.SI_unit_system:
            param = Flow360Params(
                initial_condition=ExpressionInitialCondition(rho=1, u=1, v=1, w=1, p=1),
                volume_zones={
                    "blk-1": HeatTransferVolumeZone(thermal_conductivity=0.1),
                    "blk-2": FluidDynamicsVolumeZone(),
                },
            )
