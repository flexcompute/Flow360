import unittest

import numpy as np
import pydantic.v1 as pd
import pytest

import flow360 as fl
from flow360.component.v1.flow360_params import VolumeZones
from flow360.component.v1.volume_zones import (
    FluidDynamicsVolumeZone,
    HeatTransferVolumeZone,
    InitialConditionHeatTransfer,
    ReferenceFrame,
)
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.mark.usefixtures("array_equality_override")
def test_volume_zones():
    with pytest.raises(pd.ValidationError):
        with fl.SI_unit_system:
            rf = ReferenceFrame(
                center=(0, 0, 0),
                axis=(0, 0, 1),
            )

    with fl.SI_unit_system:
        rf = ReferenceFrame(center=(0, 0, 0), axis=(0, 0, 1), omega=1)

    assert rf

    with pytest.raises(pd.ValidationError):
        zone = HeatTransferVolumeZone(thermal_conductivity=-1)

    zone = HeatTransferVolumeZone(thermal_conductivity=1, volumetric_heat_source=0)

    assert zone

    zone = HeatTransferVolumeZone(thermal_conductivity=1, volumetric_heat_source="0")

    assert zone

    zone = HeatTransferVolumeZone(thermal_conductivity=1, volumetric_heat_source=1)

    assert zone

    zone = HeatTransferVolumeZone(thermal_conductivity=1, volumetric_heat_source="1")

    assert zone

    with pytest.raises(pd.ValidationError):
        zone = HeatTransferVolumeZone(thermal_conductivity=1, volumetric_heat_source=-1)

    zones = VolumeZones(
        zone1=FluidDynamicsVolumeZone(), zone2=HeatTransferVolumeZone(thermal_conductivity=1)
    )

    assert zones

    with fl.SI_unit_system:
        param = fl.Flow360Params(
            geometry=fl.Geometry(mesh_unit=1),
            fluid_properties=fl.air,
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=288.15, mu_ref=1),
            volume_zones={
                "zone1": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrameDynamic(axis=(1, 2, 3), center=(1, 1, 1))
                ),
                "zone2": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrameExpression(
                        axis=(1, 2, 3), center=(1, 1, 1), theta_degrees="0.2*t"
                    )
                ),
                "zone3": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrameOmegaRadians(
                        axis=(1, 4, 3), center=(1, 1, 1), omega_radians=0.5
                    )
                ),
                "zone4": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrameOmegaDegrees(
                        axis=(1, 5, 3), center=(1, 1, 1), omega_degrees=1.5
                    )
                ),
                "zone5": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrame(axis=(-5, 2, 3), center=(1, 1, 1), omega=2.5)
                ),
            },
        )

    param.flow360_json()
    assert (
        abs(np.linalg.norm(np.array(param.volume_zones["zone1"].reference_frame.axis)) - 1) < 1e-10
    )
    assert (
        abs(np.linalg.norm(np.array(param.volume_zones["zone2"].reference_frame.axis)) - 1) < 1e-10
    )
    assert (
        abs(np.linalg.norm(np.array(param.volume_zones["zone3"].reference_frame.axis)) - 1) < 1e-10
    )
    assert (
        abs(np.linalg.norm(np.array(param.volume_zones["zone4"].reference_frame.axis)) - 1) < 1e-10
    )
    assert (
        abs(np.linalg.norm(np.array(param.volume_zones["zone5"].reference_frame.axis)) - 1) < 1e-10
    )

    with pytest.raises(pd.ValidationError):
        zone = HeatTransferVolumeZone(thermal_conductivity=-1)

    to_file_from_file_test(zones)

    with fl.SI_unit_system:
        param = fl.Flow360Params(
            geometry=fl.Geometry(mesh_unit=1),
            fluid_properties=fl.air,
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=288.15, mu_ref=1),
            volume_zones={
                "zone1": fl.HeatTransferVolumeZone(
                    thermal_conductivity=1,
                    heat_capacity=0.123,
                    initial_condition=fl.InitialConditionHeatTransfer(T=1234),
                )
            },
            time_stepping=fl.UnsteadyTimeStepping(
                physical_steps=123, time_step_size=0.1234, max_pseudo_steps=441
            ),
        )
    solver_params = param.to_solver()
    assert solver_params.heat_equation_solver is not None
    assert solver_params.heat_equation_solver.equation_eval_frequency == 11

    with fl.SI_unit_system:
        param = fl.Flow360Params(
            geometry=fl.Geometry(mesh_unit=1),
            fluid_properties=fl.air,
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=288.15, mu_ref=1),
            volume_zones={
                "zone1": fl.HeatTransferVolumeZone(
                    thermal_conductivity=1,
                    heat_capacity=0.123,
                    initial_condition=fl.InitialConditionHeatTransfer(T="1.5*y^0.5/1.56 ^3"),
                )
            },
            time_stepping=fl.SteadyTimeStepping(max_pseudo_steps=441),
        )
    solver_params = param.to_solver()
    assert solver_params.heat_equation_solver is not None
    assert solver_params.heat_equation_solver.equation_eval_frequency == 10
    assert (
        solver_params.volume_zones["zone1"].initial_condition.T == "1.5*powf(y, 0.5)/powf(1.56, 3)"
    )

    zones = VolumeZones(
        zone1=FluidDynamicsVolumeZone(reference_frame=rf),
        zone2=HeatTransferVolumeZone(
            thermal_conductivity=1,
            heat_capacity=1,
            initial_condition=InitialConditionHeatTransfer(T=100),
        ),
    )

    assert zones

    to_file_from_file_test(zones)
