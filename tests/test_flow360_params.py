import json
import math
import re
import unittest

import pydantic as pd
import pytest

import flow360 as fl
from flow360.component.flow360_params.flow360_params import (
    ActuatorDisk,
    AeroacousticOutput,
    Flow360MeshParams,
    Flow360Params,
    FluidDynamicsVolumeZone,
    ForcePerArea,
    Freestream,
    FreestreamBoundary,
    Geometry,
    HeatEquationSolver,
    HeatTransferVolumeZone,
    InitialConditionHeatTransfer,
    IsothermalWall,
    LinearSolver,
    MassInflow,
    MassOutflow,
    MeshBoundary,
    MeshSlidingInterface,
    NavierStokesSolver,
    NoSlipWall,
    ReferenceFrame,
    SlidingInterface,
    SlidingInterfaceBoundary,
    SlipWall,
    SolidAdiabaticWall,
    SolidIsothermalWall,
    SubsonicInflow,
    SubsonicOutflowMach,
    SubsonicOutflowPressure,
    TimeStepping,
    TurbulenceModelSolver,
    VolumeZones,
    WallFunction,
)
from flow360.component.types import TimeStep
from flow360.exceptions import ConfigError, ValidationError

from .utils import compare_to_ref, to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_mesh_boundary():
    mesh_boundary = MeshBoundary.parse_raw(
        """
        {
        "noSlipWalls": [
            "fluid/fuselage",
            "fluid/leftWing",
            "fluid/rightWing"
        ]
    }
        """
    )
    assert mesh_boundary

    compare_to_ref(mesh_boundary, "ref/flow360mesh/mesh_boundary/yaml.yaml")
    compare_to_ref(mesh_boundary, "ref/flow360mesh/mesh_boundary/json.json")
    to_file_from_file_test(mesh_boundary)

    assert MeshBoundary.parse_raw(
        """
        {
        "noSlipWalls": [
            1,
            2,
            3
        ]
    }
        """
    )


def test_case_boundary():
    with pytest.raises(ValidationError):
        param = Flow360Params(
            boundaries={
                "fluid/fuselage": TimeStepping(),
                "fluid/leftWing": NoSlipWall(),
                "fluid/rightWing": NoSlipWall(),
            }
        )

    with pytest.raises(ValidationError):
        param = Flow360Params.parse_raw(
            """
            {
                "boundaries": {
                    "fluid/fuselage": {
                        "type": "UnsupportedBC"
                    },
                    "fluid/leftWing": {
                        "type": "NoSlipWall"
                    },
                    "fluid/rightWing": {
                        "type": "NoSlipWall"
                    } 
                }
            }
            """
        )
        print(param)

    param = Flow360Params.parse_raw(
        """
        {
            "boundaries": {
                "fluid/fuselage": {
                    "type": "SlipWall"
                },
                "fluid/leftWing": {
                    "type": "NoSlipWall"
                },
                "fluid/rightWing": {
                    "type": "NoSlipWall"
                } 
            }
        }
        """
    )

    assert param

    boundaries = fl.Boundaries(
        wing=NoSlipWall(), symmetry=SlipWall(), freestream=FreestreamBoundary()
    )

    assert boundaries

    boundaries["wing"] = fl.WallFunction()

    with pytest.raises(ValidationError):
        param = Flow360Params(
            boundaries={
                "fluid/fuselage": "NoSlipWall",
                "fluid/leftWing": NoSlipWall(),
                "fluid/rightWing": NoSlipWall(),
            }
        )

    param = Flow360Params(
        boundaries={
            "fluid/ fuselage": NoSlipWall(),
            "fluid/rightWing": NoSlipWall(),
            "fluid/leftWing": SolidIsothermalWall(temperature=10),
        }
    )

    assert param
    param.boundaries["fluid/leftWing"].temperature = 1

    compare_to_ref(param.boundaries, "ref/case_params/boundaries/yaml.yaml")
    compare_to_ref(param.boundaries, "ref/case_params/boundaries/json.json")
    to_file_from_file_test(param)
    to_file_from_file_test(param.boundaries)

    SolidAdiabaticWall()
    SolidIsothermalWall(Temperature=10)

    with pytest.raises(pd.ValidationError):
        SolidIsothermalWall(Temperature=-1)


def test_boundary_incorrect():
    with pytest.raises(pd.ValidationError):
        MeshBoundary.parse_raw(
            """
            {
            "NoSlipWalls": [
                "fluid/fuselage",
                "fluid/leftWing",
                "fluid/rightWing"
            ]
        }
            """
        )


def test_boundary_types():
    assert NoSlipWall().type == "NoSlipWall"
    assert NoSlipWall(velocity=(0, 0, 0))
    assert NoSlipWall(name="name", velocity=[0, 0, 0])
    assert NoSlipWall(velocity=("0", "0.1*x+exp(y)+z^2", "cos(0.2*x*pi)+sqrt(z^2+1)"))
    assert SlipWall().type == "SlipWall"
    assert FreestreamBoundary().type == "Freestream"
    assert FreestreamBoundary(name="freestream")

    assert IsothermalWall(Temperature=1).type == "IsothermalWall"
    assert IsothermalWall(Temperature="exp(x)")

    assert SubsonicOutflowPressure(staticPressureRatio=1).type == "SubsonicOutflowPressure"
    with pytest.raises(pd.ValidationError):
        SubsonicOutflowPressure(staticPressureRatio=-1)

    assert SubsonicOutflowMach(Mach=1).type == "SubsonicOutflowMach"
    with pytest.raises(pd.ValidationError):
        SubsonicOutflowMach(Mach=-1)

    assert (
        SubsonicInflow(totalPressureRatio=1, totalTemperatureRatio=1, rampSteps=10).type
        == "SubsonicInflow"
    )
    assert SlidingInterfaceBoundary().type == "SlidingInterface"
    assert WallFunction().type == "WallFunction"
    assert MassInflow(massFlowRate=1).type == "MassInflow"
    with pytest.raises(pd.ValidationError):
        MassInflow(massFlowRate=-1)
    assert MassOutflow(massFlowRate=1).type == "MassOutflow"
    with pytest.raises(pd.ValidationError):
        MassOutflow(massFlowRate=-1)


def test_actuator_disk():
    fpa = ForcePerArea(radius=[0, 1], thrust=[1, 1], circumferential=[1, 1])
    assert fpa
    ad = ActuatorDisk(center=(0, 0, 0), axis_thrust=(0, 0, 1), thickness=20, force_per_area=fpa)
    assert ad

    with pytest.raises(ValidationError):
        fpa = ForcePerArea(radius=[0, 1, 3], thrust=[1, 1], circumferential=[1, 1])

    to_file_from_file_test(ad)
    compare_to_ref(ad, "ref/case_params/actuator_disk/json.json")
    compare_to_ref(ad, "ref/case_params/actuator_disk/yaml.yaml")


def test_freesteam():
    fs = Freestream(Mach=1, temperature=300, density=1.22)
    assert fs
    with pytest.raises(ConfigError):
        print(fs.to_flow360_json())
    assert fs.to_flow360_json(mesh_unit_length=1)

    with pytest.raises(pd.ValidationError):
        fs = Freestream(Mach=-1, Temperature=100)

    fs = Freestream.from_speed(speed=(10, "m/s"))
    to_file_from_file_test(fs)
    assert fs

    fs = Freestream.from_speed(speed=10)
    assert fs

    with pytest.raises(ConfigError):
        print(fs.to_flow360_json())

    assert fs.to_flow360_json(mesh_unit_length=1)
    assert "speed" in json.loads(fs.json())
    assert "density" in json.loads(fs.json())
    assert "speed" not in json.loads(fs.to_flow360_json(mesh_unit_length=1))
    assert "density" not in json.loads(fs.to_flow360_json(mesh_unit_length=1))

    to_file_from_file_test(fs)


def test_mesh_sliding_interface():
    msi = MeshSlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
    )
    assert msi
    to_file_from_file_test(msi)

    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name="volume1",
        omega=1,
    )

    msi = MeshSlidingInterface.from_case_sliding_interface(si)
    assert msi


def test_sliding_interface():
    with pytest.raises(ConfigError):
        si = SlidingInterface(
            center=(0, 0, 0),
            axis=(0, 0, 1),
            stationary_patches=["patch1"],
            rotating_patches=["patch2"],
            volume_name="volume1",
        )

    # setting up both omega and rpm, or
    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name=1,
        omega=1,
        rpm=1,
    )

    assert si

    si = SlidingInterface.parse_raw(
        """
    {
        "stationaryPatches" : ["farField/rotationInterface"],
        "rotatingPatches" : ["innerRotating/rotationInterface"],
        "axisOfRotation" : [0,0,-1],
        "centerOfRotation" : [0,0,0],
        "omegaRadians" : 1.84691e-01,
        "volumeName" : ["innerRotating"]
    }
    """
    )

    assert si

    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name=["volume1", "volume2"],
        omega=1,
        rpm=1,
    )

    assert si
    to_file_from_file_test(si)

    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name=[0, 1],
        omegaRadians=1,
        rpm=1,
    )

    assert si

    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name="volume1",
        omega=(1, "rad/s"),
    )

    assert si
    assert si.json()

    to_file_from_file_test(si)
    compare_to_ref(si, "ref/case_params/sliding_interface/json.json")
    compare_to_ref(si, "ref/case_params/sliding_interface/yaml.yaml")

    with pytest.raises(ConfigError):
        print(si.to_flow360_json())

    assert "omega" in json.loads(si.json())
    assert "omegaRadians" not in json.loads(si.json())
    assert "omega" not in json.loads(si.to_flow360_json(mesh_unit_length=0.01, C_inf=1))
    assert "omegaRadians" in json.loads(si.to_flow360_json(mesh_unit_length=0.01, C_inf=1))
    assert json.loads(si.to_flow360_json(mesh_unit_length=0.01, C_inf=1))["omegaRadians"] == 0.01

    si = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name="volume1",
        omega=(1, "deg/s"),
    )

    assert si
    assert si.json()

    with pytest.raises(ConfigError):
        print(si.to_flow360_json())

    assert "omega" in json.loads(si.json())
    assert "omegaDegrees" not in json.loads(si.json())
    assert "omega" not in json.loads(si.to_flow360_json(mesh_unit_length=1, C_inf=1))
    assert "omegaDegrees" in json.loads(si.to_flow360_json(mesh_unit_length=1, C_inf=1))
    assert json.loads(si.to_flow360_json(mesh_unit_length=0.01, C_inf=1))["omegaDegrees"] == 0.01

    rpm = 100
    si_rpm = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name="volume1",
        rpm=rpm,
    )

    si_omega = SlidingInterface(
        center=(0, 0, 0),
        axis=(0, 0, 1),
        stationary_patches=["patch1"],
        rotating_patches=["patch2"],
        volume_name="volume1",
        omega=(rpm * 2 * math.pi / 60, "rad/s"),
    )

    assert si_rpm.to_flow360_json(mesh_unit_length=0.01, C_inf=1) == si_omega.to_flow360_json(
        mesh_unit_length=0.01, C_inf=1
    )

    params = Flow360Params(sliding_interfaces=[si], freestream=Freestream.from_speed(10))

    assert params.json()
    with pytest.raises(ConfigError):
        print(params.to_flow360_json())

    params = Flow360Params(
        geometry=Geometry(mesh_unit="mm"),
        freestream=Freestream.from_speed(10),
        sliding_interfaces=[si],
    )

    assert params.json()
    assert params.to_flow360_json()
    assertions.assertAlmostEqual(
        json.loads(params.to_flow360_json())["slidingInterfaces"][0]["omegaDegrees"], 2.938e-06
    )


def test_time_stepping():
    ts = TimeStepping()
    assert ts.json()
    assert ts.to_flow360_json()
    to_file_from_file_test(ts)

    with pytest.raises(pd.ValidationError):
        ts = TimeStepping(physical_steps=10, time_step_size=-0.01)

    with pytest.raises(pd.ValidationError):
        ts = TimeStepping(physical_steps=10, time_step_size=(-0.01, "s"))

    with pytest.raises(pd.ValidationError):
        ts = TimeStepping(physical_steps=10, time_step_size="infinity")

    ts = TimeStepping(time_step_size="inf")
    to_file_from_file_test(ts)

    ts = TimeStepping(physical_steps=10, time_step_size=(0.01, "s"))
    assert isinstance(ts.time_step_size, TimeStep)

    to_file_from_file_test(ts)

    assert ts.json()
    with pytest.raises(ConfigError):
        ts.to_flow360_json()

    assert ts.to_flow360_json(mesh_unit_length=0.2, C_inf=2)

    params = Flow360Params(
        geometry=Geometry(mesh_unit="mm"), freestream=Freestream.from_speed(10), time_stepping=ts
    )

    assertions.assertAlmostEqual(
        json.loads(params.to_flow360_json())["timeStepping"]["timeStepSize"], 0.1
    )
    to_file_from_file_test(ts)

    params = Flow360Params(
        geometry={"meshUnit": "mm"},
        freestream={"temperature": 1, "Mach": 1, "density": 1},
        time_stepping=ts,
    )
    exported_json = json.loads(params.to_flow360_json())
    assert "meshUnit" not in exported_json["geometry"]

    ts = TimeStepping.parse_obj({"maxPhysicalSteps": 3})
    assert ts.physical_steps == 3

    ts = TimeStepping.parse_obj({"physicalSteps": 2})
    assert ts.physical_steps == 2

    with pytest.raises(ValidationError):
        ts = TimeStepping.parse_obj({"maxPhysicalSteps": 3, "physical_steps": 2})

    with pytest.raises(ValidationError):
        ts = TimeStepping.parse_obj({"maxPhysicalSteps": 3, "physicalSteps": 2})


def test_time_stepping_cfl():
    cfl = fl.RampCFL(rampSteps=20, initial=10, final=100)
    assert cfl

    cfl = fl.AdaptiveCFL(min=0.1, max=2000, max_relative_change=1, convergence_limiting_factor=0.25)
    assert cfl


def test_navier_stokes():
    with pytest.raises(pd.ValidationError):
        ns = NavierStokesSolver(kappaMUSCL=-2)
    assert NavierStokesSolver(kappaMUSCL=-1)
    assert NavierStokesSolver(kappaMUSCL=1)
    with pytest.raises(pd.ValidationError):
        ns = NavierStokesSolver(kappaMUSCL=2)

    with pytest.raises(pd.ValidationError):
        ns = NavierStokesSolver(order_of_accuracy=0)

    assert NavierStokesSolver(order_of_accuracy=1)
    assert NavierStokesSolver(order_of_accuracy=2)

    with pytest.raises(pd.ValidationError):
        ns = NavierStokesSolver(order_of_accuracy=3)

    ns = NavierStokesSolver(
        absolute_tolerance=1e-10,
        kappaMUSCL=-1,
        relative_tolerance=0,
        CFL_multiplier=1,
        linear_iterations=30,
        update_jacobian_frequency=4,
        equation_eval_frequency=1,
        max_force_jac_update_physical_steps=1,
        order_of_accuracy=2,
        limit_velocity=True,
        limit_pressure_density=False,
        numerical_dissipation_factor=0.2,
    )
    p = Flow360Params(
        navier_stokes_solver=ns,
        freestream={"Mach": 1, "Temperature": 1},
    )
    to_file_from_file_test(p)


def test_turbulence_solver():
    ts = TurbulenceModelSolver(model_type=fl.turbulence.SA)
    ts = TurbulenceModelSolver(model_type=fl.turbulence.SST)
    ts = TurbulenceModelSolver(model_type=fl.turbulence.NONE)
    ts = TurbulenceModelSolver(model_type="SA")
    ts = TurbulenceModelSolver(model_type="SpalartAllmaras")
    ts = TurbulenceModelSolver(model_type="SST")
    ts = TurbulenceModelSolver(model_type="kOmegaSST")
    ts = TurbulenceModelSolver(model_type="None")

    with pytest.raises(pd.ValidationError):
        ts = TurbulenceModelSolver(model_type="OtherSolver")

    with pytest.raises(pd.ValidationError):
        ts = TurbulenceModelSolver(model_type="SA", grid_size_for_LES="other_option")

    ts = TurbulenceModelSolver(
        model_type=fl.turbulence.SA,
        absolute_tolerance=1e-10,
        relative_tolerance=0,
        linear_iterations=30,
        update_jacobian_frequency=4,
        equation_eval_frequency=1,
        max_force_jac_update_physical_steps=1,
        order_of_accuracy=2,
        DDES=True,
        grid_size_for_LES="maxEdgeLength",
        model_constants={"C_DES1": 0.85, "C_d1": 8.0},
    )
    to_file_from_file_test(ts)


def test_flow360meshparam():
    mp0 = Flow360MeshParams.parse_raw(
        """
    {
        "boundaries": {
            "noSlipWalls": [
                "fluid/fuselage",
                "fluid/leftWing",
                "fluid/rightWing"
            ]
        }
    }
    """
    )
    assert mp0
    to_file_from_file_test(mp0)

    mp1 = Flow360MeshParams.parse_raw(
        """
        {
        "boundaries": {
            "noSlipWalls": [
                1,
                2,
                3
            ]
        }
    }
        """
    )

    assert mp1
    to_file_from_file_test(mp1)

    mp2 = Flow360MeshParams(
        boundaries=MeshBoundary(
            no_slip_walls=["fluid/fuselage", "fluid/leftWing", "fluid/rightWing"]
        )
    )
    assert mp2
    assert mp0 == mp2
    to_file_from_file_test(mp2)


def test_flow360param():
    mesh = Flow360Params.parse_raw(
        """
        {
    "boundaries": {
        "fluid/fuselage": {
            "type": "NoSlipWall"
        },
        "fluid/leftWing": {
            "type": "NoSlipWall"
        },
        "fluid/rightWing": {
            "type": "NoSlipWall"
        },
        "fluid/farfield": {
            "type": "Freestream"
        }
    },
    "actuatorDisks": [
        {
            "center": [
                3.6,
                -5.08354845,
                0
            ],
            "axisThrust": [
                -0.96836405,
                -0.06052275,
                0.24209101
            ],
            "thickness": 0.42,
            "forcePerArea": {
                "radius": [],
                "thrust": [],
                "circumferential": []
            }
        },
        {
            "center": [
                3.6,
                5.08354845,
                0
            ],
            "axisThrust": [
                -0.96836405,
                0.06052275,
                0.24209101
            ],
            "thickness": 0.42,
            "forcePerArea": {
                "radius": [],
                "thrust": [],
                "circumferential": []
            }
        }
    ],
    "freestream": {"temperature": 1, "Mach": 0.5}
}
        """
    )

    assert mesh


def test_volume_zones():
    with pytest.raises(ConfigError):
        rf = ReferenceFrame(
            center=(0, 0, 0),
            axis=(0, 0, 1),
        )

    rf = ReferenceFrame(center=(0, 0, 0), axis=(0, 0, 1), omega_radians=1)

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

    with pytest.raises(pd.ValidationError):
        zone = HeatTransferVolumeZone(thermal_conductivity=-1)

    to_file_from_file_test(zones)

    zones = VolumeZones(
        zone1=FluidDynamicsVolumeZone(reference_frame=rf),
        zone2=HeatTransferVolumeZone(
            thermal_conductivity=1,
            heat_capacity=1,
            initial_condition=InitialConditionHeatTransfer(T_solid=100),
        ),
    )

    assert zones

    to_file_from_file_test(zones)

    zones = VolumeZones.parse_obj(
        {
            "fluid": {
                "modelType": "FluidDynamics",
            },
            "solid": {
                "modelType": "HeatTransfer",
                "thermalConductivity": 0.003,
                "volumetricHeatSource": "0.01",
            },
        }
    )

    assert zones


def test_heat_equation():
    he = HeatEquationSolver(
        equation_eval_frequency=10,
        linear_solver=LinearSolver(
            absolute_tolerance=1e-10,
            max_iterations=50,
        ),
    )
    assert he

    he = HeatEquationSolver.from_json("ref/case_params/heat_equation/ref_old.json")

    assert he

    compare_to_ref(he, "ref/case_params/heat_equation/ref.json", content_only=True)


def test_aeroacoustic_output():
    with pytest.raises(pd.ValidationError):
        output = AeroacousticOutput()

    output = AeroacousticOutput(observers=[(0, 0, 0), (0, 1, 1)])

    to_file_from_file_test(output)

    assert output

    output = AeroacousticOutput(observers=[])

    to_file_from_file_test(output)

    assert output

    with pytest.raises(pd.ValidationError):
        output = AeroacousticOutput(observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=0)

    output = AeroacousticOutput(
        observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=1, animation_frequency_offset=-2
    )

    assert output

    output = AeroacousticOutput(
        observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=1, animation_frequency_offset=2
    )

    assert output

    output = AeroacousticOutput(
        observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=1, animation_frequency_offset=0
    )

    assert output

    output = AeroacousticOutput(
        observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=-1, animation_frequency_offset=0
    )

    assert output

    to_file_from_file_test(output)

    with pytest.raises(pd.ValidationError):
        output = AeroacousticOutput(
            observers=[(0, 0, 0), (0, 1, 1)], animation_frequency=-2, animation_frequency_offset=0
        )

    output = AeroacousticOutput(
        observers=[(0, 0, 0), (0, 1, 1)],
        animation_frequency=1,
        animation_frequency_offset=-2,
        patch_type="solid",
    )

    assert output

    to_file_from_file_test(output)

    with pytest.raises(pd.ValidationError):
        output = AeroacousticOutput(
            observers=[(0, 0, 0), (0, 1, 1)],
            animation_frequency=1,
            animation_frequency_offset=-2,
            patch_type="other",
        )


def test_flow360param1():
    params = Flow360Params(freestream=Freestream.from_speed(10))
    assert params.time_stepping.max_pseudo_steps is None
    params.time_stepping = TimeStepping(physical_steps=100)
    assert params


def test_tuple_from_yaml():
    fs = Freestream("data/case_params/freestream/yaml.yaml")
    assert fs


def test_update_from_multiple_files():
    params = fl.Flow360Params(
        geometry=fl.Geometry("data/case_params/geometry.yaml"),
        boundaries=fl.Boundaries("data/case_params/boundaries.yaml"),
        freestream=fl.Freestream.from_speed((286, "m/s"), alpha=3.06),
        navier_stokes_solver=fl.NavierStokesSolver(linear_iterations=10),
    )

    outputs = fl.Flow360Params("data/case_params/outputs.yaml")
    params.append(outputs)

    assert params
    to_file_from_file_test(params)
    compare_to_ref(params, "ref/case_params/params.yaml")
    compare_to_ref(params, "ref/case_params/params.json", content_only=True)


def test_update_from_multiple_files_dont_overwrite():
    params = fl.Flow360Params(
        geometry=fl.Geometry("data/case_params/geometry.yaml"),
        boundaries=fl.Boundaries("data/case_params/boundaries.yaml"),
        freestream=fl.Freestream.from_speed((286, "m/s"), alpha=3.06),
        navier_stokes_solver=fl.NavierStokesSolver(linear_iterations=10),
    )

    outputs = fl.Flow360Params("data/case_params/outputs.yaml")
    outputs.geometry = fl.Geometry(ref_area=2)
    params.append(outputs)

    assert params.geometry.ref_area == 1.15315084119231


def test_update_from_multiple_files_overwrite():
    params = fl.Flow360Params(
        geometry=fl.Geometry("data/case_params/geometry.yaml"),
        boundaries=fl.Boundaries("data/case_params/boundaries.yaml"),
        freestream=fl.Freestream.from_speed((286, "m/s"), alpha=3.06),
        navier_stokes_solver=fl.NavierStokesSolver(linear_iterations=10),
    )

    outputs = fl.Flow360Params("data/case_params/outputs.yaml")
    outputs.geometry = fl.Geometry(ref_area=2)
    params.append(outputs, overwrite=True)

    assert params.geometry.ref_area == 2


def clear_formatting(message):
    # Remove color formatting escape codes
    ansi_escape = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")
    cleared = ansi_escape.sub("", message).replace("\n", "")
    cleared = re.sub(r" +", " ", cleared)
    return cleared


def test_depracated(capfd):
    ns = fl.NavierStokesSolver(tolerance=1e-8)
    captured = capfd.readouterr()
    expected = f'WARNING: "tolerance" is deprecated. Use "absolute_tolerance" OR "absoluteTolerance" instead'
    assert expected in clear_formatting(captured.out)

    ns = fl.TimeStepping(maxPhysicalSteps=10)
    captured = capfd.readouterr()
    expected = f'WARNING: "maxPhysicalSteps" is deprecated. Use "physical_steps" OR "physicalSteps" instead'
    assert expected in clear_formatting(captured.out)
