import json
import re
import unittest

import pydantic.v1 as pd
import pytest

import flow360.v1 as fl
from flow360.component.v1 import units as u
from flow360.component.v1.flow360_params import (
    Flow360MeshParams,
    Flow360Params,
    FreestreamFromVelocity,
    MeshBoundary,
    PorousMediumBox,
)
from flow360.component.v1.time_stepping import UnsteadyTimeStepping
from flow360.exceptions import Flow360RuntimeError

from ..utils import compare_to_ref, to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


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
    with fl.SI_unit_system:
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
        "freestream": {"modelType": "FromMach", "temperature": 1, "Mach": 0.5, "mu_ref": 1},
        "geometry":{"momentCenter": [
            0.0,
            0.0,
            0.0
        ],
        "momentLength": [
            1.0,
            1.0,
            1.0
        ],
        "refArea": 0.1}
    }
            """
        )

        assert mesh


def test_flow360param1():
    with fl.SI_unit_system:
        params = Flow360Params(
            freestream=FreestreamFromVelocity(velocity=10 * u.m / u.s),
            boundaries={},
        )
        assert params.time_stepping.max_pseudo_steps == 2000
        params.time_stepping = UnsteadyTimeStepping(physical_steps=100, time_step_size=2 * u.s)
        assert params


def test_tuple_from_yaml():
    fs = FreestreamFromVelocity("data/case_params/freestream/yaml.yaml")
    assert fs


@pytest.mark.usefixtures("array_equality_override")
def test_update_from_multiple_files():
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry("data/case_params/geometry.yaml"),
            boundaries=fl.Boundaries("data/case_params/boundaries.yaml"),
            freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
            navier_stokes_solver=fl.NavierStokesSolver(absolute_tolerance=1e-10),
        )

        outputs = fl.Flow360Params.construct("data/case_params/outputs.yaml")
        params.append(outputs, overwrite=True)

    assert params

    to_file_from_file_test(params)


def test_update_from_multiple_files_dont_overwrite():
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry("data/case_params/geometry.yaml"),
            boundaries=fl.Boundaries("data/case_params/boundaries.yaml"),
            freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
            navier_stokes_solver=fl.NavierStokesSolver(absolute_tolerance=1e-10),
        )

    outputs = fl.Flow360Params.construct("data/case_params/outputs.yaml")
    params.append(outputs)

    assert params.geometry.ref_area == 1.15315084119231


def test_update_from_multiple_files_overwrite():
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry("data/case_params/geometry.yaml"),
            boundaries=fl.Boundaries("data/case_params/boundaries.yaml"),
            freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
            navier_stokes_solver=fl.NavierStokesSolver(absolute_tolerance=1e-10),
        )

        outputs = fl.Flow360Params.construct("data/case_params/outputs.yaml")
        params.append(outputs, overwrite=True)

    assert params.geometry.ref_area == 2 * u.m**2


def clear_formatting(message):
    # Remove color formatting escape codes
    ansi_escape = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")
    cleared = ansi_escape.sub("", message).replace("\n", "")
    cleared = re.sub(r" +", " ", cleared)
    return cleared


def test_depracated(capfd):
    fl.NavierStokesSolver(tolerance=1e-8)
    captured = capfd.readouterr()
    expected = 'WARNING: "tolerance" is deprecated. Use "absolute_tolerance" OR "absoluteTolerance" instead'
    assert expected in clear_formatting(captured.out)

    fl.UnsteadyTimeStepping(maxPhysicalSteps=10, time_step_size=1.3 * u.s)
    captured = capfd.readouterr()
    expected = (
        'WARNING: "maxPhysicalSteps" is deprecated. Use "physical_steps" OR "physicalSteps" instead'
    )
    assert expected in clear_formatting(captured.out)


@pytest.mark.usefixtures("array_equality_override")
def test_params_with_units():
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry(
                ref_area=1.0 * u.flow360_area_unit,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            freestream=fl.FreestreamFromVelocity(velocity=286, alpha=3.06),
            time_stepping=fl.UnsteadyTimeStepping(
                max_pseudo_steps=500,
                CFL=fl.AdaptiveCFL(),
                time_step_size=1.2 * u.s,
                physical_steps=20,
            ),
            boundaries={
                "1": fl.NoSlipWall(name="wing", velocity=(1, 2, 3) * u.km / u.hr),
                "2": fl.SlipWall(name="symmetry"),
                "3": fl.FreestreamBoundary(name="freestream"),
            },
            fluid_properties=fl.air,
            volume_zones={
                "zone1": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrame(
                        center=(0, 0, 0), axis=(1, 0, 0), omega=10 * u.rpm
                    )
                ),
                "zone2": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrame(
                        center=(0, 0, 0), axis=(1, 0, 0), omega=10 * 2 * fl.pi / 60
                    )
                ),
                "zone3": fl.FluidDynamicsVolumeZone(
                    reference_frame=fl.ReferenceFrame(
                        center=(0, 0, 0), axis=(1, 0, 0), omega=10 * 360 / 60 * u.deg / u.s
                    )
                ),
            },
        )

    to_file_from_file_test(params)

    params_solver = params.to_solver()

    to_file_from_file_test(params_solver)

    params_as_json = params_solver.flow360_json()

    with open("ref/case_params/params_units_solver.json") as fh:
        a = json.load(fh)
    b = json.loads(params_as_json)
    assert sorted(a.items()) == sorted(b.items())


def test_params_with_units_consistency():
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry(
                ref_area=1,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            fluid_properties=fl.air,
            freestream=fl.FreestreamFromVelocity(velocity=286),
            time_stepping=fl.UnsteadyTimeStepping(
                max_pseudo_steps=500,
                CFL=fl.AdaptiveCFL(),
                time_step_size=1.2 * u.s,
                physical_steps=20,
            ),
            boundaries={},
        )

        with pytest.raises(ValueError):
            params.unit_system = fl.CGS_unit_system

    with fl.CGS_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry(
                ref_area=1,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            fluid_properties=fl.air,
            freestream=fl.FreestreamFromVelocity(velocity=286),
            time_stepping=fl.UnsteadyTimeStepping(
                max_pseudo_steps=500,
                CFL=fl.AdaptiveCFL(),
                time_step_size=1.2 * u.s,
                physical_steps=10,
            ),
            boundaries={},
        )

    params_as_json = params.json()

    with fl.UnitSystem(base_system=u.BaseSystemType.CGS, length=2.0 * u.cm):
        with pytest.raises(Flow360RuntimeError):
            params_reimport = fl.Flow360Params(**json.loads(params_as_json))

    # should NOT raise RuntimeError error from inconsistent unit systems because systems are consistent
    with fl.CGS_unit_system:
        params_reimport = fl.Flow360Params(**json.loads(params_as_json))

    with fl.SI_unit_system:
        with pytest.raises(Flow360RuntimeError):
            params_copy = params_reimport.copy()

    # should NOT raise RuntimeError error from inconsistent unit systems because systems are consistent
    with fl.CGS_unit_system:
        params_copy = params_reimport.copy()

    # should raise RuntimeError error from no context
    with pytest.raises(Flow360RuntimeError):
        params = fl.Flow360Params(
            geometry=fl.Geometry(
                ref_area=u.m**2,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            freestream=fl.FreestreamFromVelocity(velocity=286 * u.m / u.s),
            time_stepping=fl.UnsteadyTimeStepping(
                max_pseudo_steps=500,
                CFL=fl.AdaptiveCFL(),
                time_step_size=1.2 * u.s,
                physical_steps=10,
            ),
            boundaries={},
        )

    # should raise RuntimeError error from using context on file import
    with pytest.raises(Flow360RuntimeError):
        with fl.CGS_unit_system:
            fl.Flow360Params("ref/case_params/params_units.json")

    with fl.SI_unit_system:
        with pytest.raises(Flow360RuntimeError):
            params_copy.to_solver()

    # should NOT raise RuntimeError error from inconsistent unit systems because systems are consistent
    with fl.CGS_unit_system:
        params_copy.to_solver()

    # should NOT raise RuntimeError error from inconsistent unit systems because systems NO system
    params_copy.to_solver()

    with fl.SI_unit_system:
        with pytest.raises(Flow360RuntimeError):
            params_copy.flow360_json()

    # should NOT raise RuntimeError error from inconsistent unit systems because systems are consistent
    with fl.CGS_unit_system:
        params_copy.flow360_json()

    # should NOT raise RuntimeError error from inconsistent unit systems because systems NO system
    params_copy.flow360_json()


@pytest.mark.usefixtures("array_equality_override")
def test_params_with_units_conversion():
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry(
                ref_area=1.0 * u.flow360_area_unit,
                moment_length=(1.47602, 0.801672958512342, 1.47602) * u.inch,
                moment_center=(1, 2, 3) * u.flow360_length_unit,
                mesh_unit=u.mm,
            ),
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

    to_file_from_file_test(params)
    params = params.to_solver()
    to_file_from_file_test(params)


@pytest.mark.usefixtures("array_equality_override")
def test_params_with_solver_units():
    with fl.flow360_unit_system:
        params = fl.Flow360Params(
            geometry=fl.Geometry(
                ref_area=1.0,
                moment_length=(1.47602, 0.801672958512342, 1.47602),
                moment_center=[1, 2, 3],
            ),
            boundaries={},
            freestream=fl.FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

    to_file_from_file_test(params)
    params = params.to_solver()
    to_file_from_file_test(params)


def test_params_temperature_consistency():
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            freestream=fl.FreestreamFromMach(Mach=0.95, mu_ref=0.2, temperature=288.15 * u.K),
            fluid_properties=fl.AirDensityTemperature(
                temperature=288.15 * u.K, density=1.225 * u.kg / u.m**3
            ),
            boundaries={},
        )

    assert params

    with pytest.raises(pd.ValidationError):
        with fl.SI_unit_system:
            params = fl.Flow360Params(
                freestream=fl.FreestreamFromMach(mach=0.95, mu_ref=0.2, temperature=288.15 * u.K),
                fluid_properties=fl.AirDensityTemperature(
                    temperature=300.15 * u.K, density=1.225 * u.kg / u.m**3
                ),
            )

    with pytest.raises(pd.ValidationError):
        (fl.AirDensityTemperature(temperature=-2 * u.K, density=1.225 * u.kg / u.m**3),)

    with pytest.raises(pd.ValidationError):
        (fl.AirDensityTemperature(temperature=-288.15 * u.degC, density=1.225 * u.kg / u.m**3),)

    with pytest.raises(pd.ValidationError):
        (fl.AirPressureTemperature(temperature=-500.00 * u.degF, pressure=1.225 * u.N / u.m**2),)


def test_flow360_will_export():
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            freestream=fl.FreestreamFromMach(Mach=0.95, mu_ref=0.2, temperature=288.15 * u.K),
            fluid_properties=fl.AirDensityTemperature(
                temperature=288.15 * u.K, density=1.225 * u.kg / u.m**3
            ),
            boundaries={},
            porous_media=[
                PorousMediumBox(
                    darcy_coefficient=[1, 1, 1],
                    forchheimer_coefficient=[1, 1, 1],
                    volumetric_heat_source=0,
                    zone_type="box",
                    center=[1, 2, 3],
                    lengths=[3, 4, 5],
                    axes=[[0, 1, 0], [1, 0, 0]],
                    windowing_lengths=[0.5, 0.5, 0.5],
                )
            ],
        )

    params.set_will_export_to_flow360(True)
    assert params.freestream._will_export_to_flow360 is True
    assert params.porous_media[0]._will_export_to_flow360 is True
