import unittest

import pydantic.v1 as pd
import pytest

import flow360.component.v1 as fl
from flow360.component.v1.boundaries import (
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
    WallFunction,
)
from flow360.component.v1.flow360_params import (
    Flow360Params,
    FreestreamFromMach,
    MeshBoundary,
    SteadyTimeStepping,
)
from flow360.component.v1.turbulence_quantities import TurbulenceQuantities
from tests.utils import compare_to_ref, to_file_from_file_test

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

    compare_to_ref(mesh_boundary, "../ref/flow360mesh/mesh_boundary/yaml.yaml")
    compare_to_ref(mesh_boundary, "../ref/flow360mesh/mesh_boundary/json.json")
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
    with fl.SI_unit_system:
        with pytest.raises(ValueError):
            param = Flow360Params(
                boundaries={
                    "fluid/fuselage": SteadyTimeStepping(),
                    "fluid/leftWing": NoSlipWall(),
                    "fluid/rightWing": NoSlipWall(),
                },
                freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )

        with pytest.raises(ValueError):
            param = Flow360Params.parse_raw(
                """
                {
                    "geometry": {
                        "momentCenter": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "momentLength": [
                            1.0,
                            1.0,
                            1.0
                        ],
                        "refArea": 0.5325
                    },
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
                    }.
                    "freestream": {
                        "modelType": "FromMach",
                        "Mach": 1,
                        "temperature": 1,
                        "mu_ref": 1
                    }
                }
                """
            )

        param = Flow360Params.parse_raw(
            """
            {
                "geometry": {
                    "momentCenter": [
                        0.0,
                        0.0,
                        0.0
                    ],
                    "momentLength": [
                        1.0,
                        1.0,
                        1.0
                    ],
                    "refArea": 0.5325
                },
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
                },
                "freestream": {
                    "modelType": "FromMach",
                    "Mach": 1,
                    "temperature": 1,
                    "mu_ref": 1
                }
            }
            """
        )

        assert param

        boundaries = fl.Boundaries(
            wing=NoSlipWall(), symmetry=SlipWall(), freestream=FreestreamBoundary()
        )

        assert boundaries

        with pytest.raises(ValueError):
            param = Flow360Params(
                boundaries={
                    "fluid/fuselage": "NoSlipWall",
                    "fluid/leftWing": NoSlipWall(),
                    "fluid/rightWing": NoSlipWall(),
                },
                freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
            )

        param = Flow360Params(
            boundaries={
                "fluid/fuselage": NoSlipWall(),
                "fluid/rightWing": NoSlipWall(),
                "fluid/leftWing": NoSlipWall(),
            },
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

        assert param

        param = Flow360Params(
            boundaries={
                "fluid/ fuselage": NoSlipWall(),
                "fluid/rightWing": NoSlipWall(),
                "fluid/leftWing": SolidIsothermalWall(temperature=1.0),
            },
            freestream=FreestreamFromMach(Mach=1, temperature=1, mu_ref=1),
        )

    compare_to_ref(param.boundaries, "../ref/case_params/boundaries/yaml.yaml")
    compare_to_ref(param.boundaries, "../ref/case_params/boundaries/json.json")
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
    with fl.flow360_unit_system:
        assert NoSlipWall(velocity=(0, 0, 0))
    with fl.flow360_unit_system:
        assert NoSlipWall(name="name", velocity=[0, 0, 0])
    assert NoSlipWall(velocity=("0", "0.1*x+exp(y)+z^2", "cos(0.2*x*pi)+sqrt(z^2+1)"))
    assert SlipWall().type == "SlipWall"
    assert FreestreamBoundary().type == "Freestream"
    assert FreestreamBoundary(name="freestream")

    assert IsothermalWall(Temperature=1).type == "IsothermalWall"
    assert IsothermalWall(Temperature="exp(x)")

    assert HeatFluxWall(heatFlux=-0.01).type == "HeatFluxWall"
    with fl.flow360_unit_system:
        assert HeatFluxWall(heatFlux="exp(x)", velocity=(0, 0, 0))

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
    assert MassInflow(massFlowRate=1, totalTemperatureRatio=1).type == "MassInflow"
    with pytest.raises(pd.ValidationError):
        MassInflow(massFlowRate=-1)
    with pytest.raises(pd.ValidationError):
        MassInflow(totalTemperatureRatio=-1)
    assert MassOutflow(massFlowRate=1).type == "MassOutflow"
    with pytest.raises(pd.ValidationError):
        MassOutflow(massFlowRate=-1)

    # Test the turbulence quantities on the boundaries
    bc = SubsonicOutflowMach(
        name="SomeBC",
        Mach=0.2,
        turbulence_quantities=TurbulenceQuantities(turbulent_intensity=0.2, viscosity_ratio=10),
    )

    assert bc.turbulence_quantities.turbulent_intensity == 0.2
    assert bc.turbulence_quantities.turbulent_viscosity_ratio == 10

    bc = SubsonicOutflowPressure(
        name="SomeBC",
        static_pressure_ratio=0.2,
        turbulence_quantities=TurbulenceQuantities(),
    )

    assert bc.turbulence_quantities is None

    bc = FreestreamBoundary(
        name="SomeBC",
        turbulence_quantities=TurbulenceQuantities(viscosity_ratio=14),
    )

    assert bc.turbulence_quantities.turbulent_viscosity_ratio == 14

    bc = SubsonicOutflowPressure(
        name="SomeBC",
        static_pressure_ratio=0.2,
        turbulence_quantities=TurbulenceQuantities(
            viscosity_ratio=124, turbulent_kinetic_energy=0.2
        ),
    )

    assert bc.turbulence_quantities.turbulent_viscosity_ratio == 124
    assert bc.turbulence_quantities.turbulent_kinetic_energy == 0.2

    bc = SubsonicOutflowMach(
        name="SomeBC",
        Mach=0.2,
        turbulence_quantities=TurbulenceQuantities(
            specific_dissipation_rate=124, viscosity_ratio=0.2
        ),
    )

    assert bc.turbulence_quantities.specific_dissipation_rate == 124
    assert bc.turbulence_quantities.turbulent_viscosity_ratio == 0.2

    bc = SubsonicInflow(
        name="SomeBC",
        total_pressure_ratio=0.2,
        total_temperature_ratio=0.43,
        turbulence_quantities=TurbulenceQuantities(viscosity_ratio=124, turbulent_length_scale=1.2),
    )

    assert bc.turbulence_quantities.turbulent_viscosity_ratio == 124
    assert bc.turbulence_quantities.turbulent_length_scale == 1.2

    bc = MassInflow(
        name="SomeBC",
        mass_flow_rate=0.2,
        total_temperature_ratio=0.43,
        turbulence_quantities=TurbulenceQuantities(modified_viscosity_ratio=1.2),
    )

    assert bc.turbulence_quantities.modified_turbulent_viscosity_ratio == 1.2

    bc = MassInflow(
        name="SomeBC",
        mass_flow_rate=0.2,
        total_temperature_ratio=0.43,
        turbulence_quantities=TurbulenceQuantities(turbulent_intensity=0.2),
    )

    assert bc.turbulence_quantities.turbulent_intensity == 0.2

    bc = MassInflow(
        name="SomeBC",
        mass_flow_rate=0.2,
        total_temperature_ratio=0.43,
        turbulence_quantities=TurbulenceQuantities(turbulent_kinetic_energy=12.2),
    )

    assert bc.turbulence_quantities.turbulent_kinetic_energy == 12.2

    bc = MassInflow(
        name="SomeBC",
        mass_flow_rate=0.2,
        total_temperature_ratio=0.43,
        turbulence_quantities=TurbulenceQuantities(turbulent_length_scale=1.23),
    )

    assert bc.turbulence_quantities.turbulent_length_scale == 1.23

    bc = MassOutflow(
        name="SomeBC",
        mass_flow_rate=0.2,
        turbulence_quantities=TurbulenceQuantities(modified_viscosity=1.2),
    )

    assert bc.turbulence_quantities.modified_turbulent_viscosity == 1.2

    bc = MassOutflow(
        name="SomeBC",
        mass_flow_rate=0.2,
        turbulence_quantities=TurbulenceQuantities(
            turbulent_intensity=0.88, specific_dissipation_rate=100
        ),
    )

    assert bc.turbulence_quantities.turbulent_intensity == 0.88
    assert bc.turbulence_quantities.specific_dissipation_rate == 100

    bc = MassOutflow(
        name="SomeBC",
        mass_flow_rate=0.2,
        turbulence_quantities=TurbulenceQuantities(
            turbulent_intensity=0.88, turbulent_length_scale=10
        ),
    )

    assert bc.turbulence_quantities.turbulent_intensity == 0.88
    assert bc.turbulence_quantities.turbulent_length_scale == 10

    bc = MassOutflow(
        name="SomeBC",
        mass_flow_rate=0.2,
        turbulence_quantities=TurbulenceQuantities(
            turbulent_kinetic_energy=0.88, specific_dissipation_rate=10
        ),
    )

    assert bc.turbulence_quantities.turbulent_kinetic_energy == 0.88
    assert bc.turbulence_quantities.specific_dissipation_rate == 10

    bc = MassOutflow(
        name="SomeBC",
        mass_flow_rate=0.2,
        turbulence_quantities=TurbulenceQuantities(
            turbulent_kinetic_energy=0.88, specific_dissipation_rate=10
        ),
    )

    assert bc.turbulence_quantities.turbulent_kinetic_energy == 0.88
    assert bc.turbulence_quantities.specific_dissipation_rate == 10

    bc = MassOutflow(
        name="SomeBC",
        mass_flow_rate=0.2,
        turbulence_quantities=TurbulenceQuantities(
            turbulent_kinetic_energy=0.88, turbulent_length_scale=10
        ),
    )

    assert bc.turbulence_quantities.turbulent_kinetic_energy == 0.88
    assert bc.turbulence_quantities.turbulent_length_scale == 10

    bc = MassOutflow(
        name="SomeBC",
        mass_flow_rate=0.2,
        turbulence_quantities=TurbulenceQuantities(
            specific_dissipation_rate=0.88, turbulent_length_scale=10
        ),
    )

    assert bc.turbulence_quantities.specific_dissipation_rate == 0.88
    assert bc.turbulence_quantities.turbulent_length_scale == 10

    with pytest.raises(ValueError):
        MassOutflow(
            name="SomeBC",
            mass_flow_rate=0.2,
            turbulence_quantities=TurbulenceQuantities(
                specific_dissipation_rate=0.88, modified_viscosity=10
            ),
        )

    with pytest.raises(ValueError):
        MassOutflow(
            name="SomeBC",
            mass_flow_rate=0.2,
            turbulence_quantities=TurbulenceQuantities(specific_dissipation_rate=0.88),
        )

        assert bc.turbulence_quantities.turbulent_intensity == 0.88
        assert bc.turbulence_quantities.turbulent_length_scale == 10

        bc = MassOutflow(
            name="SomeBC",
            mass_flow_rate=0.2,
            turbulence_quantities=TurbulenceQuantities(
                turbulent_kinetic_energy=0.88, specific_dissipation_rate=10
            ),
        )

        assert bc.turbulence_quantities.turbulent_kinetic_energy == 0.88
        assert bc.turbulence_quantities.specific_dissipation_rate == 10

        bc = MassOutflow(
            name="SomeBC",
            mass_flow_rate=0.2,
            turbulence_quantities=TurbulenceQuantities(
                turbulent_kinetic_energy=0.88, specific_dissipation_rate=10
            ),
        )

        assert bc.turbulence_quantities.turbulent_kinetic_energy == 0.88
        assert bc.turbulence_quantities.specific_dissipation_rate == 10

        bc = MassOutflow(
            name="SomeBC",
            mass_flow_rate=0.2,
            turbulence_quantities=TurbulenceQuantities(
                turbulent_kinetic_energy=0.88, turbulent_length_scale=10
            ),
        )

        assert bc.turbulence_quantities.turbulent_kinetic_energy == 0.88
        assert bc.turbulence_quantities.turbulent_length_scale == 10

        bc = MassOutflow(
            name="SomeBC",
            mass_flow_rate=0.2,
            turbulence_quantities=TurbulenceQuantities(
                specific_dissipation_rate=0.88, turbulent_length_scale=10
            ),
        )

        assert bc.turbulence_quantities.specific_dissipation_rate == 0.88
        assert bc.turbulence_quantities.turbulent_length_scale == 10

        with pytest.raises(ValueError):
            MassOutflow(
                name="SomeBC",
                mass_flow_rate=0.2,
                turbulence_quantities=TurbulenceQuantities(
                    specific_dissipation_rate=0.88, modified_viscosity=10
                ),
            )

        with pytest.raises(ValueError):
            MassOutflow(
                name="SomeBC",
                mass_flow_rate=0.2,
                turbulence_quantities=TurbulenceQuantities(specific_dissipation_rate=0.88),
            )

        with fl.SI_unit_system:
            params = fl.Flow360Params(
                fluid_properties=fl.air,
                geometry=fl.Geometry(),
                boundaries={
                    "FreestreamBC": fl.FreestreamBoundary(),
                    "PressureOutflowBC": fl.PressureOutflow(
                        turbulence_quantities=fl.TurbulenceQuantities(modified_viscosity_ratio=0.5)
                    ),
                },
                freestream=fl.FreestreamFromVelocity(
                    velocity=286,
                    alpha=3.06,
                    turbulence_quantities=fl.TurbulenceQuantities(viscosity_ratio=0.2),
                ),
                navier_stokes_solver=fl.NavierStokesSolver(absolute_tolerance=1e-10),
                turbulence_model_solver=fl.SpalartAllmaras(),
            )

            params.to_solver()
            assert (
                params.boundaries["FreestreamBC"].turbulence_quantities.turbulent_viscosity_ratio
                == 0.2
            )
            assert (
                params.boundaries[
                    "PressureOutflowBC"
                ].turbulence_quantities.modified_turbulent_viscosity_ratio
                == 0.5
            )
            assert not hasattr(
                params.boundaries["PressureOutflowBC"].turbulence_quantities,
                "turbulent_viscosity_ratio",
            )


def test_boundary_expression():
    with fl.SI_unit_system:
        params = fl.Flow360Params(
            fluid_properties=fl.air,
            geometry=fl.Geometry(mesh_unit=1),
            boundaries={
                "NSW": fl.NoSlipWall(velocity=("x*y^z", "1.2/45", "y^0.5-123")),
                "FS": fl.FreestreamBoundary(velocity=("x*y^z", "1.2/45", "y^0.5-123")),
                "ISW": fl.IsothermalWall(
                    velocity=("x*y^z", "1.2/45", "y^0.5-123"), temperature="1.23*x^2.34/2"
                ),
                "HFW": fl.HeatFluxWall(velocity=("x*y^z", "1.2/45", "y^0.5-123"), heat_flux=1.234),
                "VIF": fl.VelocityInflow(velocity=("x*y^z", "1.2/45", "y^0.5-123")),
            },
            freestream=fl.FreestreamFromVelocity(
                velocity=123,
                alpha=1,
            ),
            navier_stokes_solver=fl.NavierStokesSolver(absolute_tolerance=1e-10),
            turbulence_model_solver=fl.SpalartAllmaras(),
        )
    solver_params = params.to_solver()
    for bc_name in ["NSW", "FS", "ISW", "HFW", "VIF"]:
        assert solver_params.boundaries[bc_name].velocity == (
            "x*powf(y, z)",
            "1.2/45",
            "powf(y, 0.5)-123",
        )
    assert solver_params.boundaries["ISW"].temperature == "1.23*powf(x, 2.34)/2"
    assert solver_params.boundaries["HFW"].heat_flux == "1.234"
