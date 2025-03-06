import re
import unittest

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.migration.extra_operating_condition import (
    operating_condition_from_mach_muref,
)
from flow360.component.simulation.models.material import SolidMaterial
from flow360.component.simulation.models.surface_models import (
    HeatFlux,
    Inflow,
    MassFlowRate,
    SlipWall,
    TotalPressure,
    Wall,
)
from flow360.component.simulation.models.turbulence_quantities import (
    TurbulenceQuantities,
)
from flow360.component.simulation.models.volume_models import (
    AngularVelocity,
    Fluid,
    HeatEquationInitialCondition,
    PorousMedium,
    Rotation,
    Solid,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
    ThermalState,
    operating_condition_from_mach_reynolds,
)
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    GenericVolume,
    ReferenceGeometry,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Unsteady
from flow360.component.simulation.unit_system import (
    CGS_unit_system,
    LengthType,
    SI_unit_system,
)
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture()
def get_the_param():
    my_wall_surface = Surface(name="my_wall")
    my_slip_wall_surface = Surface(name="my_slip_wall")
    my_inflow1 = Surface(name="my_inflow1")
    my_inflow2 = Surface(name="my_inflow2")
    with CGS_unit_system:
        my_box = Box.from_principal_axes(
            name="my_box",
            center=(1.2, 2.3, 3.4) * u.m,
            size=(1.0, 2.0, 3.0) * u.m,
            axes=((3, 4, 0), (0, 0, 1)),
        )
        my_cylinder_1 = Cylinder(
            name="my_cylinder-1",
            axis=(5, 0, 0),
            center=(1.2, 2.3, 3.4) * u.m,
            height=3.0 * u.m,
            inner_radius=3.0 * u.m,
            outer_radius=5.0 * u.m,
        )
        my_solid_zone = GenericVolume(
            name="my_cylinder-2",
        )
        param = SimulationParams(
            meshing=MeshingParams(
                refinement_factor=1.0,
                gap_treatment_strength=0.5,
                defaults=MeshingDefaults(surface_edge_growth_rate=1.5),
                refinements=[UniformRefinement(entities=[my_box], spacing=0.1 * u.m)],
            ),
            reference_geometry=ReferenceGeometry(
                moment_center=(1, 2, 3), moment_length=1.0 * u.m, area=1.0 * u.cm**2
            ),
            operating_condition=AerospaceCondition.from_mach(
                mach=0.8,
                alpha=30 * u.deg,
                beta=20 * u.deg,
                thermal_state=ThermalState(temperature=300 * u.K, density=1 * u.g / u.cm**3),
                reference_mach=0.5,
            ),
            models=[
                Fluid(),
                Wall(
                    entities=[my_wall_surface],
                    use_wall_function=True,
                    velocity=(1.0, 1.2, 2.4) * u.ft / u.s,
                    heat_spec=HeatFlux(1.0 * u.W / u.m**2),
                ),
                SlipWall(entities=[my_slip_wall_surface]),
                Rotation(volumes=[my_cylinder_1], spec=AngularVelocity(0.45 * u.rad / u.s)),
                PorousMedium(
                    volumes=[my_box],
                    darcy_coefficient=(0.1, 2, 1.0) / u.cm / u.m,
                    forchheimer_coefficient=(0.1, 2, 1.0) / u.ft,
                    volumetric_heat_source=123 * u.lb / u.s**3 / u.ft,
                ),
                Solid(
                    volumes=[my_solid_zone],
                    material=SolidMaterial(
                        name="abc",
                        thermal_conductivity=1.0 * u.W / u.m / u.K,
                        specific_heat_capacity=1.0 * u.J / u.kg / u.K,
                        density=1.0 * u.kg / u.m**3,
                    ),
                    initial_condition=HeatEquationInitialCondition(temperature="1"),
                ),
                Inflow(
                    surfaces=[my_inflow1],
                    total_temperature=300 * u.K,
                    spec=TotalPressure(value=123 * u.Pa),
                    turbulence_quantities=TurbulenceQuantities(
                        turbulent_kinetic_energy=123, specific_dissipation_rate=1e3
                    ),
                ),
                Inflow(
                    surfaces=[my_inflow2],
                    total_temperature=300 * u.K,
                    spec=MassFlowRate(value=123 * u.lb / u.s),
                ),
            ],
            time_stepping=Unsteady(step_size=2 * 0.2 * u.s, steps=123),
            user_defined_dynamics=[
                UserDefinedDynamic(
                    name="fake",
                    input_vars=["fake"],
                    constants={"ff": 123},
                    state_vars_initial_value=["fake"],
                    update_law=["fake"],
                )
            ],
        )
        return param


@pytest.mark.usefixtures("array_equality_override")
def test_simulation_params_serialization(get_the_param):
    to_file_from_file_test(get_the_param)


@pytest.mark.usefixtures("array_equality_override")
def test_simulation_params_unit_conversion(get_the_param):
    converted = get_the_param._preprocess(mesh_unit=10 * u.m)

    # pylint: disable=fixme
    # TODO: Please perform hand calculation and update the following assertions
    # LengthType
    assertions.assertAlmostEqual(converted.reference_geometry.moment_length.value, 0.1)
    # AngleType
    assertions.assertAlmostEqual(converted.operating_condition.alpha.value, 0.5235987755982988)
    # TimeType
    assertions.assertAlmostEqual(converted.time_stepping.step_size.value, 13.8888282)
    # AbsoluteTemperatureType
    assertions.assertAlmostEqual(
        converted.models[0].material.dynamic_viscosity.effective_temperature.value, 0.368
    )
    # VelocityType
    assertions.assertAlmostEqual(converted.operating_condition.velocity_magnitude.value, 0.8)
    # AreaType
    assertions.assertAlmostEqual(converted.reference_geometry.area.value, 1e-6)
    # PressureType
    assertions.assertAlmostEqual(converted.models[6].spec.value.value, 1.0454827495346328e-06)
    # ViscosityType
    assertions.assertAlmostEqual(
        converted.models[0].material.dynamic_viscosity.reference_viscosity.value,
        1.0005830903790088e-11,
    )
    # AngularVelocityType
    assertions.assertAlmostEqual(converted.models[3].spec.value.value, 0.01296006)
    # HeatFluxType
    assertions.assertAlmostEqual(converted.models[1].heat_spec.value.value, 2.47809322e-11)
    # HeatSourceType
    assertions.assertAlmostEqual(
        converted.models[4].volumetric_heat_source.value, 4.536005048050727e-08
    )
    # HeatSourceType
    assertions.assertAlmostEqual(
        converted.models[4].volumetric_heat_source.value, 4.536005048050727e-08
    )
    # HeatCapacityType
    assertions.assertAlmostEqual(
        converted.models[5].material.specific_heat_capacity.value, 0.00248834
    )
    # ThermalConductivityType
    assertions.assertAlmostEqual(
        converted.models[5].material.thermal_conductivity.value, 7.434279666747016e-10
    )
    # InverseAreaType
    assertions.assertAlmostEqual(converted.models[4].darcy_coefficient.value[0], 1000.0)
    # InverseLengthType
    assertions.assertAlmostEqual(
        converted.models[4].forchheimer_coefficient.value[0], 3.280839895013123
    )
    # MassFlowRateType
    assertions.assertAlmostEqual(converted.models[7].spec.value.value, 1.6265848836734695e-06)

    # SpecificEnergyType
    assertions.assertAlmostEqual(
        converted.models[6].turbulence_quantities.turbulent_kinetic_energy.value,
        1.0454827495346325e-07,
    )

    # FrequencyType
    assertions.assertAlmostEqual(
        converted.models[6].turbulence_quantities.specific_dissipation_rate.value,
        28.80012584,
    )


def test_standard_atmosphere():
    # ref values from here: https://aerospaceweb.org/design/scripts/atmosphere/
    # alt, temp_offset, temp, density, pressure, viscosity
    ref_data = [
        (-1000, 0, 294.651, 1.347, 1.1393e5, 0.000018206),
        (0, 0, 288.15, 1.225, 101325, 0.000017894),
        (999, 0, 281.6575, 1.1118, 89887, 0.000017579),
        (1000, 0, 281.651, 1.11164, 89876, 0.000017579),
        (10000, 0, 223.2521, 0.41351, 26500, 0.000014577),
        (15000, 0, 216.65, 0.19476, 12112, 0.000014216),
        (20000, 0, 216.65, 0.088910, 5529.3, 0.000014216),
        (30000, 0, 226.5091, 0.018410, 1197.0, 0.000014753),
        (40000, 0, 250.3496, 0.0039957, 287.14, 0.000016009),
        (70000, 0, 219.5848, 0.000082829, 5.2209, 0.000014377),
        (0, -10, 278.15, 1.2690, 101325, 0.000017407),
        (1000, -9, 272.651, 1.1484, 89876, 0.000017136),
    ]

    for alt, temp_offset, temp, density, pressure, viscosity in ref_data:
        atm = ThermalState.from_standard_atmosphere(
            altitude=alt * u.m, temperature_offset=temp_offset * u.K
        )

        assert atm.temperature == pytest.approx(temp, rel=1e-6)
        assert atm.density == pytest.approx(density, rel=1e-4)
        assert atm.pressure == pytest.approx(pressure, rel=1e-4)
        assert atm.dynamic_viscosity == pytest.approx(viscosity, rel=1e-4)

    for alt, temp_offset, temp, density, pressure, viscosity in ref_data:
        delta_temp_in_F = (temp_offset * u.K).in_units("delta_degF")
        atm = ThermalState.from_standard_atmosphere(
            altitude=alt * u.m, temperature_offset=delta_temp_in_F
        )

        assert atm.temperature == pytest.approx(temp, rel=1e-6)
        assert atm.density == pytest.approx(density, rel=1e-4)
        assert atm.pressure == pytest.approx(pressure, rel=1e-4)
        assert atm.dynamic_viscosity == pytest.approx(viscosity, rel=1e-4)


def test_subsequent_param_with_different_unit_system():
    with SI_unit_system:
        param_SI = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=0.2)
            )
        )
    with CGS_unit_system:
        param_CGS = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(boundary_layer_first_layer_thickness=0.3)
            )
        )
    assert param_SI.unit_system.name == "SI"
    assert param_SI.meshing.defaults.boundary_layer_first_layer_thickness == 0.2 * u.m
    assert param_CGS.unit_system.name == "CGS"
    assert param_CGS.meshing.defaults.boundary_layer_first_layer_thickness == 0.3 * u.cm


def test_mach_reynolds_op_cond():

    condition = operating_condition_from_mach_reynolds(
        mach=0.2,
        reynolds=5e6,
        temperature=288.15 * u.K,
        alpha=2.0 * u.deg,
        beta=0.0 * u.deg,
        project_length_unit=u.m,
    )
    assertions.assertAlmostEqual(condition.thermal_state.dynamic_viscosity.value, 1.78929763e-5)
    assertions.assertAlmostEqual(condition.thermal_state.density.value, 1.31452332)

    with pytest.raises(ValueError, match="Input should be greater than 0"):
        condition = operating_condition_from_mach_reynolds(
            mach=0.2,
            reynolds=0,
            temperature=288.15 * u.K,
            project_length_unit=u.m,
        )


def test_mach_muref_op_cond():

    condition = operating_condition_from_mach_muref(
        mach=0.2,
        mu_ref=4e-8,
        temperature=288.15 * u.K,
        alpha=2.0 * u.deg,
        beta=0.0 * u.deg,
        project_length_unit=u.m,
    )
    assertions.assertAlmostEqual(condition.thermal_state.dynamic_viscosity.value, 1.78929763e-5)
    assertions.assertAlmostEqual(condition.thermal_state.density.value, 1.31452332)

    assertions.assertAlmostEqual(
        condition.flow360_reynolds_number(length_unit=1 * u.m), (1.0 / 4e-8) * condition.mach
    )  # 1/muRef * freestream mach

    with pytest.raises(ValueError, match="Input should be greater than 0"):
        condition = operating_condition_from_mach_muref(
            mach=0.2,
            mu_ref=0,
            temperature=288.15 * u.K,
        )


def test_delta_temperature_scaling():
    with CGS_unit_system:
        param = SimulationParams(
            operating_condition=AerospaceCondition(
                thermal_state=ThermalState.from_standard_atmosphere(
                    temperature_offset=123 * u.delta_degF
                )
            )
        )
    reference_temperature = param.operating_condition.thermal_state.temperature.to("K")

    scaled_temperature_offset = (123 * u.delta_degF / reference_temperature).value
    processed_param = param._preprocess(mesh_unit=1 * u.m)

    assert (
        processed_param.operating_condition.thermal_state.temperature_offset.value
        == scaled_temperature_offset
    )
