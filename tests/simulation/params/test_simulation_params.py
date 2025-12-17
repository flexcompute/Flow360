import json
import unittest

import numpy as np
import pytest

import flow360.component.simulation.units as u
from flow360.component.project import create_draft
from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.simulation import services
from flow360.component.simulation.conversion import LIQUID_IMAGINARY_FREESTREAM_MACH
from flow360.component.simulation.entity_info import (
    GeometryEntityInfo,
    VolumeMeshEntityInfo,
)
from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.meshing_param import snappy
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
    ModularMeshingWorkflow,
)
from flow360.component.simulation.meshing_param.volume_params import (
    CustomZones,
    UniformRefinement,
)
from flow360.component.simulation.migration.extra_operating_condition import (
    operating_condition_from_mach_muref,
)
from flow360.component.simulation.models.material import SolidMaterial, Water
from flow360.component.simulation.models.surface_models import (
    Freestream,
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
    LiquidOperatingCondition,
    ThermalState,
)
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    Edge,
    GenericVolume,
    ReferenceGeometry,
    SeedpointVolume,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Unsteady
from flow360.component.simulation.translator.surface_meshing_translator import (
    _inject_body_group_transformations_for_mesher,
)
from flow360.component.simulation.unit_system import CGS_unit_system, SI_unit_system
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)
from flow360.component.simulation.utils import model_attribute_unlock
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


@pytest.fixture()
def get_param_with_liquid_operating_condition():
    with SI_unit_system:
        water = Water(
            name="h2o", density=1000 * u.kg / u.m**3, dynamic_viscosity=0.001 * u.kg / u.m / u.s
        )
        param = SimulationParams(
            operating_condition=LiquidOperatingCondition(
                velocity_magnitude=50,
                reference_velocity_magnitude=100,
                material=water,
            ),
            models=[
                Fluid(material=water),
                Wall(
                    entities=[Surface(name="wall1")],
                    velocity=(1.0, 1.2, 2.4),
                ),
                Rotation(
                    volumes=[
                        Cylinder(
                            name="cyl-1",
                            axis=(5, 0, 0),
                            center=(1.2, 2.3, 3.4),
                            height=3.0,
                            inner_radius=3.0,
                            outer_radius=5.0,
                        )
                    ],
                    spec=AngularVelocity(0.45 * u.rad / u.s),
                ),
                Freestream(
                    entities=Surface(name="my_fs"),
                    turbulence_quantities=TurbulenceQuantities(
                        modified_viscosity=10,
                    ),
                ),
            ],
            time_stepping=Unsteady(step_size=2 * 0.2 * u.s, steps=123),
        )
        return param


@pytest.fixture()
def get_param_with_list_of_lengths():
    with SI_unit_system:
        params = SimulationParams(
            meshing=ModularMeshingWorkflow(
                surface_meshing=snappy.SurfaceMeshingParams(
                    defaults=snappy.SurfaceMeshingDefaults(
                        min_spacing=10 * u.mm, max_spacing=2 * u.m, gap_resolution=0.01 * u.m
                    ),
                    refinements=[
                        snappy.SurfaceEdgeRefinement(
                            spacing=[1e-3, 8] * u.m,
                            distances=[0.4 * u.mm, 2 * u.m],
                            entities=[Surface(name="test")],
                        )
                    ],
                ),
                zones=[
                    CustomZones(
                        name="custom_zones",
                        entities=[SeedpointVolume(name="fluid", point_in_mesh=[1, 1, 1])],
                    )
                ],
            ),
            operating_condition=AerospaceCondition.from_mach(
                mach=0.8,
                alpha=30 * u.deg,
                beta=20 * u.deg,
                thermal_state=ThermalState(temperature=300 * u.K, density=1 * u.g / u.cm**3),
                reference_mach=0.5,
            ),
        )
    return params


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


@pytest.mark.usefixtures("array_equality_override")
def test_simulation_params_unit_conversion_with_list_of_lengths(get_param_with_list_of_lengths):
    converted = get_param_with_list_of_lengths._preprocess(mesh_unit=10 * u.m)

    assertions.assertAlmostEqual(converted.meshing.surface_meshing.defaults.min_spacing.value, 1e-3)
    assertions.assertAlmostEqual(converted.meshing.surface_meshing.defaults.max_spacing, 0.2)

    assertions.assertAlmostEqual(converted.meshing.surface_meshing.defaults.gap_resolution, 1e-3)

    assertions.assertAlmostEqual(converted.meshing.surface_meshing.refinements[0].spacing[0], 1e-4)

    assertions.assertAlmostEqual(converted.meshing.surface_meshing.refinements[0].spacing[1], 0.8)

    assertions.assertAlmostEqual(
        converted.meshing.surface_meshing.refinements[0].distances[0], 4e-5
    )

    assertions.assertAlmostEqual(converted.meshing.surface_meshing.refinements[0].distances[1], 0.2)


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
    condition = AerospaceCondition.from_mach_reynolds(
        mach=0.2,
        reynolds_mesh_unit=5e6,
        temperature=288.15 * u.K,
        alpha=2.0 * u.deg,
        beta=0.0 * u.deg,
        project_length_unit=u.m,
    )
    assertions.assertAlmostEqual(condition.thermal_state.dynamic_viscosity.value, 1.78929763e-5)
    assertions.assertAlmostEqual(condition.thermal_state.density.value, 1.31452332)

    condition = AerospaceCondition.from_mach_reynolds(
        mach=0.2,
        reynolds_mesh_unit=5e6,
        temperature=288.15 * u.K,
        alpha=2.0 * u.deg,
        beta=0.0 * u.deg,
        project_length_unit=u.m,
        reference_mach=0.4,
    )
    assertions.assertAlmostEqual(condition.thermal_state.density.value, 1.31452332)

    with pytest.raises(ValueError, match="Input should be greater than 0"):
        condition = AerospaceCondition.from_mach_reynolds(
            mach=0.2,
            reynolds_mesh_unit=0,
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


def test_simulation_params_unit_conversion_with_liquid_condition(
    get_param_with_liquid_operating_condition,
):
    params: SimulationParams = get_param_with_liquid_operating_condition
    converted = params._preprocess(mesh_unit=1 * u.m)

    fake_water_speed_of_sound = (
        params.operating_condition.velocity_magnitude / LIQUID_IMAGINARY_FREESTREAM_MACH
    )
    # TimeType
    assertions.assertAlmostEqual(
        converted.time_stepping.step_size.value,
        params.time_stepping.step_size / (1.0 / fake_water_speed_of_sound),
    )

    # VelocityType
    assertions.assertAlmostEqual(
        converted.models[1].velocity.value[0],
        1.0 / fake_water_speed_of_sound,
    )

    # AngularVelocityType
    assertions.assertAlmostEqual(
        converted.models[2].spec.value.value,
        0.45 / fake_water_speed_of_sound.value,
        # Note: We did not use original value from params like the others b.c
        # for some unknown reason THIS value in params will also converted to flow360 unit system...
    )
    # ViscosityType
    assertions.assertAlmostEqual(
        converted.models[3].turbulence_quantities.modified_turbulent_viscosity.value,
        10 / (1 * fake_water_speed_of_sound.value),
        # Note: We did not use original value from params like the others b.c
        # for some unknown reason THIS value in params will also converted to flow360 unit system...
    )


def test_persistent_entity_info_update_geometry():
    #### Geometry Entity Info ####
    with open("./data/geometry_metadata_asset_cache.json", "r") as fp:
        geometry_info_dict = json.load(fp)["project_entity_info"]
        geometry_info = GeometryEntityInfo.model_validate(geometry_info_dict)
    # modify a surface
    selected_surface = geometry_info.get_boundaries()[0]
    selected_surface_dict = selected_surface.model_dump(mode="json")
    selected_surface_dict["name"] = "new_surface_name"
    brand_new_surface = Surface.model_validate(selected_surface_dict)
    # modify an edge
    selected_edge = geometry_info._get_list_of_entities(entity_type_name="edge")[0]
    selected_edge_dict = selected_edge.model_dump(mode="json")
    selected_edge_dict["name"] = "new_edge_name"
    brand_new_edge = Edge.model_validate(selected_edge_dict)

    fake_asset_entity_registry = EntityRegistry()
    fake_asset_entity_registry.register(brand_new_surface)
    fake_asset_entity_registry.register(brand_new_edge)

    geometry_info.update_persistent_entities(asset_entity_registry=fake_asset_entity_registry)

    assert geometry_info.get_boundaries()[0].name == "new_surface_name"
    assert geometry_info._get_list_of_entities(entity_type_name="edge")[0].name == "new_edge_name"


def test_persistent_entity_info_update_volume_mesh():
    #### VolumeMesh Entity Info ####
    with open("./data/volume_mesh_metadata_asset_cache.json", "r") as fp:
        volume_mesh_info_dict = json.load(fp)
        volume_mesh_info = VolumeMeshEntityInfo.model_validate(volume_mesh_info_dict)

    # modify the axis and center of a zone
    selected_zone = volume_mesh_info.zones[0]
    selected_zone_dict = selected_zone.model_dump(mode="json")
    selected_zone_dict["axes"] = [[1, 0, 0], [0, 0, 1]]
    selected_zone_dict["center"] = {"units": "cm", "value": [1.2, 2.3, 3.4]}
    brand_new_zone = GenericVolume.model_validate(selected_zone_dict)

    fake_asset_entity_registry = EntityRegistry()
    fake_asset_entity_registry.register(brand_new_zone)

    volume_mesh_info.update_persistent_entities(asset_entity_registry=fake_asset_entity_registry)

    assert volume_mesh_info.zones[0].axes == ((1, 0, 0), (0, 0, 1))
    assert all(volume_mesh_info.zones[0].center == [1.2, 2.3, 3.4] * u.cm)


def test_geometry_entity_info_to_file_list_and_entity_to_file_map():
    with open("./data/geometry_metadata_asset_cache_mixed_file.json", "r") as fp:
        geometry_entity_info_dict = json.load(fp)
        geometry_entity_info = GeometryEntityInfo.model_validate(geometry_entity_info_dict)

    assert geometry_entity_info._get_processed_file_list() == (
        ["airplane_simple_obtained_from_csm_by_esp.step.egads"],
        ["airplane_translate_in_z_-5.stl", "farfield_only_sphere_volume_mesh.lb8.ugrid"],
    )

    assert sorted(
        geometry_entity_info._get_id_to_file_map(entity_type_name="body").items()
    ) == sorted(
        {
            "body00001": "airplane_simple_obtained_from_csm_by_esp.step.egads",
            "airplane_translate_in_z_-5.stl": "airplane_translate_in_z_-5.stl",
            "farfield_only_sphere_volume_mesh.lb8.ugrid": "farfield_only_sphere_volume_mesh.lb8.ugrid",
        }.items()
    )


def test_geometry_entity_info_get_body_group_to_face_group_name_map():
    with open("./data/geometry_metadata_asset_cache_multiple_bodies.json", "r") as fp:
        geometry_entity_info_dict = json.load(fp)
        geometry_entity_info = GeometryEntityInfo.model_validate(geometry_entity_info_dict)
    assert sorted(geometry_entity_info.get_body_group_to_face_group_name_map().items()) == sorted(
        {
            "cube-holes.egads": ["body00001", "body00002"],
            "cylinder.stl": ["cylinder.stl"],
        }.items()
    )
    geometry_entity_info._group_entity_by_tag("face", "faceId")
    assert sorted(geometry_entity_info.get_body_group_to_face_group_name_map().items()) == sorted(
        {
            "cube-holes.egads": [
                "body00001_face00001",
                "body00001_face00002",
                "body00001_face00003",
                "body00001_face00004",
                "body00001_face00005",
                "body00001_face00006",
                "body00002_face00001",
                "body00002_face00002",
                "body00002_face00003",
                "body00002_face00004",
                "body00002_face00005",
                "body00002_face00006",
            ],
            "cylinder.stl": ["cylinder.stl_body"],
        }.items()
    )


def test_transformation_matrix(mock_geometry):
    """
    End-to-end test: DraftContext coordinate system assignment -> params setup -> mesher JSON injection.
    """

    def _get_selected_grouped_bodies(entity_info_dict: dict) -> list[dict]:
        grouped_bodies = entity_info_dict.get("grouped_bodies", None)
        assert isinstance(grouped_bodies, list)

        body_group_tag = entity_info_dict.get("body_group_tag")
        body_attribute_names = entity_info_dict.get("body_attribute_names")
        assert isinstance(body_group_tag, str)
        assert isinstance(body_attribute_names, list)
        assert body_group_tag in body_attribute_names

        selected_group_index = body_attribute_names.index(body_group_tag)
        selected_group = grouped_bodies[selected_group_index]
        assert isinstance(selected_group, list)
        assert selected_group
        return selected_group

    # Prefer a body grouping that yields multiple GeometryBodyGroup entities (if available),
    # so we can validate identity matrices for unassigned body groups too.
    entity_info = mock_geometry.entity_info
    candidate_body_group_tag = None
    if getattr(entity_info, "body_attribute_names", None):
        for tag in ("groupByBodyId", "bodyId"):
            if tag in entity_info.body_attribute_names:
                candidate_body_group_tag = tag
                break
    if candidate_body_group_tag is not None:
        with model_attribute_unlock(entity_info, "body_group_tag"):
            entity_info.body_group_tag = candidate_body_group_tag

    with create_draft(new_run_from=mock_geometry) as draft:
        body_groups = list(draft.body_groups)
        target_body_group = body_groups[0]

        with SI_unit_system:
            # Nested coordinate system test:
            # Parent: Translate [10, 0, 0]
            # Child: Parent + Translate [0, 5, 0] -> Total [10, 5, 0]
            cs_parent = CoordinateSystem(name="parent", translation=[10, 0, 0] * u.m)
            cs_child = CoordinateSystem(name="child", translation=[0, 5, 0] * u.m)

        cs_parent = draft.coordinate_systems.add(coordinate_system=cs_parent)
        cs_child = draft.coordinate_systems.add(coordinate_system=cs_child, parent=cs_parent)
        draft.coordinate_systems.assign(entities=target_body_group, coordinate_system=cs_child)

        with SI_unit_system:
            params = SimulationParams(operating_condition=AerospaceCondition())

        processed_params = set_up_params_for_uploading(
            root_asset=mock_geometry,
            length_unit=1 * u.m,
            params=params,
            use_beta_mesher=False,
            use_geometry_AI=False,
        )

    json_data = processed_params.model_dump(mode="json", exclude_none=True)
    _inject_body_group_transformations_for_mesher(
        json_data=json_data, input_params=processed_params, mesh_unit=1 * u.m
    )

    entity_info_dict = json_data["private_attribute_asset_cache"]["project_entity_info"]
    selected_group = _get_selected_grouped_bodies(entity_info_dict=entity_info_dict)

    expected_child_matrix = [
        1.0,
        0.0,
        0.0,
        10.0,
        0.0,
        1.0,
        0.0,
        5.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]

    target_body_group_dict = None
    for body_group_dict in selected_group:
        if (
            body_group_dict.get("private_attribute_entity_type_name")
            == target_body_group.private_attribute_entity_type_name
            and body_group_dict.get("private_attribute_id")
            == target_body_group.private_attribute_id
        ):
            target_body_group_dict = body_group_dict

    assert isinstance(target_body_group_dict, dict)

    # All bodies in the selected grouping must have an explicit matrix payload.
    # The transformation object should only contain private_attribute_matrix (no redundant fields).
    for body_group_dict in selected_group:
        transformation = body_group_dict.get("transformation")
        assert isinstance(transformation, dict)
        assert list(transformation.keys()) == ["private_attribute_matrix"], (
            f"Expected transformation to only contain 'private_attribute_matrix', "
            f"but got keys: {list(transformation.keys())}"
        )
        assert isinstance(transformation.get("private_attribute_matrix"), list)
        assert len(transformation["private_attribute_matrix"]) == 12

    assert (
        target_body_group_dict["transformation"]["private_attribute_matrix"]
        == expected_child_matrix
    )

    # Unassigned body groups (if any) default to identity.
    for body_group_dict in selected_group:
        if body_group_dict is target_body_group_dict:
            continue
        assert body_group_dict["transformation"]["private_attribute_matrix"] == [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]


def test_default_params_for_local_test():
    # Test to ensure the default params for local test is validated
    with SI_unit_system:
        param = SimulationParams()

    param = services._store_project_length_unit(1 * u.m, param)
    param_as_dict = param.model_dump(
        exclude_none=True,
        exclude={
            "operating_condition": {"velocity_magnitude": True},
            "private_attribute_asset_cache": {"registry": True},
        },
    )

    with SI_unit_system:
        SimulationParams(**param_as_dict)
